#include <sys/stat.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "spirv_patcher.hpp"
#include "wrapper_entrypoints.h"
#include "wrapper_trampolines.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_dispatch_table.h"
#include "vk_extensions.h"
#include "vk_queue.h"
#include "vk_util.h"
#include "util/list.h"
#include "util/simple_mtx.h"

const struct vk_device_extension_table wrapper_device_extensions =
{
   /* Wrapper-owned WSI extensions: handled by the wrapper, not forwarded. */
   .KHR_swapchain = true,
   .EXT_swapchain_maintenance1 = true,
   .KHR_swapchain_mutable_format = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
   .EXT_display_control = true,
#endif
   .KHR_present_id = true,
   .KHR_present_wait = true,
   .KHR_incremental_present = true,

   /* Extensions the wrapper itself must enable in the underlying driver. */
   .KHR_external_memory = true,
   .KHR_external_memory_fd = true,
   .EXT_external_memory_dma_buf = true,
   .ANDROID_external_memory_android_hardware_buffer = true,
   .KHR_buffer_device_address = true,
   .KHR_vulkan_memory_model = true,
   .KHR_timeline_semaphore = true,
   /* App requests are skipped here because wrapper_CreateDevice appends it. */
   .EXT_device_fault = true,
};

/* Only these extensions are implemented by the wrapper's WSI layer and may
 * be advertised without the underlying Mali driver exposing them. */
const struct vk_device_extension_table wrapper_wsi_extensions =
{
   .KHR_swapchain = true,
   .EXT_swapchain_maintenance1 = true,
   .KHR_swapchain_mutable_format = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
   .EXT_display_control = true,
#endif
   .KHR_present_id = true,
   .KHR_present_wait = true,
   .KHR_incremental_present = true,
};


/*
 * Extensions device creation *requires* unconditionally from the underlying
 * driver -- checked and force-enabled explicitly in wrapper_CreateDevice()
 * (buffer_device_address), or conditionally required + force-enabled for
 * Mali Valhall parts with more than one shader core (the dynamic-rendering
 * set -- see the Mali Valhall block below). Skipped by the blanket sweep;
 * added to the enable list by their own dedicated code paths instead.
 */
const struct vk_device_extension_table wrapper_mandatory_extensions =
{
   .KHR_buffer_device_address = true,
   .KHR_dynamic_rendering = true,
   .KHR_dynamic_rendering_local_read = true,
   .EXT_dynamic_rendering_unused_attachments = true,
   .KHR_multiview = true,
};

const struct vk_device_extension_table wrapper_filter_extensions =
{
   .EXT_hdr_metadata = true,
   .GOOGLE_display_timing = true,
   .KHR_shared_presentable_image = true,
   .EXT_image_compression_control_swapchain = true,
};
/*
 * ---------------------------------------------------------------------
 * Mali Valhall enforcement
 * ---------------------------------------------------------------------
 * GPU generation and shader-core count come from kbase (/dev/mali0, see
 * wrapper_mali.c); the Vulkan driver's deviceID/deviceName are only the
 * fallback when kbase can't be opened. This wrapper only ever runs on Mali
 * (enumerate_physical_device rejects every other GPU).
 *
 * Valhall parts with more than one shader core want dynamic rendering (plus
 * local read / unused attachments) and multiview. Whatever the driver can
 * really back is forced on; whatever it can't is logged and skipped -- a
 * missing optional extension must never make vkCreateDevice fail.
 * ---------------------------------------------------------------------
 */

struct wrapper_valhall_plan {
   bool enforce;
   bool dynamic_rendering;
   bool local_read;
   bool unused_attachments;
   bool multiview;
};

static void
wrapper_valhall_plan_init(struct wrapper_physical_device *pdevice,
                          struct wrapper_valhall_plan *plan)
{
   const struct wrapper_mali_info *mali = &pdevice->mali;
   const struct vk_features *f = &pdevice->base_supported_features;
   const struct vk_device_extension_table *e = &pdevice->base_supported_extensions;
   const uint32_t api = pdevice->properties2.properties.apiVersion;

   memset(plan, 0, sizeof(*plan));

   if (!mali->is_valhall || mali->core_count < 2)
      return;

   plan->enforce = true;
   plan->dynamic_rendering = f->dynamicRendering &&
      (e->KHR_dynamic_rendering || api >= VK_API_VERSION_1_3);
   plan->local_read = f->dynamicRenderingLocalRead && e->KHR_dynamic_rendering_local_read;
   plan->unused_attachments = f->dynamicRenderingUnusedAttachments &&
      e->EXT_dynamic_rendering_unused_attachments;
   plan->multiview = f->multiview && (e->KHR_multiview || api >= VK_API_VERSION_1_1);

   WRAPPER_LOG(info,
      "Mali Valhall GPU_ID 0x%08x (arch v%u, %u shader cores, %s) -- "
      "dynamic_rendering=%d local_read=%d unused_attachments=%d multiview=%d",
      mali->gpu_id, mali->arch_major, mali->core_count,
      mali->core_count_from_kbase ? "from kbase" : "from device name",
      plan->dynamic_rendering, plan->local_read,
      plan->unused_attachments, plan->multiview);

   if (!plan->dynamic_rendering || !plan->local_read ||
       !plan->unused_attachments || !plan->multiview) {
      WRAPPER_LOG(error,
         "Mali Valhall MC2+ driver lacks part of the dynamic-rendering set; "
         "enabling only what it supports");
   }
}

static void
wrapper_append_valhall_extensions(const struct wrapper_valhall_plan *plan,
                                  const struct wrapper_physical_device *pdevice,
                                  uint32_t *count, const char **exts)
{
   const struct vk_device_extension_table *e = &pdevice->base_supported_extensions;

   /* only names the driver really lists: promoted-to-core features are
    * enabled through their feature bits, not by an extension string */
   if (plan->dynamic_rendering && e->KHR_dynamic_rendering)
      exts[(*count)++] = "VK_KHR_dynamic_rendering";
   if (plan->local_read && e->KHR_dynamic_rendering_local_read)
      exts[(*count)++] = "VK_KHR_dynamic_rendering_local_read";
   if (plan->unused_attachments && e->EXT_dynamic_rendering_unused_attachments)
      exts[(*count)++] = "VK_EXT_dynamic_rendering_unused_attachments";
   if (plan->multiview && e->KHR_multiview)
      exts[(*count)++] = "VK_KHR_multiview";
}

inline struct wrapper_buffer *
get_wrapper_buffer_from_handle_locked(struct wrapper_device *device, VkBuffer buffer) {
   return _mesa_hash_table_u64_search(device->buffer_table, (uint64_t) buffer);
}

struct wrapper_buffer *
get_wrapper_buffer_from_handle(struct wrapper_device *device, VkBuffer buffer) {
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle_locked(device, buffer);
   simple_mtx_unlock(&device->resource_mutex);

   return wb;
}

inline struct wrapper_image *
get_wrapper_image_from_handle_locked(struct wrapper_device *device, VkImage image) {
   return _mesa_hash_table_u64_search(device->image_table, (uint64_t) image);
}

struct wrapper_image *
get_wrapper_image_from_handle(struct wrapper_device *device, VkImage image) {   
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_image *wi = get_wrapper_image_from_handle_locked(device, image);
   simple_mtx_unlock(&device->resource_mutex);
   
   return wi;
}

static void
wrapper_filter_enabled_extensions(const struct wrapper_device *device,
                                  uint32_t *enable_extension_count,
                                  const char **enable_extensions)
{
   for (int idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
      if (!device->vk.enabled_extensions.extensions[idx])
         continue;

      if (!device->physical->base_supported_extensions.extensions[idx])
         continue;

      if (wrapper_device_extensions.extensions[idx])
         continue;

      if (wrapper_mandatory_extensions.extensions[idx])
         continue;

      if (wrapper_filter_extensions.extensions[idx])
         continue;

      enable_extensions[(*enable_extension_count)++] =
         vk_device_extensions[idx].extensionName;
   }

}


static inline void
wrapper_append_required_extensions(const struct vk_device *device,
                                  uint32_t *count,
                                  const char **exts)
{
   const struct wrapper_physical_device *pdevice =
      (const struct wrapper_physical_device *)device->physical;

   /* These are used by the wrapper's AHardwareBuffer / dma-buf memory path.
    * Do not force unrelated driver extensions: applications can still request
    * any supported pass-through extension through wrapper_filter_enabled_extensions(). */
#define REQUIRED_EXTENSION(name) \
   do { \
      if (device->physical->supported_extensions.name && \
          pdevice->base_supported_extensions.name) \
         exts[(*count)++] = "VK_" #name; \
   } while (0)

   REQUIRED_EXTENSION(KHR_external_memory);
   REQUIRED_EXTENSION(KHR_external_memory_fd);
   REQUIRED_EXTENSION(EXT_external_memory_dma_buf);
   REQUIRED_EXTENSION(ANDROID_external_memory_android_hardware_buffer);

   /* These feature extensions are enabled by wrapper-owned feature paths. */
   REQUIRED_EXTENSION(KHR_buffer_device_address);
   REQUIRED_EXTENSION(KHR_vulkan_memory_model);
   REQUIRED_EXTENSION(KHR_timeline_semaphore);
#undef REQUIRED_EXTENSION
}

static void unlink_vk_struct(VkBaseInStructure *create_info, const VkBaseInStructure **current, VkBaseInStructure **prev) {
   if (!*prev) 
      create_info->pNext = (*current)->pNext;
   else
      (*prev)->pNext = (*current)->pNext;                                                

   *current = (*current)->pNext;
}

static void process_pnext_chain(VkBaseInStructure *create_info, struct wrapper_physical_device *pdevice) {
   const uint32_t api_version = pdevice->properties2.properties.apiVersion;
   const VkBaseInStructure *current = (VkBaseInStructure *)create_info->pNext;
   VkBaseInStructure *prev = NULL;

   while (current != NULL) {
      switch(current->sType) {
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT: {
             VkPhysicalDeviceTransformFeedbackFeaturesEXT *transform_features =
                (VkPhysicalDeviceTransformFeedbackFeaturesEXT *)current;
             transform_features->geometryStreams &= pdevice->base_supported_features.geometryStreams;
             break;
          }
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_robustness2)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceRobustness2FeaturesEXT from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_dynamic_rendering_unused_attachments)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES:
             if (pdevice->base_supported_extensions.KHR_maintenance5)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceMaintenance5Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_vertex_attribute_divisor)
                break;
             WRAPPER_LOG(info, "Unlinking VertexAttributeDivisorFeaturesEXT (driver lacks the EXT extension)");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 1, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan11Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 2, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan12Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
             if (api_version < VK_MAKE_VERSION(1, 3, 0)) {
                WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan13Features from pNext chain");
                unlink_vk_struct(create_info, &current, &prev);
                continue;
             }
             /*
              * Driver reports Vulkan 1.3. Rather than trusting the whole
              * struct blindly (or stripping it wholesale like the 1.1/1.2
              * fallback above), clamp every individual bit to what this
              * specific driver actually advertises.
              */
             VkPhysicalDeviceVulkan13Features *features13 =
                (VkPhysicalDeviceVulkan13Features *)current;
#define CLAMP13(f) features13->f &= pdevice->base_supported_features.f
             CLAMP13(robustImageAccess);
             CLAMP13(inlineUniformBlock);
             CLAMP13(descriptorBindingInlineUniformBlockUpdateAfterBind);
             CLAMP13(pipelineCreationCacheControl);
             CLAMP13(privateData);
             CLAMP13(shaderDemoteToHelperInvocation);
             CLAMP13(shaderTerminateInvocation);
             CLAMP13(subgroupSizeControl);
             CLAMP13(computeFullSubgroups);
             CLAMP13(synchronization2);
             CLAMP13(textureCompressionASTC_HDR);
             CLAMP13(shaderZeroInitializeWorkgroupMemory);
             CLAMP13(dynamicRendering);
             CLAMP13(shaderIntegerDotProduct);
             CLAMP13(maintenance4);
#undef CLAMP13
             break;
          }
          default:
             break;
      }
      prev = (VkBaseInStructure *)current;
      current = current->pNext;
   }
}

/*
 * Force VK_KHR_buffer_device_address / core bufferDeviceAddress on. The
 * wrapper treats this feature as mandatory and actually used internally, so
 * it can't just be advertised -- it must be enabled regardless of whether
 * (or how) the app asked for it. If the app's pNext chain already carries a
 * struct with the bit, flip it in place (same "mutate the app's request"
 * pattern already used by the DISABLE_FEATURE macro in wrapper_CreateDevice).
 * If not, splice `storage` into the (already-copied) wrapper create-info
 * chain. `storage` must outlive the driver's vkCreateDevice() call.
 */
static void
wrapper_force_buffer_device_address(VkBaseInStructure *create_info,
                                    VkPhysicalDeviceBufferDeviceAddressFeatures *storage)
{
   VkBaseInStructure *current = (VkBaseInStructure *)create_info->pNext;

   while (current != NULL) {
      if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
         ((VkPhysicalDeviceBufferDeviceAddressFeatures *)current)->bufferDeviceAddress = VK_TRUE;
         return;
      }
      if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
         ((VkPhysicalDeviceVulkan12Features *)current)->bufferDeviceAddress = VK_TRUE;
         return;
      }
      current = (VkBaseInStructure *)current->pNext;
   }

   storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
   storage->bufferDeviceAddress = VK_TRUE;
   storage->pNext = (void *)create_info->pNext;
   create_info->pNext = (VkBaseInStructure *)storage;
}

/*
 * Turn on VK_KHR_vulkan_memory_model's feature bit when the driver actually
 * supports it. Unlike buffer_device_address above this is best-effort, not
 * mandatory: if the driver doesn't report the feature we just leave the
 * chain alone instead of failing device creation.
 */
static void
wrapper_use_vulkan_memory_model(VkBaseInStructure *create_info,
                                struct wrapper_physical_device *pdevice,
                                VkPhysicalDeviceVulkanMemoryModelFeatures *storage)
{
   if (!pdevice->base_supported_features.vulkanMemoryModel)
      return;

   VkBaseInStructure *current = (VkBaseInStructure *)create_info->pNext;

   while (current != NULL) {
      if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES) {
         ((VkPhysicalDeviceVulkanMemoryModelFeatures *)current)->vulkanMemoryModel = VK_TRUE;
         return;
      }
      if (current->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
         ((VkPhysicalDeviceVulkan12Features *)current)->vulkanMemoryModel = VK_TRUE;
         return;
      }
      current = (VkBaseInStructure *)current->pNext;
   }

   storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
   storage->vulkanMemoryModel = VK_TRUE;
   storage->pNext = (void *)create_info->pNext;
   create_info->pNext = (VkBaseInStructure *)storage;
}

/*
 * Force the Valhall dynamic-rendering feature bits on (only those the driver
 * really supports, see wrapper_valhall_plan), with the same "mutate in place
 * if present, splice a struct in if not" pattern as
 * wrapper_force_buffer_device_address() above. dynamicRendering and multiview
 * are also recognized via their core-promoted aggregate structs (Vulkan 1.3 /
 * 1.1 features) to avoid adding a redundant, spec-invalid duplicate struct
 * when the app already requested the core version struct.
 */
static void
wrapper_force_valhall_features(VkBaseInStructure *create_info,
   const struct wrapper_valhall_plan *plan,
   VkPhysicalDeviceDynamicRenderingFeatures *dr_storage,
   VkPhysicalDeviceDynamicRenderingLocalReadFeatures *drlr_storage,
   VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT *dru_storage,
   VkPhysicalDeviceMultiviewFeatures *mv_storage)
{
   bool have_dynamic_rendering = false;
   bool have_local_read = false;
   bool have_unused_attachments = false;
   bool have_multiview = false;

   for (VkBaseInStructure *current = (VkBaseInStructure *)create_info->pNext;
        current != NULL; current = (VkBaseInStructure *)current->pNext) {
      switch (current->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
         if (plan->dynamic_rendering)
            ((VkPhysicalDeviceVulkan13Features *)current)->dynamicRendering = VK_TRUE;
         have_dynamic_rendering = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
         if (plan->multiview)
            ((VkPhysicalDeviceVulkan11Features *)current)->multiview = VK_TRUE;
         have_multiview = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES:
         if (plan->dynamic_rendering)
            ((VkPhysicalDeviceDynamicRenderingFeatures *)current)->dynamicRendering = VK_TRUE;
         have_dynamic_rendering = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES:
         if (plan->local_read)
            ((VkPhysicalDeviceDynamicRenderingLocalReadFeatures *)current)->dynamicRenderingLocalRead = VK_TRUE;
         have_local_read = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT:
         if (plan->unused_attachments)
            ((VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT *)current)->dynamicRenderingUnusedAttachments = VK_TRUE;
         have_unused_attachments = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES:
         if (plan->multiview)
            ((VkPhysicalDeviceMultiviewFeatures *)current)->multiview = VK_TRUE;
         have_multiview = true;
         break;
      default:
         break;
      }
   }

   if (plan->dynamic_rendering && !have_dynamic_rendering) {
      dr_storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
      dr_storage->dynamicRendering = VK_TRUE;
      dr_storage->pNext = (void *)create_info->pNext;
      create_info->pNext = (VkBaseInStructure *)dr_storage;
   }
   if (plan->local_read && !have_local_read) {
      drlr_storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
      drlr_storage->dynamicRenderingLocalRead = VK_TRUE;
      drlr_storage->pNext = (void *)create_info->pNext;
      create_info->pNext = (VkBaseInStructure *)drlr_storage;
   }
   if (plan->unused_attachments && !have_unused_attachments) {
      dru_storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT;
      dru_storage->dynamicRenderingUnusedAttachments = VK_TRUE;
      dru_storage->pNext = (void *)create_info->pNext;
      create_info->pNext = (VkBaseInStructure *)dru_storage;
   }
   if (plan->multiview && !have_multiview) {
      mv_storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
      mv_storage->multiview = VK_TRUE;
      mv_storage->pNext = (void *)create_info->pNext;
      create_info->pNext = (VkBaseInStructure *)mv_storage;
   }
}

static VkResult
wrapper_create_device_queue(struct wrapper_device *device,
                            const VkDeviceCreateInfo* pCreateInfo)
{
   const VkDeviceQueueCreateInfo *create_info;
   struct wrapper_queue *queue;
   VkResult result;

   for (int i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      create_info = &pCreateInfo->pQueueCreateInfos[i];
      for (int j = 0; j < create_info->queueCount; j++) {
         queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!queue)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         if (create_info->flags) {
            device->dispatch_table.GetDeviceQueue2(
               device->dispatch_handle,
               &(VkDeviceQueueInfo2) {
                  .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
                  .flags = create_info->flags,
                  .queueFamilyIndex = create_info->queueFamilyIndex,
                  .queueIndex = j,
               },
               &queue->dispatch_handle);;
         } else {
            device->dispatch_table.GetDeviceQueue(
               device->dispatch_handle, create_info->queueFamilyIndex,
               j, &queue->dispatch_handle);
         }
         queue->device = device;

         result = vk_queue_init(&queue->vk, &device->vk, create_info, j);
         if (result != VK_SUCCESS) {
            vk_free(&device->vk.alloc, queue);
            return result;
         }
      }
   }

   return VK_SUCCESS;
}

static const char *
wrapper_driver_id_str(VkDriverId id)
{
   switch (id) {
   case VK_DRIVER_ID_ARM_PROPRIETARY:         return "ARM (Mali)";
   case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:    return "Qualcomm (Adreno)";
   case VK_DRIVER_ID_SAMSUNG_PROPRIETARY:     return "Samsung (Xclipse)";
   case VK_DRIVER_ID_MESA_TURNIP:             return "Mesa Turnip (Adreno)";
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY: return "Imagination (PowerVR)";
   default:                                   return "other/unknown";
   }
}

/* WRAPPER_DIAG=1 emits a self-contained device-capabilities report — to
 * stderr (so it lands in logcat) and to a file (WRAPPER_DIAG_FILE, or
 * $TMPDIR/wrapper_diag.txt) a user can share. It reports what the device
 * NATIVELY supports vs what the wrapper advertises to the D3D layer, so a
 * failing game on any device can be diagnosed from the report alone. Pair it
 * with app-side VKD3D_DEBUG=warn / DXVK_LOG_LEVEL=info / WINEDEBUG=+vulkan. */
static void
wrapper_emit_diag(struct wrapper_physical_device *pdev,
                  const VkDeviceCreateInfo *ci, VkResult result)
{
   const char *en = getenv("WRAPPER_DIAG");
   if (!en || !atoi(en))
      return;

   /* Report file lives in the container tmp dir, one per game (unique by app
    * name) so multiple games don't clobber each other. WRAPPER_DIAG_FILE
    * overrides the full path. */
   char tag[128], defpath[512];
   /* Prefer the app id passed by the launcher (WRAPPER_DIAG_APPID); fall back
    * to the executable name so the file is still unique per game either way. */
   const char *app = getenv("WRAPPER_DIAG_APPID");
   if (!app || !app[0]) {
      app = pdev->instance->vk.app_info.app_name;
      if (!app || !app[0])
         app = "unknown";
      const char *bslash = strrchr(app, '\\');   /* strip Windows/Unix path */
      if (bslash) app = bslash + 1;
      const char *fslash = strrchr(app, '/');
      if (fslash) app = fslash + 1;
   }
   size_t j = 0;
   for (size_t i = 0; app[i] && j < sizeof(tag) - 1; i++) {
      char c = app[i];
      int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
      tag[j++] = ok ? c : '_';
   }
   tag[j] = 0;
   snprintf(defpath, sizeof(defpath),
            "/data/data/app.gamenative/files/imagefs/usr/tmp/wrapper_diag_%s.txt", tag);
   const char *path = getenv("WRAPPER_DIAG_FILE") ? getenv("WRAPPER_DIAG_FILE") : defpath;

   /* Put EVERYTHING needed to diagnose a failing game in ONE shareable file. For
    * a D3D engine, redirect this process's stdout+stderr into the diag file once,
    * so the vkd3d/DXVK/wine output that actually explains the failure
    * (Heap-too-small, device-lost, format rejects, crashes) is captured in the
    * same file as the wrapper's capability report below — instead of only the
    * wrapper's half. Non-D3D processes (zink infra) stay on stderr/logcat, so the
    * file is just the D3D process. usr/tmp is recreated per launch. */
   {
      const char *e = pdev->instance->vk.app_info.engine_name;
      static int redirected = 0;
      if (!redirected && e && (strstr(e, "DXVK") || strstr(e, "vkd3d"))) {
         redirected = 1;
         int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
         if (fd >= 0) {
            dup2(fd, 1);   /* stdout: vkd3d/wine 'System.out' errors */
            dup2(fd, 2);   /* stderr */
            if (fd > 2) close(fd);
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
         }
      }
   }

#define D(...) fprintf(stderr, "[WRAPPER_DIAG] " __VA_ARGS__)

   const VkPhysicalDeviceProperties *p = &pdev->properties2.properties;
   const char *eng = pdev->instance->vk.app_info.engine_name
      ? pdev->instance->vk.app_info.engine_name : "";
   uint32_t ev = pdev->instance->vk.app_info.engine_version;
   char detected[64];
   if (strstr(eng, "DXVK"))
      snprintf(detected, sizeof(detected), "DXVK %u.%u.%u",
               VK_VERSION_MAJOR(ev), VK_VERSION_MINOR(ev), VK_VERSION_PATCH(ev));
   else if (strstr(eng, "vkd3d"))
      snprintf(detected, sizeof(detected), "vkd3d %u.%u.%u",
               VK_VERSION_MAJOR(ev), VK_VERSION_MINOR(ev), VK_VERSION_PATCH(ev));
   else
      snprintf(detected, sizeof(detected), "native/other");

   D("\n================ WRAPPER DIAGNOSTICS ================\n");
   D("build: %s %s\n", __DATE__, __TIME__);
   D("device: %s\n", p->deviceName);
   D("  driver=%s (driverID=%u)  driverVersion=0x%08x  apiVersion=%u.%u.%u\n",
     wrapper_driver_id_str(pdev->driver_properties.driverID),
     pdev->driver_properties.driverID, p->driverVersion,
     VK_VERSION_MAJOR(p->apiVersion), VK_VERSION_MINOR(p->apiVersion),
     VK_VERSION_PATCH(p->apiVersion));
   D("  vendorID=0x%04x deviceID=0x%04x\n", p->vendorID, p->deviceID);
   D("engine: '%s' -> detected %s (app '%s')\n", eng, detected,
     pdev->instance->vk.app_info.app_name ? pdev->instance->vk.app_info.app_name : "");
   D("--- native capabilities (what THIS device actually has) ---\n");
   D("  textureCompressionBC       : %d\n", pdev->base_supported_features.textureCompressionBC);
   D("  textureCompressionASTC_LDR : %d\n", pdev->base_supported_features.textureCompressionASTC_LDR);
   D("  robustBufferAccess2        : %d\n", pdev->base_supported_features.robustBufferAccess2);
   D("  nullDescriptor             : %d\n", pdev->base_supported_features.nullDescriptor);
   D("  robustImageAccess2         : %d\n", pdev->base_supported_features.robustImageAccess2);
   D("  extendedDynamicState       : %d\n", pdev->base_supported_features.extendedDynamicState);
   D("  extendedDynamicState2      : %d\n", pdev->base_supported_features.extendedDynamicState2);
   D("  dualSrcBlend               : %d\n", pdev->base_supported_features.dualSrcBlend);
   D("  multiDrawIndirect          : %d\n", pdev->base_supported_features.multiDrawIndirect);
   D("  ext EXT_robustness2              : %d\n", pdev->base_supported_extensions.EXT_robustness2);
   D("  ext EXT_vertex_attribute_divisor : %d\n", pdev->base_supported_extensions.EXT_vertex_attribute_divisor);
   D("  ext KHR_vertex_attribute_divisor : %d\n", pdev->base_supported_extensions.KHR_vertex_attribute_divisor);
   {
      /* extensions advertised although the driver does not list them: only the
       * wrapper-implemented ones (WSI, map_memory_placed/map_memory2) may show up */
      unsigned not_in_driver = 0;
      D("--- extensions advertised without driver support ---\n");
      for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
         if (pdev->vk.supported_extensions.extensions[i] &&
             !pdev->base_supported_extensions.extensions[i]) {
            D("  %s%s\n", vk_device_extensions[i].extensionName,
              wrapper_wsi_extensions.extensions[i] ? " (wrapper WSI)" : " (wrapper implemented)");
            not_in_driver++;
         }
      }
      D("  total: %u\n", not_in_driver);
   }
   D("--- Mali / kernel interfaces ---\n");
   D("  kbase %s: ABI %u.%u GPU_ID=0x%08x arch=v%u cores=%u (%s) L2=%uKiB/%uB x%u threads=%u\n",
     WRAPPER_KBASE_NODE, pdev->mali.kbase_uk_major, pdev->mali.kbase_uk_minor,
     pdev->mali.gpu_id, pdev->mali.arch_major, pdev->mali.core_count,
     pdev->mali.kbase_ok ? "kbase" : "kbase unavailable, device name",
     pdev->mali.l2_cache_bytes / 1024, pdev->mali.l2_line_bytes,
     pdev->mali.l2_slices, pdev->mali.max_threads);
   D("  drm: node=%s (%s) driver=%s libdrm=%s prime(import=%d export=%d) dumb=%d\n",
     pdev->drm.fd >= 0 ? pdev->drm.node_path : "none",
     pdev->drm.is_render_node ? "render" : "primary",
     pdev->drm.driver_name[0] ? pdev->drm.driver_name : "?",
     pdev->drm.libdrm_path[0] ? pdev->drm.libdrm_path : "not loaded",
     pdev->drm.prime_import, pdev->drm.prime_export, pdev->drm.dumb_buffer);
   D("  memory backend: AHardwareBuffer (zero-copy) -> DRM dma-buf only for dma-buf-only resources\n");
   D("  persistent pipeline cache: %s\n",
     (getenv("WRAPPER_PIPELINE_CACHE") && !atoi(getenv("WRAPPER_PIPELINE_CACHE"))) ? "off" : "on");
   D("  VK_EXT_device_fault report    : %s\n",
     !pdev->base_supported_extensions.EXT_device_fault ? "unsupported by base driver" :
     (!getenv("WRAPPER_DEVICE_FAULT") || atoi(getenv("WRAPPER_DEVICE_FAULT")))
        ? "ON (GPU fault dumped on device loss)" : "off (WRAPPER_DEVICE_FAULT=0)");
   D("--- vkCreateDevice ---\n");
   D("  result: %d (%s)\n", result, result == VK_SUCCESS ? "VK_SUCCESS" : "FAILED");
   D("  requested device extensions (%u):\n", ci ? ci->enabledExtensionCount : 0);
   if (ci)
      for (uint32_t i = 0; i < ci->enabledExtensionCount; i++)
         D("    %s\n", ci->ppEnabledExtensionNames[i]);
   D("NOTE: seeing this block means a VkDevice was created. If a game fails\n");
   D("  and this block never appears, the D3D layer rejected caps BEFORE\n");
   D("  device creation -- the VKD3D_DEBUG/DXVK logs will show which.\n");
   D("full diagnosis also needs (app-side): VKD3D_DEBUG=warn DXVK_LOG_LEVEL=info WINEDEBUG=+vulkan\n");
   D("====================================================\n");

#undef D
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDevice(VkPhysicalDevice physicalDevice,
                     const VkDeviceCreateInfo* pCreateInfo,
                     const VkAllocationCallbacks* pAllocator,
                     VkDevice* pDevice)
{
   VK_FROM_HANDLE(wrapper_physical_device, physical_device, physicalDevice);
   const char *wrapper_enable_extensions[VK_DEVICE_EXTENSION_COUNT];
   uint32_t wrapper_enable_extension_count = 0;
   VkDeviceCreateInfo wrapper_create_info = *pCreateInfo;
   struct vk_device_dispatch_table dispatch_table;
   struct wrapper_device *device;
   VkPhysicalDeviceFeatures2 *pdf2;
   VkPhysicalDeviceFeatures *pdf;
   VkResult result;
   static int wrapper_safe_create_device = -1;
   static int wrapper_device_fault = -1;
   VkPhysicalDeviceFaultFeaturesEXT fault_features_ext = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
   };
   bool used_fallback_create = false;

   /*
    * bufferDeviceAddress is switched on whenever the driver can back it, but it
    * must never make vkCreateDevice fail: Arm's 32-bit driver build has no
    * 64-bit GPU VA space and does not expose it, which used to turn every
    * 32-bit process into VK_ERROR_FEATURE_NOT_PRESENT.
    */
   const bool enable_bda =
      physical_device->base_supported_features.bufferDeviceAddress &&
      (physical_device->base_supported_extensions.KHR_buffer_device_address ||
       physical_device->properties2.properties.apiVersion >= VK_API_VERSION_1_2);
   if (!enable_bda)
      WRAPPER_LOG(info, "Driver has no bufferDeviceAddress, continuing without it");

   /*
    * GPU/driver identification -- read straight off the device, never
    * assumed -- decides whether this is a Mali Valhall part with more than
    * one shader core, in which case dynamic rendering + multiview become
    * mandatory too.
    */
   struct wrapper_valhall_plan valhall_plan;
   wrapper_valhall_plan_init(physical_device, &valhall_plan);

   device = vk_zalloc2(&physical_device->instance->vk.alloc, pAllocator,
                       sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&device->command_buffer_list);
   list_inithead(&device->device_memory_list);
   list_inithead(&device->image_list);
   list_inithead(&device->buffer_list);
   device->image_table = _mesa_hash_table_u64_create(NULL);
   device->buffer_table = _mesa_hash_table_u64_create(NULL);
   device->memory_table = _mesa_hash_table_u64_create(NULL);
   
   simple_mtx_init(&device->resource_mutex, mtx_plain);
   device->physical = physical_device;

   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wrapper_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wrapper_device_trampolines, false);

   result = vk_device_init(&device->vk, &physical_device->vk,
                           &dispatch_table, pCreateInfo, pAllocator);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to init Vulkan device, res %d", result);
      _mesa_hash_table_u64_destroy(device->image_table);
      _mesa_hash_table_u64_destroy(device->buffer_table);
      _mesa_hash_table_u64_destroy(device->memory_table);
      simple_mtx_destroy(&device->resource_mutex);
      vk_free2(&physical_device->instance->vk.alloc, pAllocator,
               device);
      return vk_error(physical_device, result);
   }

   wrapper_filter_enabled_extensions(device,
      &wrapper_enable_extension_count, wrapper_enable_extensions);
   wrapper_append_required_extensions(&device->vk,
      &wrapper_enable_extension_count, wrapper_enable_extensions);
   if (valhall_plan.enforce) {
      wrapper_append_valhall_extensions(&valhall_plan, physical_device,
         &wrapper_enable_extension_count, wrapper_enable_extensions);
   }

   /* VK_EXT_device_fault turns the generic VK_ERROR_DEVICE_LOST into an actual
    * GPU fault report (faulting address + vendor fault codes) that we dump in
    * QueueSubmit. It's universally available on Mali and cheap when no fault
    * occurs, so enable it by default whenever the base driver supports it;
    * WRAPPER_DEVICE_FAULT=0 opts out. */
   if (wrapper_device_fault == -1)
      wrapper_device_fault = getenv("WRAPPER_DEVICE_FAULT")
         ? atoi(getenv("WRAPPER_DEVICE_FAULT")) : 1;

   bool enable_device_fault = wrapper_device_fault &&
      physical_device->base_supported_extensions.EXT_device_fault;

   if (enable_device_fault)
      wrapper_enable_extensions[wrapper_enable_extension_count++] = "VK_EXT_device_fault";

   wrapper_create_info.enabledExtensionCount = wrapper_enable_extension_count;
   wrapper_create_info.ppEnabledExtensionNames = wrapper_enable_extensions;
   
   pdf = (void *)pCreateInfo->pEnabledFeatures;
   pdf2 = __vk_find_struct((void *)pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            
#define DISABLE_FEATURE(f) \
if (pdf && pdf->f) { \
   pdf->f &= physical_device->base_supported_features.f; \
} \
\
if (pdf2 && pdf2->features.f) { \
   pdf2->features.f &= physical_device->base_supported_features.f; \
}

   DISABLE_FEATURE(textureCompressionBC);
   DISABLE_FEATURE(multiViewport);
   DISABLE_FEATURE(depthClamp);
   DISABLE_FEATURE(depthBiasClamp);
   DISABLE_FEATURE(fillModeNonSolid);
   DISABLE_FEATURE(shaderClipDistance);
   DISABLE_FEATURE(shaderCullDistance);
   DISABLE_FEATURE(dualSrcBlend);
   DISABLE_FEATURE(multiDrawIndirect);

#undef DISABLE_FEATURE

   process_pnext_chain((VkBaseInStructure *)&wrapper_create_info, device->physical);

   /* These storage structs must stay alive through the driver
    * CreateDevice() call below, so they live in this stack frame rather
    * than inside the helper functions. */
   VkPhysicalDeviceBufferDeviceAddressFeatures wrapper_bda_features = {0};
   if (enable_bda)
      wrapper_force_buffer_device_address(
         (VkBaseInStructure *)&wrapper_create_info, &wrapper_bda_features);

   VkPhysicalDeviceVulkanMemoryModelFeatures wrapper_vmm_features = {0};
   wrapper_use_vulkan_memory_model(
      (VkBaseInStructure *)&wrapper_create_info, device->physical,
      &wrapper_vmm_features);

   VkPhysicalDeviceDynamicRenderingFeatures wrapper_dr_features = {0};
   VkPhysicalDeviceDynamicRenderingLocalReadFeatures wrapper_drlr_features = {0};
   VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT wrapper_dru_features = {0};
   VkPhysicalDeviceMultiviewFeatures wrapper_mv_features = {0};

   if (valhall_plan.enforce) {
      wrapper_force_valhall_features(
         (VkBaseInStructure *)&wrapper_create_info, &valhall_plan,
         &wrapper_dr_features, &wrapper_drlr_features,
         &wrapper_dru_features, &wrapper_mv_features);
   }

   /* Request the deviceFault feature. Only inject our struct if the client
    * didn't already provide one (it manages its own if so). */
   if (enable_device_fault &&
       !vk_find_struct_const(wrapper_create_info.pNext, PHYSICAL_DEVICE_FAULT_FEATURES_EXT)) {
      WRAPPER_LOG(info, "Enabling VK_EXT_device_fault for GPU fault reporting");
      fault_features_ext.deviceFault =
         physical_device->base_supported_features.deviceFault;
      fault_features_ext.deviceFaultVendorBinary =
         physical_device->base_supported_features.deviceFaultVendorBinary;
      fault_features_ext.pNext = (void *)wrapper_create_info.pNext;
      wrapper_create_info.pNext = &fault_features_ext;
   }

   if (WRAPPER_LOG_LEVEL(info)) {
      for (int i = 0; i < wrapper_enable_extension_count; i++) {
         WRAPPER_LOG(info, "Enabling device extension %s", wrapper_enable_extensions[i]);
      }
   }

   if (wrapper_safe_create_device == -1) {
      wrapper_safe_create_device = getenv("WRAPPER_SAFE_CREATE_DEVICE") ? atoi(getenv("WRAPPER_SAFE_CREATE_DEVICE")) : 1;
   }
   
   result = physical_device->dispatch_table.CreateDevice(
      physical_device->dispatch_handle, &wrapper_create_info,
         pAllocator, &device->dispatch_handle);

   if (result != VK_SUCCESS) {
      if (wrapper_safe_create_device) {
         WRAPPER_LOG(info, "Forcing device creation with a NULL pNext chain");
         wrapper_create_info.pNext = NULL;
         used_fallback_create = true;
         result = physical_device->dispatch_table.CreateDevice(
            physical_device->dispatch_handle, &wrapper_create_info,
               pAllocator, &device->dispatch_handle);
      }
      
      if (result != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed driver createDevice, res %d", result);
         wrapper_emit_diag(physical_device, pCreateInfo, result);
         wrapper_DestroyDevice(wrapper_device_to_handle(device),
                               &device->vk.alloc);
         return vk_error(physical_device, result);
      }
   }

   void *gdpa = physical_device->instance->dispatch_table.GetInstanceProcAddr(
      physical_device->instance->dispatch_handle, "vkGetDeviceProcAddr");
   vk_device_dispatch_table_load(&device->dispatch_table, gdpa,
                                 device->dispatch_handle);

   /* The fallback create with a NULL pNext drops the deviceFault feature, so
    * only treat fault reporting as usable when the primary create succeeded. */
   device->device_fault_enabled = enable_device_fault && !used_fallback_create;

   /* Persistent pipeline cache (Mali compiles are slow): see wrapper_pipeline_cache.c */
   wrapper_pcache_create(device);

   result = wrapper_create_device_queue(device, pCreateInfo);
   if (result != VK_SUCCESS) {
      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   if (!physical_device->vk.supported_features.memoryMapPlaced) {
      device->vk.dispatch_table.AllocateMemory =
         wrapper_device_trampolines.AllocateMemory;
      device->vk.dispatch_table.MapMemory2 =
         wrapper_device_trampolines.MapMemory2;
      device->vk.dispatch_table.UnmapMemory =
         wrapper_device_trampolines.UnmapMemory;
      device->vk.dispatch_table.UnmapMemory2 =
         wrapper_device_trampolines.UnmapMemory2;
      device->vk.dispatch_table.FreeMemory =
         wrapper_device_trampolines.FreeMemory;
   }

   wrapper_emit_diag(physical_device, pCreateInfo, VK_SUCCESS);

   *pDevice = wrapper_device_to_handle(device);

   return VK_SUCCESS;
}

static void 
wrapper_buffer_destroy(struct wrapper_device *device,
					   struct wrapper_buffer *wb,
					   const VkAllocationCallbacks *pAllocator)
{
   if (wb == NULL)
      return;

   simple_mtx_lock(&device->resource_mutex);
      
   device->dispatch_table.DestroyBuffer(device->dispatch_handle,
      wb->dispatch_handle, pAllocator);

   _mesa_hash_table_u64_remove(device->buffer_table, (uint64_t)wb->dispatch_handle);
   list_del(&wb->link);

   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wb);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateBuffer(VkDevice _device,
					 const VkBufferCreateInfo *pCreateInfo,
					 const VkAllocationCallbacks *pAllocator,
					 VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;
   VkExternalMemoryHandleTypeFlags handle_types = 0;

   const VkExternalMemoryBufferCreateInfo *ext_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   if (ext_info) {
      handle_types = ext_info->handleTypes;
   }

   res = device->dispatch_table.CreateBuffer(device->dispatch_handle,
      pCreateInfo, pAllocator, pBuffer);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create buffer, res %d", res);
      return res;
   }

   simple_mtx_lock(&device->resource_mutex);

   struct wrapper_buffer *wb = vk_object_zalloc(&device->vk, 
      &device->vk.alloc, sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);

   if (!wb) {
      WRAPPER_LOG(error, "Failed to allocate wrapper_buffer");
      simple_mtx_unlock(&device->resource_mutex);
      device->dispatch_table.DestroyBuffer(device->dispatch_handle, *pBuffer, pAllocator);
      *pBuffer = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
      
   wb->device = device;
   wb->size = pCreateInfo->size;
   wb->dispatch_handle = *pBuffer;
   wb->handle_types = handle_types;

   list_add(&wb->link, &device->buffer_list);
   _mesa_hash_table_u64_insert(device->buffer_table, (uint64_t)wb->dispatch_handle, wb);

   simple_mtx_unlock(&device->resource_mutex);
   
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyBuffer(VkDevice _device,
					  VkBuffer buffer,
					  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle(device, buffer);
   wrapper_buffer_destroy(device, wb, pAllocator);
}

static void 
wrapper_image_destroy(struct wrapper_device *device,
					  struct wrapper_image *wi,
					  const VkAllocationCallbacks *pAllocator)
{
   if (wi == NULL)
      return;

   simple_mtx_lock(&device->resource_mutex);
      
   device->dispatch_table.DestroyImage(device->dispatch_handle,
      wi->dispatch_handle, pAllocator);

   _mesa_hash_table_u64_remove(device->image_table, (uint64_t)wi->dispatch_handle);
   list_del(&wi->link);

   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wi);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateImage(VkDevice _device,
					const VkImageCreateInfo *pCreateInfo,
					const VkAllocationCallbacks *pAllocator,
					VkImage *pImage)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;
   VkImageCreateInfo create_info;
   bool is_wsi_image = false;
   VkExternalMemoryHandleTypeFlags handle_types = 0;

   /* No B8G8R8A8 emulation lives in the wrapper any more. If the WSI layer still
    * tags an image with its private marker struct, just drop the marker so the
    * driver never sees an unknown struct. */
   VkBaseInStructure *prev = (VkBaseInStructure *) pCreateInfo;
   for (const VkBaseInStructure *s = pCreateInfo->pNext; s; s = s->pNext) {
      if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EMULATED_B8G8R8A8_CREATE_INFO_EXT) {
         prev->pNext = s->pNext; // unlink
         break;
      }
      prev = (VkBaseInStructure *) s;
   }

   // Tag swapchain images using VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA
   const struct wsi_image_create_info *wsi_info =
      vk_find_struct_const(pCreateInfo->pNext, WSI_IMAGE_CREATE_INFO_MESA);
   if (wsi_info) {
      is_wsi_image = true;
   }

   const VkExternalMemoryImageCreateInfo *ext_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   if (ext_info) {
      handle_types = ext_info->handleTypes;
   }

   /* Copy after the unlink above: when the marker struct is the head of the
    * pNext chain, the unlink only updates pCreateInfo->pNext, so a copy taken
    * earlier would still pass the unknown struct to the driver. */
   create_info = *pCreateInfo;

   res = device->dispatch_table.CreateImage(device->dispatch_handle,
      &create_info, pAllocator, pImage);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create image, res %d", res);
      return res;
   }

   simple_mtx_lock(&device->resource_mutex);

   struct wrapper_image *wi = vk_object_zalloc(&device->vk,
      &device->vk.alloc, sizeof(struct wrapper_image), VK_OBJECT_TYPE_IMAGE);

   if (!wi) {
      WRAPPER_LOG(error, "Failed to allocate wrapper_image");
      simple_mtx_unlock(&device->resource_mutex);
      device->dispatch_table.DestroyImage(device->dispatch_handle, *pImage, pAllocator);
      *pImage = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   wi->device = device;
   wi->dispatch_handle = *pImage;
   wi->is_wsi_image = is_wsi_image;
   wi->handle_types = handle_types;

   list_add(&wi->link, &device->image_list);
   _mesa_hash_table_u64_insert(device->image_table, (uint64_t)wi->dispatch_handle, wi);

   simple_mtx_unlock(&device->resource_mutex);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyImage(VkDevice _device,
					 VkImage image,
					 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   struct wrapper_image *wi = get_wrapper_image_from_handle(device, image);
   wrapper_image_destroy(device, wi, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex,
                       uint32_t queueIndex, VkQueue* pQueue) {
   vk_common_GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceQueue2(VkDevice _device, const VkDeviceQueueInfo2* pQueueInfo,
                        VkQueue* pQueue) {
   VK_FROM_HANDLE(vk_device, device, _device);

   struct vk_queue *queue = NULL;
   vk_foreach_queue(iter, device) {
      if (iter->queue_family_index == pQueueInfo->queueFamilyIndex &&
          iter->index_in_family == pQueueInfo->queueIndex &&
          iter->flags == pQueueInfo->flags) {
         queue = iter;
         break;
      }
   }

   *pQueue = queue ? vk_queue_to_handle(queue) : VK_NULL_HANDLE;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wrapper_GetDeviceProcAddr(VkDevice _device, const char* pName) {
   VK_FROM_HANDLE(wrapper_device, device, _device);
   return vk_device_get_proc_addr(&device->vk, pName);
}

static const char *
wrapper_fault_addr_type_str(VkDeviceFaultAddressTypeEXT t)
{
   switch (t) {
   case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:                        return "none";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:               return "read-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:              return "write-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:            return "execute-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "ip-unknown";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "ip-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:  return "ip-fault";
   default:                                                          return "unknown";
   }
}

/* On VK_ERROR_DEVICE_LOST, pull the VK_EXT_device_fault report so a black
 * screen / GPU hang shows WHERE it faulted (address + vendor codes) instead of
 * just a generic device-lost on the next submit. Emitted straight to stderr
 * under WRAPPER_DIAG -- exactly like wrapper_emit_diag's capability block -- so
 * it lands in the shared per-game diag file regardless of WRAPPER_LOG_LEVEL. A
 * device-lost is precisely when you most need this and least want it hidden
 * behind a log flag a tester forgot to set. */
static void
wrapper_log_device_fault(struct wrapper_device *device)
{
   PFN_vkGetDeviceFaultInfoEXT get_fault_info =
      device->dispatch_table.GetDeviceFaultInfoEXT;

   if (!device->device_fault_enabled || !get_fault_info)
      return;

   static int diag = -1;
   if (diag == -1)
      diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   if (!diag)
      return;

   VkDeviceFaultCountsEXT counts = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
   };
   if (get_fault_info(device->dispatch_handle, &counts, NULL) != VK_SUCCESS)
      return;

   VkDeviceFaultInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
   };
   if (counts.addressInfoCount)
      info.pAddressInfos =
         calloc(counts.addressInfoCount, sizeof(VkDeviceFaultAddressInfoEXT));
   if (counts.vendorInfoCount)
      info.pVendorInfos =
         calloc(counts.vendorInfoCount, sizeof(VkDeviceFaultVendorInfoEXT));
   /* We don't consume the opaque vendor crash dump. */
   counts.vendorBinarySize = 0;

#define DF(...) fprintf(stderr, "[WRAPPER_DIAG] " __VA_ARGS__)
   if (get_fault_info(device->dispatch_handle, &counts, &info) == VK_SUCCESS) {
      DF("==== VK_ERROR_DEVICE_LOST: GPU fault report ====\n");
      DF("  description: %s\n", info.description);

      for (uint32_t i = 0; info.pAddressInfos && i < counts.addressInfoCount; i++) {
         const VkDeviceFaultAddressInfoEXT *a = &info.pAddressInfos[i];
         /* reportedAddress is only exact to addressPrecision (a power of two):
          * the faulting address lies within that aligned range. */
         DF("  address fault[%u]: type=%s reportedAddress=0x%llx precision=0x%llx\n",
            i, wrapper_fault_addr_type_str(a->addressType),
            (unsigned long long)a->reportedAddress,
            (unsigned long long)a->addressPrecision);
      }
      for (uint32_t i = 0; info.pVendorInfos && i < counts.vendorInfoCount; i++) {
         const VkDeviceFaultVendorInfoEXT *v = &info.pVendorInfos[i];
         DF("  vendor fault[%u]: %s code=0x%llx data=0x%llx\n",
            i, v->description,
            (unsigned long long)v->vendorFaultCode,
            (unsigned long long)v->vendorFaultData);
      }
      DF("================================================\n");
   }
#undef DF

   free(info.pAddressInfos);
   free(info.pVendorInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_QueueSubmit(VkQueue _queue, uint32_t submitCount,
                    const VkSubmitInfo* pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(wrapper_queue, queue, _queue);
   VkCommandBuffer stack_cbs[64];
   VkSubmitInfo stack_submits[8];
   VkCommandBuffer *cbs = stack_cbs;
   VkSubmitInfo *submits = stack_submits;
   uint32_t total = 0, used = 0;
   VkResult result;

   /* Translate wrapper command buffer handles to the driver's. The common
    * small submit lives entirely on the stack: no allocation, no lock. */
   for (uint32_t i = 0; i < submitCount; i++)
      total += pSubmits[i].commandBufferCount;

   if (total > ARRAY_SIZE(stack_cbs))
      cbs = malloc(sizeof(*cbs) * total);
   if (submitCount > ARRAY_SIZE(stack_submits))
      submits = malloc(sizeof(*submits) * submitCount);
   if (!cbs || !submits) {
      if (cbs != stack_cbs) free(cbs);
      if (submits != stack_submits) free(submits);
      return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t i = 0; i < submitCount; i++) {
      submits[i] = pSubmits[i];
      submits[i].pCommandBuffers = cbs + used;
      for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; j++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb, pSubmits[i].pCommandBuffers[j]);
         cbs[used++] = wcb->dispatch_handle;
      }
   }

   result = queue->device->dispatch_table.QueueSubmit(
      queue->dispatch_handle, submitCount, submits, fence);

   if (result == VK_ERROR_DEVICE_LOST)
      wrapper_log_device_fault(queue->device);

   if (cbs != stack_cbs) free(cbs);
   if (submits != stack_submits) free(submits);

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_QueueSubmit2(VkQueue _queue, uint32_t submitCount,
                     const VkSubmitInfo2* pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(wrapper_queue, queue, _queue);
   VkCommandBufferSubmitInfo stack_cbs[64];
   VkSubmitInfo2 stack_submits[8];
   VkCommandBufferSubmitInfo *cbs = stack_cbs;
   VkSubmitInfo2 *submits = stack_submits;
   uint32_t total = 0, used = 0;
   VkResult result;

   for (uint32_t i = 0; i < submitCount; i++)
      total += pSubmits[i].commandBufferInfoCount;

   if (total > ARRAY_SIZE(stack_cbs))
      cbs = malloc(sizeof(*cbs) * total);
   if (submitCount > ARRAY_SIZE(stack_submits))
      submits = malloc(sizeof(*submits) * submitCount);
   if (!cbs || !submits) {
      if (cbs != stack_cbs) free(cbs);
      if (submits != stack_submits) free(submits);
      return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t i = 0; i < submitCount; i++) {
      submits[i] = pSubmits[i];
      submits[i].pCommandBufferInfos = cbs + used;
      for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; j++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb,
                        pSubmits[i].pCommandBufferInfos[j].commandBuffer);
         cbs[used] = pSubmits[i].pCommandBufferInfos[j];
         cbs[used].commandBuffer = wcb->dispatch_handle;
         used++;
      }
   }

   result = queue->device->dispatch_table.QueueSubmit2(
      queue->dispatch_handle, submitCount, submits, fence);

   if (result == VK_ERROR_DEVICE_LOST)
      wrapper_log_device_fault(queue->device);

   if (cbs != stack_cbs) free(cbs);
   if (submits != stack_submits) free(submits);

   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdExecuteCommands(VkCommandBuffer commandBuffer,
                           uint32_t commandBufferCount,
                           const VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   VkCommandBuffer command_buffers[commandBufferCount];

   for (int i = 0; i < commandBufferCount; i++) {
      command_buffers[i] =
         wrapper_command_buffer_from_handle(pCommandBuffers[i])->dispatch_handle;
   }
   wcb->device->dispatch_table.CmdExecuteCommands(
      wcb->dispatch_handle, commandBufferCount, command_buffers);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateShaderModule(VkDevice _device,
						   const VkShaderModuleCreateInfo *pCreateInfo,
						   const VkAllocationCallbacks *pAllocator,
						   VkShaderModule *pShaderModule)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   static int wrapper_no_remove_clip_distance = -1;
   static int wrapper_no_patch_OpConstComp = -1;
   VkShaderModuleCreateInfo create_info = *pCreateInfo;
   uint32_t *patched = NULL;
   VkResult result;

   if (wrapper_no_remove_clip_distance == -1)
      wrapper_no_remove_clip_distance = getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE") && atoi(getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE"));

   if (wrapper_no_patch_OpConstComp == -1)
      wrapper_no_patch_OpConstComp = getenv("WRAPPER_NO_PATCH_OPCONSTCOMP") && atoi(getenv("WRAPPER_NO_PATCH_OPCONSTCOMP"));

   /* Pure function of the SPIR-V: no lock needed (it only serialized the
    * parallel shader compilation threads), and the patched copy is released
    * right after the driver call instead of leaking one module per shader. */
   if (device->physical->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY &&
       create_info.codeSize >= 5 * sizeof(uint32_t)) {
      patched = malloc(create_info.codeSize);
      if (patched) {
         memcpy(patched, create_info.pCode, create_info.codeSize);
         if (!wrapper_no_patch_OpConstComp)
            patch_OpConstantComposite_to_OpSpecConstantComposite(patched, create_info.codeSize);
         if (!wrapper_no_remove_clip_distance)
            remove_ClipDistance(patched, &create_info.codeSize);
         create_info.pCode = patched;
      }
   }

   if (WRAPPER_LOG_LEVEL(shader))
      dump_shader_code(create_info.pCode, create_info.codeSize);

   result = device->dispatch_table.CreateShaderModule(
      device->dispatch_handle, &create_info, pAllocator, pShaderModule);

   free(patched);
   return result;
}

static VkResult
wrapper_command_buffer_create(struct wrapper_device *device,
                              VkCommandPool pool,
                              VkCommandBuffer dispatch_handle,
                              VkCommandBuffer *pCommandBuffers) {
   struct wrapper_command_buffer *wcb;
   wcb = vk_object_zalloc(&device->vk, &device->vk.alloc,
                          sizeof(struct wrapper_command_buffer),
                          VK_OBJECT_TYPE_COMMAND_BUFFER);
   if (!wcb)
      return vk_error(&device->vk, VK_ERROR_OUT_OF_HOST_MEMORY);

   wcb->device = device;
   wcb->pool = pool;
   wcb->dispatch_handle = dispatch_handle;
   list_add(&wcb->link, &device->command_buffer_list);

   *pCommandBuffers = wrapper_command_buffer_to_handle(wcb);

   return VK_SUCCESS;
}

static void
wrapper_command_buffer_destroy(struct wrapper_device *device,
                               struct wrapper_command_buffer *wcb) {
   if (wcb == NULL)
      return;

   device->dispatch_table.FreeCommandBuffers(
      device->dispatch_handle, wcb->pool, 1, &wcb->dispatch_handle);

   list_del(&wcb->link);
   vk_object_free(&device->vk, &device->vk.alloc, wcb);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateCommandBuffers(VkDevice _device,
                               const VkCommandBufferAllocateInfo* pAllocateInfo,
                               VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult result;
   uint32_t i;
   
   result = device->dispatch_table.AllocateCommandBuffers(
      device->dispatch_handle, pAllocateInfo, pCommandBuffers);
   if (result != VK_SUCCESS)
      return result;

   simple_mtx_lock(&device->resource_mutex);

   for (i = 0; i < pAllocateInfo->commandBufferCount; i++) {
      result = wrapper_command_buffer_create(
         device, pAllocateInfo->commandPool, pCommandBuffers[i],
         pCommandBuffers + i);
      if (result != VK_SUCCESS)
         break;
   }

   if (result != VK_SUCCESS) {
      for (int q = 0; q < i; q++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb, pCommandBuffers[q]);
         wrapper_command_buffer_destroy(device, wcb);
      }

      device->dispatch_table.FreeCommandBuffers(
         device->dispatch_handle, pAllocateInfo->commandPool,
         pAllocateInfo->commandBufferCount - i, pCommandBuffers + i);
      
      for (i = 0; i < pAllocateInfo->commandBufferCount; i++) {
         pCommandBuffers[i] = VK_NULL_HANDLE;
      }
   }

   simple_mtx_unlock(&device->resource_mutex);

   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_FreeCommandBuffers(VkDevice _device,
                           VkCommandPool commandPool,
                           uint32_t commandBufferCount,
                           const VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   simple_mtx_lock(&device->resource_mutex);

   for (int i = 0; i < commandBufferCount; i++) {
      VK_FROM_HANDLE(wrapper_command_buffer, wcb, pCommandBuffers[i]);
      wrapper_command_buffer_destroy(device, wcb);
   }

   simple_mtx_unlock(&device->resource_mutex);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                           const VkAllocationCallbacks* pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry_safe(struct wrapper_command_buffer, wcb,
                            &device->command_buffer_list, link) {
      if (wcb->pool == commandPool) {
         wrapper_command_buffer_destroy(device, wcb);
      }
   }

   simple_mtx_unlock(&device->resource_mutex);

   device->dispatch_table.DestroyCommandPool(device->dispatch_handle,
                                             commandPool, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyDevice(VkDevice _device, const VkAllocationCallbacks* pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry_safe(struct wrapper_command_buffer, wcb,
                            &device->command_buffer_list, link) {
      wrapper_command_buffer_destroy(device, wcb);
   }

   simple_mtx_unlock(&device->resource_mutex);

   wrapper_device_memory_destroy_all(device);
   
   list_for_each_entry_safe(struct wrapper_buffer, wb,
                            &device->buffer_list, link) {
      wrapper_buffer_destroy(device, wb, pAllocator);
   }
   list_for_each_entry_safe(struct wrapper_image, wi,
                            &device->image_list, link) {
      wrapper_image_destroy(device, wi, pAllocator);
   }

   list_for_each_entry_safe(struct vk_queue, queue, &device->vk.queues, link) {
      vk_queue_finish(queue);
      vk_free2(&device->vk.alloc, pAllocator, queue);
   }
   if (device->dispatch_handle != VK_NULL_HANDLE) {
      wrapper_pcache_destroy(device);
      device->dispatch_table.DestroyDevice(device->
         dispatch_handle, pAllocator);
   }
   _mesa_hash_table_u64_destroy(device->image_table);
   _mesa_hash_table_u64_destroy(device->buffer_table);
   _mesa_hash_table_u64_destroy(device->memory_table);
   simple_mtx_destroy(&device->resource_mutex);
   vk_device_finish(&device->vk);
   vk_free2(&device->vk.alloc, pAllocator, device);
}

static uint64_t
unwrap_device_object(VkObjectType objectType,
                     uint64_t objectHandle)
{
   switch(objectType) {
   case VK_OBJECT_TYPE_DEVICE:
      return (uint64_t)(uintptr_t)wrapper_device_from_handle((VkDevice)(uintptr_t)objectHandle)->dispatch_handle;
   case VK_OBJECT_TYPE_QUEUE:
      return (uint64_t)(uintptr_t)wrapper_queue_from_handle((VkQueue)(uintptr_t)objectHandle)->dispatch_handle;
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      return (uint64_t)(uintptr_t)wrapper_command_buffer_from_handle((VkCommandBuffer)(uintptr_t)objectHandle)->dispatch_handle;
   default:
      return objectHandle;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_SetPrivateData(VkDevice _device, VkObjectType objectType,
                       uint64_t objectHandle,
                       VkPrivateDataSlot privateDataSlot,
                       uint64_t data) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   uint64_t object_handle = unwrap_device_object(objectType, objectHandle);
   return device->dispatch_table.SetPrivateData(device->dispatch_handle,
      objectType, object_handle, privateDataSlot, data);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPrivateData(VkDevice _device, VkObjectType objectType,
                       uint64_t objectHandle,
                       VkPrivateDataSlot privateDataSlot,
                       uint64_t* pData) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   uint64_t object_handle = unwrap_device_object(objectType, objectHandle);
   return device->dispatch_table.GetPrivateData(device->dispatch_handle,
      objectType, object_handle, privateDataSlot, pData);
}
