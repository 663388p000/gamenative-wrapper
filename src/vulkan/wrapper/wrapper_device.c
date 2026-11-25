#include <stdalign.h>
#include <stddef.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "spirv_patcher.hpp"
#include "wrapper_spirv_emu.h"
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

/*
 * Extensions the wrapper really implements itself (WSI layer). These are the
 * ONLY device extensions that are advertised without the Mali driver listing
 * them, and they are never forwarded to the driver. Every other extension is
 * advertised if and only if the driver reports it -- no fake extensions.
 */
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
 * Neither advertised nor forwarded.
 *
 * VK_EXT_map_memory_placed used to be emulated by the wrapper (fixed-address
 * mapping of exported buffers). That emulation is gone, so the extension is
 * hidden even if a driver lists it: the wrapper does not implement it.
 * The same goes for the dma-buf / DRM-modifier extensions (see below).
 */
const struct vk_device_extension_table wrapper_filter_extensions =
{
   /* no dma-buf anywhere: swapchain images are AHardwareBuffers, and nothing
    * that is only about dma-buf / DRM format modifiers is of any use here */
   .EXT_external_memory_dma_buf = true,
   .EXT_image_drm_format_modifier = true,
   .EXT_hdr_metadata = true,
   .GOOGLE_display_timing = true,
   .KHR_shared_presentable_image = true,
   .EXT_image_compression_control_swapchain = true,
   .EXT_map_memory_placed = true,
};

/* Adds `name` to the driver request list once (names come from
 * vk_device_extensions[], so the list can never outgrow that table). */
static void
wrapper_ext_add(const char **exts, uint32_t *count, const char *name)
{
   for (uint32_t i = 0; i < *count; i++) {
      if (strcmp(exts[i], name) == 0)
         return;
   }
   if (*count < VK_DEVICE_EXTENSION_COUNT)
      exts[(*count)++] = name;
}

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
      wrapper_ext_add(exts, count, "VK_KHR_dynamic_rendering");
   if (plan->local_read && e->KHR_dynamic_rendering_local_read)
      wrapper_ext_add(exts, count, "VK_KHR_dynamic_rendering_local_read");
   if (plan->unused_attachments && e->EXT_dynamic_rendering_unused_attachments)
      wrapper_ext_add(exts, count, "VK_EXT_dynamic_rendering_unused_attachments");
   if (plan->multiview && e->KHR_multiview)
      wrapper_ext_add(exts, count, "VK_KHR_multiview");
}

/*
 * The request list handed to the Mali driver: exactly the extensions the
 * application enabled that the driver itself lists. WSI extensions (served by
 * the wrapper) and hidden ones never reach the driver.
 *
 * Nothing is appended on the wrapper's behalf here -- no buffer-device-address /
 * memory-model / driver-properties extras -- so the driver cannot fail
 * vkCreateDevice over an extension the application never asked for. What the
 * wrapper itself needs is added in tiers that a failing device creation drops
 * again (WSI: external memory / semaphore / fence + AHardwareBuffer, only with
 * VK_KHR_swapchain; depth clip control and the Valhall set). Extra names can
 * still be requested explicitly with WRAPPER_EXTRA_EXTENSIONS (see below).
 */
static void
wrapper_collect_app_extensions(const struct wrapper_device *device,
                               uint32_t *enable_extension_count,
                               const char **enable_extensions)
{
   for (int idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
      if (!device->vk.enabled_extensions.extensions[idx])
         continue;

      if (!device->physical->base_supported_extensions.extensions[idx])
         continue;

      if (wrapper_wsi_extensions.extensions[idx])
         continue;

      if (wrapper_filter_extensions.extensions[idx])
         continue;

      wrapper_ext_add(enable_extensions, enable_extension_count,
                      vk_device_extensions[idx].extensionName);
   }
}

/*
 * WRAPPER_EXTRA_EXTENSIONS="VK_EXT_memory_budget,VK_EXT_memory_priority"
 * asks the driver for extra device extensions on top of the application's
 * list. Only names the driver really lists are used; anything else is logged
 * and skipped.
 */
static void
wrapper_add_env_extensions(const struct wrapper_physical_device *pdevice,
                           uint32_t *count, const char **exts)
{
   const char *env = getenv("WRAPPER_EXTRA_EXTENSIONS");
   char *list, *save = NULL;

   if (!env || !env[0])
      return;

   list = strdup(env);
   if (!list)
      return;

   for (char *name = strtok_r(list, ", ", &save); name;
        name = strtok_r(NULL, ", ", &save)) {
      bool found = false;

      for (int idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
         if (strcmp(name, vk_device_extensions[idx].extensionName) != 0)
            continue;
         found = true;
         if (!pdevice->base_supported_extensions.extensions[idx] ||
             wrapper_wsi_extensions.extensions[idx] ||
             wrapper_filter_extensions.extensions[idx]) {
            WRAPPER_LOG(info, "WRAPPER_EXTRA_EXTENSIONS: %s is not listed by the driver (or is handled/hidden by the wrapper), skipped", name);
         } else {
            wrapper_ext_add(exts, count, vk_device_extensions[idx].extensionName);
         }
         break;
      }
      if (!found)
         WRAPPER_LOG(info, "WRAPPER_EXTRA_EXTENSIONS: unknown extension %s, skipped", name);
   }

   free(list);
}

static void unlink_vk_struct(const VkAllocationCallbacks *alloc,
                             VkBaseInStructure *create_info,
                             const VkBaseInStructure **current,
                             VkBaseInStructure **prev)
{
   const VkBaseInStructure *removed = *current;
   const VkBaseInStructure *next = removed->pNext;

   if (!*prev)
      create_info->pNext = next;
   else
      (*prev)->pNext = next;

   *current = next;
   vk_free(alloc, (void *)removed);
}

static void process_pnext_chain(const VkAllocationCallbacks *alloc,
                                VkBaseInStructure *create_info,
                                struct wrapper_physical_device *pdevice) {
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
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_dynamic_rendering_unused_attachments)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT from pNext chain");
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES:
             if (pdevice->base_supported_extensions.KHR_maintenance5)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceMaintenance5Features from pNext chain");
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_vertex_attribute_divisor)
                break;
             WRAPPER_LOG(info, "Unlinking VertexAttributeDivisorFeaturesEXT (driver lacks the EXT extension)");
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 1, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan11Features from pNext chain");
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 2, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan12Features from pNext chain");
             unlink_vk_struct(alloc, create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
             if (api_version < VK_MAKE_VERSION(1, 3, 0)) {
                WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan13Features from pNext chain");
                unlink_vk_struct(alloc, create_info, &current, &prev);
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
 * Force the Valhall dynamic-rendering feature bits on (only those the driver
 * really supports, see wrapper_valhall_plan): "mutate in place if present,
 * splice a struct in if not". dynamicRendering and multiview
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
       * wrapper-implemented ones (the WSI layer) may show up */
      unsigned not_in_driver = 0;
      D("--- extensions advertised without driver support ---\n");
      for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
         if (pdev->vk.supported_extensions.extensions[i] &&
             !pdev->base_supported_extensions.extensions[i]) {
            D("  %s%s\n", vk_device_extensions[i].extensionName,
              wrapper_wsi_extensions.extensions[i] ? " (wrapper WSI)" :
              (pdev->emu.viewport_index_layer &&
               !strcmp(vk_device_extensions[i].extensionName, "VK_EXT_shader_viewport_index_layer"))
                 ? " (emulated: Vulkan 1.2 shaderOutputViewportIndex/Layer)"
                 : " (UNEXPECTED: not in driver)");
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
   D("  memory: plain driver memory; AHB imports keep a wrapper-side reference; swapchain images are AHardwareBuffers (no dma-buf)\n");
   D("  clip/cull/depthClamp/depthBiasClamp/fillModeNonSolid/multiViewport (driver-native; the rest is emulated, see 'Features:' in the log): %d/%d/%d/%d/%d/%d\n",
     pdev->base_supported_features.shaderClipDistance,
     pdev->base_supported_features.shaderCullDistance,
     pdev->base_supported_features.depthClamp,
     pdev->base_supported_features.depthBiasClamp,
     pdev->base_supported_features.fillModeNonSolid,
     pdev->base_supported_features.multiViewport);
   D("  persistent pipeline cache: %s\n",
     (getenv("WRAPPER_PIPELINE_CACHE") && !atoi(getenv("WRAPPER_PIPELINE_CACHE"))) ? "off" : "on");
   D("  VK_EXT_device_fault report    : %s\n",
     !pdev->base_supported_extensions.EXT_device_fault ? "unsupported by base driver" :
     (getenv("WRAPPER_DEVICE_FAULT") && atoi(getenv("WRAPPER_DEVICE_FAULT")))
        ? "ON (GPU fault dumped on device loss)" : "off (opt in with WRAPPER_DEVICE_FAULT=1)");
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

/*
 * Extensions the wrapper's own WSI layer needs ENABLED on the driver device.
 *
 * Mesa's WSI exports every swapchain image and synchronises through fds. The
 * application never asks for any of that (DXVK/Wine only know VK_KHR_swapchain),
 * but without these extensions the driver's entry points do not exist / the
 * handle types are invalid and vkCreateSwapchainKHR dies inside the driver.
 *
 * The images are AHardwareBuffers (VK_ANDROID_external_memory_android_hardware_buffer):
 * Android's own buffer type, which every Mali driver backs. There is no dma-buf
 * (VK_EXT_external_memory_dma_buf, DRM modifiers) anywhere in the wrapper: what
 * WSI asks for as DMA_BUF is translated to AHB in wrapper_device_memory.c, and
 * the driver is never asked for a handle type it does not list.
 *
 * Requested only when the application enabled VK_KHR_swapchain, only if the
 * driver itself lists them, in tiers: if the driver refuses the device the AHB
 * tier is dropped first (see wrapper_CreateDevice).
 *
 * WRAPPER_WSI_EXTENSIONS = none | auto (default: base + ahb)
 */
enum {
   WRAPPER_WSI_BASE = 1,
   WRAPPER_WSI_AHB  = 2,
};

static const char *const wrapper_wsi_base_names[] = {
   "VK_KHR_external_memory",
   "VK_KHR_external_semaphore",
   "VK_KHR_external_semaphore_fd",
   "VK_KHR_external_fence",
   "VK_KHR_external_fence_fd",
   "VK_EXT_queue_family_foreign",
};
#define WRAPPER_WSI_AHB_NAME "VK_ANDROID_external_memory_android_hardware_buffer"

static bool
wrapper_driver_lists(const struct wrapper_physical_device *pdevice, const char *name)
{
   for (int idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
      if (strcmp(name, vk_device_extensions[idx].extensionName) == 0)
         return pdevice->base_supported_extensions.extensions[idx];
   }
   return false;
}

/* Which WSI extension groups the driver offers at all. */
static unsigned
wrapper_wsi_available(const struct wrapper_physical_device *pdevice)
{
   unsigned mask = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(wrapper_wsi_base_names); i++) {
      if (wrapper_driver_lists(pdevice, wrapper_wsi_base_names[i]))
         mask |= WRAPPER_WSI_BASE;
   }
   if (wrapper_driver_lists(pdevice, WRAPPER_WSI_AHB_NAME))
      mask |= WRAPPER_WSI_AHB;
   return mask;
}

/* What the user wants out of the available groups. */
static unsigned
wrapper_wsi_wanted(void)
{
   const char *env = getenv("WRAPPER_WSI_EXTENSIONS");

   if (env && strcmp(env, "none") == 0)
      return 0;
   return WRAPPER_WSI_BASE | WRAPPER_WSI_AHB;
}

static void
wrapper_add_wsi_extensions(const struct wrapper_physical_device *pdevice,
                           unsigned mask, uint32_t *count, const char **exts)
{
   if (mask & WRAPPER_WSI_BASE) {
      for (unsigned i = 0; i < ARRAY_SIZE(wrapper_wsi_base_names); i++) {
         if (wrapper_driver_lists(pdevice, wrapper_wsi_base_names[i]))
            wrapper_ext_add(exts, count, wrapper_wsi_base_names[i]);
      }
   }
   if ((mask & WRAPPER_WSI_AHB) && wrapper_driver_lists(pdevice, WRAPPER_WSI_AHB_NAME))
      wrapper_ext_add(exts, count, WRAPPER_WSI_AHB_NAME);
}

/* One line telling which driver entry points the WSI layer needs exist. */
static void
wrapper_log_wsi_entrypoints(const struct wrapper_device *device)
{
#define WSI_EP(n) (device->dispatch_table.n ? #n "=ok" : #n "=MISSING")
   WRAPPER_LOG(info, "WSI driver entry points: %s %s %s %s %s",
               WSI_EP(GetMemoryAndroidHardwareBufferANDROID),
               WSI_EP(GetSemaphoreFdKHR), WSI_EP(ImportSemaphoreFdKHR),
               WSI_EP(GetFenceFdKHR), WSI_EP(ImportFenceFdKHR));
#undef WSI_EP
}

/* Is VK_EXT_depth_clip_enable (extension + feature) part of this create info? */
static bool
wrapper_create_info_has_depth_clip(const VkDeviceCreateInfo *ci)
{
   const VkPhysicalDeviceDepthClipEnableFeaturesEXT *f =
      vk_find_struct_const(ci->pNext, PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT);
   bool ext = false;

   for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
      if (strcmp(ci->ppEnabledExtensionNames[i], "VK_EXT_depth_clip_enable") == 0)
         ext = true;
   }
   return ext && f && f->depthClipEnable;
}

/*
 * Feature bits the driver call is made with differ from what the application
 * wrote: bits for features the GPU lacks are cleared (the wrapper emulates or
 * refuses them), bits the emulation builds on are set. The application's own
 * structs are patched for the duration of the driver call and put back right
 * after it, so nothing of theirs is changed behind their back.
 */
struct wrapper_feature_patch {
   VkBool32 *p;
   VkBool32 old;
};

static void
wrapper_feature_patch(struct wrapper_feature_patch *patches, unsigned *n, unsigned cap,
                      VkBool32 *p, VkBool32 value)
{
   if (*p == value || *n >= cap)
      return;
   patches[*n].p = p;
   patches[*n].old = *p;
   (*n)++;
   *p = value;
}

static void
wrapper_feature_restore(struct wrapper_feature_patch *patches, unsigned n)
{
   while (n--)
      *patches[n].p = patches[n].old;
}

/* Vulkan 1.2 features that also have a struct of their own: a chain that has
 * one of those must not get a VkPhysicalDeviceVulkan12Features next to it. */
static bool
wrapper_chain_has_split_v12_features(const void *chain)
{
   return vk_find_struct_const(chain, PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) ||
          vk_find_struct_const(chain, PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES);
}

/*
 * VK_EXT_shader_viewport_index_layer is emulated on top of the Vulkan 1.2
 * features of the same name, and the driver only compiles the (translated)
 * capabilities when those are enabled. Set them in the application's own
 * Vulkan 1.2 struct if it has one, add a struct of ours if that can not clash.
 */
static void
wrapper_force_viewport_layer_features(VkDeviceCreateInfo *ci,
                                      VkPhysicalDeviceVulkan12Features *storage,
                                      struct wrapper_feature_patch *patches, unsigned *n,
                                      unsigned cap)
{
   VkPhysicalDeviceVulkan12Features *f = __vk_find_struct(
      (void *)ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);

   if (f) {
      wrapper_feature_patch(patches, n, cap, &f->shaderOutputViewportIndex, VK_TRUE);
      wrapper_feature_patch(patches, n, cap, &f->shaderOutputLayer, VK_TRUE);
      return;
   }
   if (wrapper_chain_has_split_v12_features(ci->pNext))
      return;                     /* the driver takes the capabilities regardless */

   storage->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
   storage->shaderOutputViewportIndex = VK_TRUE;
   storage->shaderOutputLayer = VK_TRUE;
   storage->pNext = (void *)ci->pNext;
   ci->pNext = storage;
}

static size_t
wrapper_device_create_info_pnext_size(VkStructureType sType)
{
   switch (sType) {
#include "wrapper_pnext_sizes.inc"
   default:
      return 0;
   }
}

static void
wrapper_free_pnext_chain(const VkAllocationCallbacks *alloc,
                         VkBaseOutStructure *chain)
{
   while (chain) {
      VkBaseOutStructure *next = chain->pNext;
      vk_free(alloc, chain);
      chain = next;
   }
}

static VkResult
wrapper_clone_pnext_chain(const VkAllocationCallbacks *alloc,
                          const void *source,
                          void **destination)
{
   const VkBaseInStructure *src = source;
   VkBaseOutStructure *head = NULL;
   VkBaseOutStructure *tail = NULL;

   *destination = NULL;

   while (src) {
      size_t size =
         wrapper_device_create_info_pnext_size(src->sType);

      if (!size) {
         wrapper_free_pnext_chain(alloc, head);
         return VK_ERROR_INITIALIZATION_FAILED;
      }

      VkBaseOutStructure *copy =
         vk_alloc(alloc, size, alignof(max_align_t),
                  VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);

      if (!copy) {
         wrapper_free_pnext_chain(alloc, head);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      memcpy(copy, src, size);
      copy->pNext = NULL;

      if (tail)
         tail->pNext = copy;
      else
         head = copy;

      tail = copy;
      src = src->pNext;
   }

   *destination = head;
   return VK_SUCCESS;
}


VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDevice(VkPhysicalDevice physicalDevice,
                     const VkDeviceCreateInfo* pCreateInfo,
                     const VkAllocationCallbacks* pAllocator,
                     VkDevice* pDevice)
{
   VK_FROM_HANDLE(wrapper_physical_device, physical_device, physicalDevice);
   /* what the application asked for and the driver lists ... */
   const char *app_extensions[VK_DEVICE_EXTENSION_COUNT];
   /* ... plus what the wrapper adds on top (Valhall set, explicit opt-ins) */
   const char *wrapper_enable_extensions[VK_DEVICE_EXTENSION_COUNT];
   /* the list of the attempt being tried: base + WSI tier */
   const char *attempt_extensions[VK_DEVICE_EXTENSION_COUNT];
   uint32_t app_extension_count = 0;
   uint32_t wrapper_enable_extension_count = 0;
   VkDeviceCreateInfo wrapper_create_info = *pCreateInfo;
   struct vk_device_dispatch_table dispatch_table;
   struct wrapper_device *device;
   VkPhysicalDeviceFeatures2 *pdf2;
   VkPhysicalDeviceFeatures *pdf;
   VkResult result;
   void *app_chain = NULL;
   const void *full_chain = NULL;
   void *full_chain_allocated = NULL;
   VkPhysicalDeviceFeatures *private_enabled_features = NULL;
   static int wrapper_safe_create_device = -1;
   static int wrapper_device_fault = -1;
   VkPhysicalDeviceFaultFeaturesEXT fault_features_ext = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
   };
   unsigned create_stage = 1;

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
      _mesa_hash_table_u64_destroy(device->memory_table);
      simple_mtx_destroy(&device->resource_mutex);
      vk_free2(&physical_device->instance->vk.alloc, pAllocator,
               device);
      return vk_error(physical_device, result);
   }

   wrapper_emu_init_device(device);

   /* depthClamp is emulated with VK_EXT_depth_clip_enable (wrapper_emu.c):
    * when the application enables depthClamp that extension and its feature
    * must be on at the driver, even though the application never asked. */
   const VkPhysicalDeviceFeatures *app_pdf = pCreateInfo->pEnabledFeatures;
   const VkPhysicalDeviceFeatures2 *app_pdf2 =
      vk_find_struct_const(pCreateInfo->pNext, PHYSICAL_DEVICE_FEATURES_2);
   const bool want_depth_clip = device->emu.depth_clamp &&
      ((app_pdf && app_pdf->depthClamp) || (app_pdf2 && app_pdf2->features.depthClamp));

   /*
    * Never modify the application's pNext chain or pEnabledFeatures.
    * app_chain is the filtered application-only chain.
    */
   result = wrapper_clone_pnext_chain(&device->vk.alloc,
                                      pCreateInfo->pNext,
                                      &app_chain);
   if (result != VK_SUCCESS) {
      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   wrapper_create_info.pNext = app_chain;

   if (pCreateInfo->pEnabledFeatures) {
      private_enabled_features =
         vk_alloc(&device->vk.alloc,
                  sizeof(*private_enabled_features),
                  alignof(max_align_t),
                  VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);

      if (!private_enabled_features) {
         wrapper_free_pnext_chain(&device->vk.alloc, app_chain);
         wrapper_DestroyDevice(wrapper_device_to_handle(device),
                               &device->vk.alloc);
         return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      memcpy(private_enabled_features, pCreateInfo->pEnabledFeatures,
             sizeof(*private_enabled_features));

      wrapper_create_info.pEnabledFeatures = private_enabled_features;
   }

   /*
    * Driver request list: the application's extensions (that the driver
    * lists) and nothing else, except the Valhall dynamic-rendering set, what
    * an emulated feature needs, and names the user asked for explicitly.
    * See wrapper_collect_app_extensions().
    */
   wrapper_collect_app_extensions(device, &app_extension_count, app_extensions);
   memcpy(wrapper_enable_extensions, app_extensions,
          app_extension_count * sizeof(app_extensions[0]));
   wrapper_enable_extension_count = app_extension_count;

   if (valhall_plan.enforce) {
      wrapper_append_valhall_extensions(&valhall_plan, physical_device,
         &wrapper_enable_extension_count, wrapper_enable_extensions);
   }

   wrapper_add_env_extensions(physical_device,
      &wrapper_enable_extension_count, wrapper_enable_extensions);

   if (want_depth_clip)
      wrapper_ext_add(wrapper_enable_extensions,
                      &wrapper_enable_extension_count, "VK_EXT_depth_clip_enable");

   /* VK_EXT_device_fault turns the generic VK_ERROR_DEVICE_LOST into a GPU
    * fault report (faulting address + vendor fault codes) that is dumped in
    * QueueSubmit. It is an extra extension the application did not ask for,
    * so it is opt-in now: WRAPPER_DEVICE_FAULT=1. */
   if (wrapper_device_fault == -1)
      wrapper_device_fault = getenv("WRAPPER_DEVICE_FAULT")
         ? atoi(getenv("WRAPPER_DEVICE_FAULT")) : 0;

   bool enable_device_fault = wrapper_device_fault &&
      physical_device->base_supported_extensions.EXT_device_fault;

   if (enable_device_fault)
      wrapper_ext_add(wrapper_enable_extensions,
                      &wrapper_enable_extension_count, "VK_EXT_device_fault");

   wrapper_create_info.enabledExtensionCount = wrapper_enable_extension_count;
   wrapper_create_info.ppEnabledExtensionNames = wrapper_enable_extensions;

   struct wrapper_feature_patch feature_patches[40];
   unsigned n_feature_patches = 0;
   const struct vk_features *base = &physical_device->base_supported_features;
   const struct vk_features *app_enabled = &device->vk.enabled_features;

   pdf = private_enabled_features;
   pdf2 = __vk_find_struct((void *)app_chain,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);

#define PATCH_FEATURE(f, v) do { \
   if (pdf) \
      wrapper_feature_patch(feature_patches, &n_feature_patches, ARRAY_SIZE(feature_patches), \
                            &pdf->f, (v)); \
   if (pdf2) \
      wrapper_feature_patch(feature_patches, &n_feature_patches, ARRAY_SIZE(feature_patches), \
                            &pdf2->features.f, (v)); \
} while (0)
#define STRIP_FEATURE(f) do { if (!base->f) PATCH_FEATURE(f, VK_FALSE); } while (0)

   /* what the GPU lacks never reaches the driver (it refuses the device) */
   STRIP_FEATURE(textureCompressionBC);
   STRIP_FEATURE(multiViewport);
   STRIP_FEATURE(depthClamp);
   STRIP_FEATURE(depthBiasClamp);
   STRIP_FEATURE(fillModeNonSolid);
   STRIP_FEATURE(shaderClipDistance);
   STRIP_FEATURE(shaderCullDistance);
   STRIP_FEATURE(dualSrcBlend);
   STRIP_FEATURE(multiDrawIndirect);

   /* cull distance and wireframe are drawn by a geometry shader the wrapper
    * builds: the driver needs the feature on, whether the application asked
    * for it or not (only worth it when it asked for either of the two) */
   device->emu_gs = (pdf || pdf2) && base->geometryShader &&
      ((device->emu.fill_mode && app_enabled->fillModeNonSolid) ||
       (device->emu.cull_distance && app_enabled->shaderCullDistance));
   if (device->emu_gs) {
      PATCH_FEATURE(geometryShader, VK_TRUE);
      if (base->shaderTessellationAndGeometryPointSize)
         PATCH_FEATURE(shaderTessellationAndGeometryPointSize, VK_TRUE);
      device->emu_gs_point_size = base->shaderTessellationAndGeometryPointSize;
   }

#undef STRIP_FEATURE
#undef PATCH_FEATURE

   process_pnext_chain(&device->vk.alloc,
                       (VkBaseInStructure *)&wrapper_create_info,
                       device->physical);

   /*
    * Keep app_chain untouched by wrapper-forced feature additions.
    * full_chain is the separate chain passed to the primary driver attempts.
    */
   result = wrapper_clone_pnext_chain(&device->vk.alloc,
                                      app_chain,
                                      &full_chain);
   if (result != VK_SUCCESS) {
      wrapper_feature_restore(feature_patches, n_feature_patches);
      wrapper_free_pnext_chain(&device->vk.alloc, app_chain);
      vk_free(&device->vk.alloc, private_enabled_features);
      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   full_chain_allocated = full_chain;
   wrapper_create_info.pNext = full_chain;

   /* These storage structs must stay alive through the driver
    * CreateDevice() call below, so they live in this stack frame rather
    * than inside the helper functions. */
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

   VkPhysicalDeviceVulkan12Features wrapper_v12_features = {0};

   if (device->emu.viewport_index_layer &&
       device->vk.enabled_extensions.EXT_shader_viewport_index_layer)
      wrapper_force_viewport_layer_features(&wrapper_create_info, &wrapper_v12_features,
                                            feature_patches, &n_feature_patches,
                                            ARRAY_SIZE(feature_patches));

   VkPhysicalDeviceDepthClipEnableFeaturesEXT wrapper_dce_features = {0};

   if (want_depth_clip) {
      VkPhysicalDeviceDepthClipEnableFeaturesEXT *f = __vk_find_struct(
         (void *)wrapper_create_info.pNext,
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT);
      if (f) {
         wrapper_feature_patch(feature_patches, &n_feature_patches,
                               ARRAY_SIZE(feature_patches), &f->depthClipEnable, VK_TRUE);
      } else {
         wrapper_dce_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT;
         wrapper_dce_features.depthClipEnable = VK_TRUE;
         wrapper_dce_features.pNext = (void *)wrapper_create_info.pNext;
         wrapper_create_info.pNext = &wrapper_dce_features;
      }
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

   if (wrapper_safe_create_device == -1) {
      wrapper_safe_create_device = getenv("WRAPPER_SAFE_CREATE_DEVICE") ? atoi(getenv("WRAPPER_SAFE_CREATE_DEVICE")) : 1;
   }

   full_chain = wrapper_create_info.pNext;

   /* full_chain contains all wrapper additions. */

   /* WSI extension tiers, heaviest first: [base+ahb] -> [base]; only when the
    * application presents (VK_KHR_swapchain). */
   unsigned wsi_masks[2];
   unsigned n_wsi = 0;
   {
      unsigned top = device->vk.enabled_extensions.KHR_swapchain ?
         (wrapper_wsi_wanted() & wrapper_wsi_available(physical_device)) : 0;

      wsi_masks[n_wsi++] = top;
      if (top & WRAPPER_WSI_AHB)
         wsi_masks[n_wsi++] = top & ~WRAPPER_WSI_AHB;
   }

   /* The application-only attempt only makes sense if the wrapper added anything. */
   const bool wrapper_added_something =
      wrapper_enable_extension_count != app_extension_count ||
      full_chain != app_chain || wsi_masks[0] != 0;
   const unsigned total_attempts = n_wsi + (wrapper_added_something ? 1 : 0) + 1;
   unsigned attempt;
   bool used_null_chain = false;

   /*
    * attempts 1..n_wsi   application list + wrapper additions + WSI tier i
    * next                only what the application asked for (extensions and
    *                     feature chain)
    * last                only the application's extensions, NULL pNext chain
    * A failure is retried with less, so an extension or feature the wrapper
    * added can never be the reason a device is refused.
    */
   for (attempt = 0; ; attempt++) {
      if (attempt < n_wsi) {
         uint32_t n = wrapper_enable_extension_count;

         memcpy(attempt_extensions, wrapper_enable_extensions, n * sizeof(attempt_extensions[0]));
         wrapper_add_wsi_extensions(physical_device, wsi_masks[attempt], &n, attempt_extensions);
         wrapper_create_info.enabledExtensionCount = n;
         wrapper_create_info.ppEnabledExtensionNames = attempt_extensions;
         wrapper_create_info.pNext = full_chain;
         if (attempt == 0 && WRAPPER_LOG_LEVEL(info)) {
            for (uint32_t i = 0; i < n; i++)
               WRAPPER_LOG(info, "Enabling device extension %s", attempt_extensions[i]);
         } else if (attempt > 0) {
            WRAPPER_LOG(info, "createDevice attempt %u: WSI extension groups 0x%x (%u extensions)",
                        attempt + 1, wsi_masks[attempt], n);
         }
      } else if (attempt == n_wsi && wrapper_added_something) {
         wrapper_create_info.enabledExtensionCount = app_extension_count;
         wrapper_create_info.ppEnabledExtensionNames = app_extensions;
         wrapper_create_info.pNext = app_chain;
         WRAPPER_LOG(info, "createDevice attempt %u: only the extensions and features the application requested",
                     attempt + 1);
      } else {
         wrapper_create_info.enabledExtensionCount = app_extension_count;
         wrapper_create_info.ppEnabledExtensionNames = app_extensions;
         wrapper_create_info.pNext = NULL;
         used_null_chain = true;
         WRAPPER_LOG(info, "createDevice attempt %u: NULL pNext chain", attempt + 1);
      }

      result = physical_device->dispatch_table.CreateDevice(
         physical_device->dispatch_handle, &wrapper_create_info,
         pAllocator, &device->dispatch_handle);

      if (result == VK_SUCCESS || !wrapper_safe_create_device || attempt + 1 >= total_attempts)
         break;

      WRAPPER_LOG(info, "Driver createDevice failed (res %d)", result);
      device->dispatch_handle = VK_NULL_HANDLE;
   }
   create_stage = attempt + 1;

   /* the driver call is over: the application's feature structs are what they
    * were. A device made without the feature chain has no geometry shader. */
   wrapper_feature_restore(feature_patches, n_feature_patches);
   if (used_null_chain && !pdf)
      device->emu_gs = false;
   if (!device->emu_gs)
      device->emu_gs_point_size = false;

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed driver createDevice, res %d", result);
      wrapper_emit_diag(physical_device, pCreateInfo, result);

      wrapper_free_pnext_chain(&device->vk.alloc, full_chain_allocated);
      wrapper_free_pnext_chain(&device->vk.alloc, app_chain);
      vk_free(&device->vk.alloc, private_enabled_features);

      full_chain_allocated = NULL;
      full_chain = NULL;
      app_chain = NULL;
      private_enabled_features = NULL;

      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   /* swapchain images are AHardwareBuffers when the driver device got the extension */
   for (uint32_t i = 0; i < wrapper_create_info.enabledExtensionCount; i++) {
      if (strcmp(wrapper_create_info.ppEnabledExtensionNames[i], WRAPPER_WSI_AHB_NAME) == 0)
         device->wsi_ahb = true;
   }

   if (create_stage > 1)
      WRAPPER_LOG(info, "Device created on attempt %u", create_stage);
   if (create_stage <= n_wsi && wsi_masks[create_stage - 1])
      WRAPPER_LOG(info, "WSI extension groups enabled on the driver device: 0x%x", wsi_masks[create_stage - 1]);

   /* depthClamp emulation needs depth-clip control on the device that really
    * got created (a fallback attempt may have dropped it). */
   device->emu_depth_clip = want_depth_clip &&
      wrapper_create_info_has_depth_clip(&wrapper_create_info);
   if (want_depth_clip && !device->emu_depth_clip)
      WRAPPER_LOG(info, "depthClamp emulation unavailable on this device (VK_EXT_depth_clip_enable not enabled)");

   /* The driver has finished reading both private chains. */
   wrapper_free_pnext_chain(&device->vk.alloc, full_chain_allocated);
   wrapper_free_pnext_chain(&device->vk.alloc, app_chain);
   vk_free(&device->vk.alloc, private_enabled_features);

   full_chain = NULL;
   app_chain = NULL;
   private_enabled_features = NULL;

   void *gdpa = physical_device->instance->dispatch_table.GetInstanceProcAddr(
      physical_device->instance->dispatch_handle, "vkGetDeviceProcAddr");
   vk_device_dispatch_table_load(&device->dispatch_table, gdpa,
                                 device->dispatch_handle);

   wrapper_log_wsi_entrypoints(device);

   /* Only the attempts that keep the wrapper's feature chain carry the
    * deviceFault feature struct. */
   device->device_fault_enabled = enable_device_fault && create_stage <= n_wsi;

   /* Persistent pipeline cache (Mali compiles are slow): see wrapper_pipeline_cache.c */
   wrapper_pcache_create(device);

   result = wrapper_create_device_queue(device, pCreateInfo);
   if (result != VK_SUCCESS) {
      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   wrapper_emit_diag(physical_device, pCreateInfo, VK_SUCCESS);

   *pDevice = wrapper_device_to_handle(device);

   return VK_SUCCESS;
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
   VkCommandBuffer stack[32];
   VkCommandBuffer *command_buffers = stack;

   /* the usual count fits the stack; a large one must not blow it */
   if (commandBufferCount > ARRAY_SIZE(stack)) {
      command_buffers = malloc(commandBufferCount * sizeof(*command_buffers));
      if (!command_buffers)
         return;
   }
   for (uint32_t i = 0; i < commandBufferCount; i++) {
      command_buffers[i] =
         wrapper_command_buffer_from_handle(pCommandBuffers[i])->dispatch_handle;
   }
   wcb->device->dispatch_table.CmdExecuteCommands(
      wcb->dispatch_handle, commandBufferCount, command_buffers);
   if (command_buffers != stack)
      free(command_buffers);
}

/* Driver-specific SPIR-V workarounds. Shared by wrapper_CreateShaderModule()
 * and by the emulation layer, which builds shader modules of its own. */
void
wrapper_shader_driver_patch(const struct wrapper_device *device,
                            uint32_t *code, size_t size_bytes)
{
   static int wrapper_no_patch_OpConstComp = -1;

   if (device->physical->driver_properties.driverID != VK_DRIVER_ID_ARM_PROPRIETARY ||
       size_bytes < 5 * sizeof(uint32_t))
      return;

   int no_patch = __atomic_load_n(&wrapper_no_patch_OpConstComp, __ATOMIC_ACQUIRE);

   if (no_patch == -1) {
      const char *env = getenv("WRAPPER_NO_PATCH_OPCONSTCOMP");
      int value = env && atoi(env);
      int expected = -1;

      __atomic_compare_exchange_n(&wrapper_no_patch_OpConstComp,
                                  &expected, value, false,
                                  __ATOMIC_RELEASE, __ATOMIC_RELAXED);
      no_patch = __atomic_load_n(&wrapper_no_patch_OpConstComp, __ATOMIC_ACQUIRE);
   }

   if (!no_patch)
      patch_OpConstantComposite_to_OpSpecConstantComposite(code, size_bytes);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateShaderModule(VkDevice _device,
						   const VkShaderModuleCreateInfo *pCreateInfo,
						   const VkAllocationCallbacks *pAllocator,
						   VkShaderModule *pShaderModule)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkShaderModuleCreateInfo create_info = *pCreateInfo;
   uint32_t *words = NULL;
   VkResult result;

   /*
    * SPIR-V that needs a change gets a private copy; everything else goes to
    * the driver as the application wrote it. A change is the Arm workaround
    * patch, or something the GPU would refuse (ClipDistance / CullDistance it
    * lacks -- pipelines the emulation can lower get modules of their own, see
    * wrapper_emu.c --, SPV_EXT_shader_viewport_index_layer, a geometry shader
    * writing gl_ViewportIndex): wrapper_emu_fixup_module(). It is a pure
    * function of the SPIR-V, so no lock: the shader compile threads run in
    * parallel.
    */
   if (create_info.codeSize >= 5 * sizeof(uint32_t) &&
       (device->physical->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY ||
        wrapper_spv_scan(create_info.pCode, create_info.codeSize))) {
      words = malloc(create_info.codeSize);
      if (words) {
         size_t size = create_info.codeSize;

         memcpy(words, create_info.pCode, size);
         wrapper_shader_driver_patch(device, words, size);
         wrapper_emu_fixup_module(device, &words, &size);
         create_info.pCode = words;
         create_info.codeSize = size;
      }
   }

   if (WRAPPER_LOG_LEVEL(shader))
      dump_shader_code(create_info.pCode, create_info.codeSize);

   result = device->dispatch_table.CreateShaderModule(
      device->dispatch_handle, &create_info, pAllocator, pShaderModule);

   /* the emulation layer rewrites from the application's own SPIR-V */
   if (result == VK_SUCCESS && wrapper_emu_wants_shaders(device) &&
       pCreateInfo->codeSize >= 5 * sizeof(uint32_t))
      wrapper_emu_add_shader(device, *pShaderModule, pCreateInfo->pCode,
                             pCreateInfo->codeSize);

   free(words);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyShaderModule(VkDevice _device, VkShaderModule shaderModule,
                            const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   wrapper_emu_remove_shader(device, shaderModule);
   device->dispatch_table.DestroyShaderModule(device->dispatch_handle,
                                              shaderModule, pAllocator);
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

   list_for_each_entry_safe(struct vk_queue, queue, &device->vk.queues, link) {
      vk_queue_finish(queue);
      vk_free2(&device->vk.alloc, pAllocator, queue);
   }
   wrapper_emu_finish_device(device);
   if (device->dispatch_handle != VK_NULL_HANDLE) {
      wrapper_pcache_destroy(device);
      device->dispatch_table.DestroyDevice(device->
         dispatch_handle, pAllocator);
   }
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
