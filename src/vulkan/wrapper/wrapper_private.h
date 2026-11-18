#ifndef __WRAPPER_PRIVATE_H
#define __WRAPPER_PRIVATE_H

#include "vulkan/runtime/vk_instance.h"
#include "vulkan/runtime/vk_physical_device.h"
#include "vulkan/runtime/vk_device.h"
#include "vulkan/runtime/vk_queue.h"
#include "vulkan/runtime/vk_image.h"
#include "vulkan/runtime/vk_fence.h"
#include "vulkan/runtime/vk_command_buffer.h"
#include "vulkan/runtime/vk_log.h"
#include "vulkan/runtime/vk_buffer.h"
#include "vulkan/util/vk_dispatch_table.h"
#include "vulkan/wsi/wsi_common.h"
#include "util/simple_mtx.h"
#include "util/hash_table.h"

#include "wrapper_mali.h"

/* Limit advertised for emulated VK_KHR_push_descriptor. 32 is the common
 * hardware value and comfortably covers what vkd3d/DXVK push. */
#define WRAPPER_MAX_PUSH_DESCRIPTORS 32

extern const struct vk_instance_extension_table wrapper_instance_extensions;
extern const struct vk_device_extension_table wrapper_wsi_extensions;
extern const struct vk_device_extension_table wrapper_filter_extensions;

struct wrapper_instance {
   struct vk_instance vk;

   VkInstance dispatch_handle;
   struct vk_instance_dispatch_table dispatch_table;
};

VK_DEFINE_HANDLE_CASTS(wrapper_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)

/*
 * Which of the "big" core features are emulated by the wrapper (wrapper_emu.c)
 * because the Mali driver does not report them. A feature the driver reports
 * is never emulated: it is passed through untouched. A feature that is neither
 * native nor emulated stays off.
 *
 *   shaderClipDistance  emulated by lowering ClipDistance to a varying and
 *                       discarding in the fragment shader (wrapper_spirv_emu.cpp)
 *   depthClamp          emulated with VK_EXT_depth_clip_enable (clip off == clamp)
 *   multiViewport       emulated by mapping every viewport/scissor array to
 *                       element 0 -- exact, because without geometry shaders or
 *                       viewport-index shader outputs no shader can pick another one
 *
 * shaderCullDistance, fillModeNonSolid and depthBiasClamp can not be emulated
 * by a layer on top of a closed driver, so they are exactly what the driver says.
 */
struct wrapper_emu {
   bool clip_distance;
   bool depth_clamp;
   bool multi_viewport;
};

struct wrapper_physical_device {
   struct vk_physical_device vk;

   bool is_vkd3d;
   struct wrapper_mali_info mali;   /* GPU identity read from kbase (/dev/mali0) */
   VkPhysicalDevice dispatch_handle;
   VkPhysicalDeviceProperties2 properties2;
   VkPhysicalDeviceDriverProperties driver_properties;
   VkPhysicalDeviceMemoryProperties memory_properties;
   struct wsi_device wsi_device;
   struct wrapper_instance *instance;
   struct vk_features base_supported_features;
   struct vk_device_extension_table base_supported_extensions;
   struct vk_physical_device_dispatch_table dispatch_table;
   struct wrapper_emu emu;
};

VK_DEFINE_HANDLE_CASTS(wrapper_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

struct wrapper_queue {
   struct vk_queue vk;

   struct wrapper_device *device;
   VkQueue dispatch_handle;
};

VK_DEFINE_HANDLE_CASTS(wrapper_queue, vk.base, VkQueue,
                       VK_OBJECT_TYPE_QUEUE)

struct wrapper_device {
   struct vk_device vk;

   VkDevice dispatch_handle;
   simple_mtx_t resource_mutex;
   struct list_head command_buffer_list;
   struct list_head device_memory_list;
   struct list_head buffer_list;
   struct list_head image_list;
   struct hash_table_u64 *buffer_table;
   struct hash_table_u64 *image_table;
   struct hash_table_u64 *memory_table;   /* VkDeviceMemory -> wrapper_device_memory (AHB imports only) */
   uint32_t ahb_memory_count;             /* entries in memory_table (atomic) */
   struct wrapper_physical_device *physical;
   struct vk_device_dispatch_table dispatch_table;

   bool device_fault_enabled;

   /* feature emulation (wrapper_emu.c) */
   struct wrapper_emu emu;
   bool emu_depth_clip;                    /* VK_EXT_depth_clip_enable is enabled on the driver device */
   struct hash_table_u64 *shader_table;    /* VkShaderModule -> wrapper_emu_shader (only if emu needs SPIR-V) */
   struct list_head shader_list;           /* the same entries, for teardown */

   /* persistent pipeline cache shared by every pipeline the app creates without one */
   struct wrapper_pcache *pcache;
};

VK_DEFINE_HANDLE_CASTS(wrapper_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

struct wrapper_buffer {
   struct vk_buffer vk;

   struct wrapper_device *device;
   struct list_head link;
   VkBuffer dispatch_handle;
   VkDeviceSize size;
   VkExternalMemoryHandleTypeFlags handle_types;
};

struct wrapper_buffer *
get_wrapper_buffer_from_handle_locked(struct wrapper_device *device, VkBuffer buffer);

struct wrapper_buffer *
get_wrapper_buffer_from_handle(struct wrapper_device *device, VkBuffer buffer);

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EMULATED_B8G8R8A8_CREATE_INFO_EXT 1000701234
typedef struct VkEmulatedB8G8R8A8CreateInfoExt {
    VkStructureType          sType;
    const void*              pNext;
} VkEmulatedB8G8R8A8CreateInfoExt;

struct wrapper_image {
   struct vk_image vk;

   struct wrapper_device *device;
   struct list_head link;
   VkImage dispatch_handle;

   bool is_wsi_image;
   VkExternalMemoryHandleTypeFlags handle_types;
};

struct wrapper_image *
get_wrapper_image_from_handle_locked(struct wrapper_device *device, VkImage image);

struct wrapper_image *
get_wrapper_image_from_handle(struct wrapper_device *device, VkImage image);

struct wrapper_command_buffer {
   struct vk_command_buffer vk;

   struct wrapper_device *device;
   struct list_head link;
   VkCommandPool pool;
   VkCommandBuffer dispatch_handle;
};

VK_DEFINE_HANDLE_CASTS(wrapper_command_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

/* Only allocations that import an AHardwareBuffer are tracked: the wrapper
 * keeps its own reference on the buffer until vkFreeMemory(). Everything else
 * is plain driver memory and has no wrapper-side record. */
struct wrapper_device_memory {
   struct AHardwareBuffer *ahardware_buffer;
   struct wrapper_device *device;
   struct list_head link;
   VkDeviceMemory dispatch_handle;
};

VkResult enumerate_physical_device(struct vk_instance *_instance);
void destroy_physical_device(struct vk_physical_device *pdevice);

void
wrapper_setup_device_features(struct wrapper_physical_device *physical_device);

/* Drops the wrapper's AHardwareBuffer references of every still-tracked
 * allocation (device teardown). */
void
wrapper_device_memory_destroy_all(struct wrapper_device *device);

/* ---- persistent pipeline cache (wrapper_pipeline_cache.c) ---- */
struct wrapper_pcache;

struct wrapper_pcache *
wrapper_pcache_create(struct wrapper_device *device);

void
wrapper_pcache_destroy(struct wrapper_device *device);

VkPipelineCache
wrapper_pcache_handle(struct wrapper_device *device);

void
wrapper_pcache_note_created(struct wrapper_device *device, uint32_t count);

/* ---- feature emulation (wrapper_emu.c) ---- */
struct wrapper_emu_rewrite {
   VkGraphicsPipelineCreateInfo *infos;   /* patched copy of the application's array */
   void **blocks;                         /* scratch memory to free afterwards */
   uint32_t block_count, block_cap;
   VkShaderModule *modules;               /* temporary shader modules to destroy */
   uint32_t module_count, module_cap;
};

/* Decides what is emulated, adjusts the advertised features/limits. */
void wrapper_emu_init_physical(struct wrapper_physical_device *pdevice);
void wrapper_emu_patch_limits(const struct wrapper_physical_device *pdevice,
                              VkPhysicalDeviceLimits *limits);
void wrapper_emu_init_device(struct wrapper_device *device);
void wrapper_emu_finish_device(struct wrapper_device *device);

/* SPIR-V registry (only filled when an emulation needs the original shader). */
bool wrapper_emu_wants_shaders(const struct wrapper_device *device);
void wrapper_emu_add_shader(struct wrapper_device *device, VkShaderModule module,
                            const uint32_t *code, size_t size_bytes);
void wrapper_emu_remove_shader(struct wrapper_device *device, VkShaderModule module);

/* Graphics pipelines: returns `infos` untouched when nothing has to change. */
const VkGraphicsPipelineCreateInfo *
wrapper_emu_rewrite_graphics(struct wrapper_device *device, uint32_t count,
                             const VkGraphicsPipelineCreateInfo *infos,
                             struct wrapper_emu_rewrite *rw);
void wrapper_emu_rewrite_free(struct wrapper_device *device,
                              struct wrapper_emu_rewrite *rw);

/* Applies the driver-specific SPIR-V workarounds (shared with
 * wrapper_CreateShaderModule). */
void wrapper_shader_driver_patch(const struct wrapper_device *device,
                                 uint32_t *code, size_t size_bytes);

#endif /* WRAPPER_PRIVATE_H */
