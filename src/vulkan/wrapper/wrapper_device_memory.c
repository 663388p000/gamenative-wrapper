/*
 * Device memory.
 *
 * Everything the application does with VkDeviceMemory (allocate, map, flush,
 * invalidate, free) is forwarded to the Mali driver untouched: the entry
 * points that are not defined in this file resolve to the plain trampolines
 * in wrapper_device_entrypoints/wrapper_device_trampolines.
 *
 * What was removed, on purpose:
 *   - VK_EXT_map_memory_placed and the emulation behind it (fixed-address
 *     mmap of an exported buffer),
 *   - the DRM / dma-buf allocation tier and the "host-visible memory that is
 *     really something else" paths,
 *   - every memory type / property rewrite.
 * Nothing here pretends the driver can do something it cannot.
 *
 * What is left is the one piece that is real and always reachable: when an
 * application imports an AHardwareBuffer
 * (VK_ANDROID_external_memory_android_hardware_buffer) the wrapper holds its
 * own reference on that buffer for the lifetime of the VkDeviceMemory, so the
 * buffer cannot disappear underneath a still-bound allocation even if the
 * application drops its reference right after vkAllocateMemory() returns.
 * Allocations without an AHardwareBuffer cost one relaxed atomic read.
 *
 * The kbase node (/dev/mali0) is deliberately not used for memory: kbase has
 * no dma-buf export, and memory allocated in a kbase context of our own is not
 * visible to the Vulkan driver's context, so a kbase-backed buffer could never
 * be handed to the driver. The wrapper itself allocates no GPU buffer.
 */

#include <android/hardware_buffer.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "vk_alloc.h"
#include "vk_util.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"

static void
wrapper_device_memory_register(struct wrapper_device *device,
                               struct wrapper_device_memory *mem)
{
   simple_mtx_lock(&device->resource_mutex);
   list_add(&mem->link, &device->device_memory_list);
   _mesa_hash_table_u64_insert(device->memory_table,
                               (uint64_t)mem->dispatch_handle, mem);
   simple_mtx_unlock(&device->resource_mutex);

   p_atomic_inc(&device->ahb_memory_count);
}

static struct wrapper_device_memory *
wrapper_device_memory_unregister(struct wrapper_device *device,
                                 VkDeviceMemory handle)
{
   struct wrapper_device_memory *mem;

   simple_mtx_lock(&device->resource_mutex);
   mem = _mesa_hash_table_u64_search(device->memory_table, (uint64_t)handle);
   if (mem) {
      list_del(&mem->link);
      _mesa_hash_table_u64_remove(device->memory_table, (uint64_t)handle);
   }
   simple_mtx_unlock(&device->resource_mutex);

   if (mem)
      p_atomic_dec(&device->ahb_memory_count);

   return mem;
}

static void
wrapper_device_memory_free_entry(struct wrapper_device *device,
                                 struct wrapper_device_memory *mem)
{
   if (mem->ahardware_buffer)
      AHardwareBuffer_release(mem->ahardware_buffer);
   vk_free(&device->vk.alloc, mem);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateMemory(VkDevice _device,
                       const VkMemoryAllocateInfo *pAllocateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkDeviceMemory *pMemory)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   const VkImportAndroidHardwareBufferInfoANDROID *import_ahb =
      vk_find_struct_const(pAllocateInfo->pNext,
                           IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID);
   struct wrapper_device_memory *mem = NULL;
   VkResult result;

   if (import_ahb && import_ahb->buffer) {
      mem = vk_zalloc(&device->vk.alloc, sizeof(*mem), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!mem)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
                                                  pAllocateInfo, pAllocator,
                                                  pMemory);
   if (result != VK_SUCCESS) {
      if (mem)
         vk_free(&device->vk.alloc, mem);
      return result;
   }

   if (mem) {
      AHardwareBuffer_acquire(import_ahb->buffer);
      mem->device = device;
      mem->dispatch_handle = *pMemory;
      mem->ahardware_buffer = import_ahb->buffer;
      wrapper_device_memory_register(device, mem);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_FreeMemory(VkDevice _device,
                   VkDeviceMemory _memory,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem = NULL;

   if (_memory == VK_NULL_HANDLE)
      return;

   /* fast path: nothing AHB-backed is alive, no lock, no lookup */
   if (p_atomic_read(&device->ahb_memory_count) > 0)
      mem = wrapper_device_memory_unregister(device, _memory);

   device->dispatch_table.FreeMemory(device->dispatch_handle, _memory,
                                     pAllocator);

   /* drop our reference only after the driver has let go of the buffer */
   if (mem)
      wrapper_device_memory_free_entry(device, mem);
}

void
wrapper_device_memory_destroy_all(struct wrapper_device *device)
{
   /* The application leaked these; the driver frees its side in
    * vkDestroyDevice(). Only the references the wrapper took are ours. */
   simple_mtx_lock(&device->resource_mutex);
   list_for_each_entry_safe(struct wrapper_device_memory, mem,
                            &device->device_memory_list, link) {
      list_del(&mem->link);
      _mesa_hash_table_u64_remove(device->memory_table,
                                  (uint64_t)mem->dispatch_handle);
      wrapper_device_memory_free_entry(device, mem);
   }
   simple_mtx_unlock(&device->resource_mutex);

   p_atomic_set(&device->ahb_memory_count, 0);
}
