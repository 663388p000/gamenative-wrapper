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
 *
 * WSI and AHardwareBuffer
 * -----------------------
 * Mesa's WSI creates every swapchain image / blit buffer exportable as DMA_BUF
 * and fetches the fd with vkGetMemoryFdKHR(). The wrapper never gives the
 * driver a dma-buf handle type: the four calls WSI makes (vkCreateImage,
 * vkCreateBuffer, vkAllocateMemory, vkGetMemoryFdKHR) are translated here, so
 * the driver sees VK_ANDROID_external_memory_android_hardware_buffer only --
 * Android's own buffer type, which gralloc and every Mali driver agree on.
 *
 *   handleTypes DMA_BUF          -> ANDROID_HARDWARE_BUFFER (create image / buffer,
 *                                   allocate memory; the caller's struct is put
 *                                   back right after the driver call)
 *   swapchain image, OPTIMAL     -> LINEAR when the driver can export that as an
 *                                   AHardwareBuffer: the consumer gets a raw
 *                                   buffer, with no tiling / AFBC metadata to
 *                                   interpret it by
 *   vkGetMemoryFdKHR(DMA_BUF)    -> vkGetMemoryAndroidHardwareBufferANDROID, then
 *                                   the buffer's own fd (gralloc native handle,
 *                                   or AHardwareBuffer_sendHandleToUnixSocket on
 *                                   releases without getNativeHandle)
 *
 * Active only when the driver device got the AHB extension (device->wsi_ahb),
 * which is only requested for applications that use VK_KHR_swapchain; anything
 * else passes straight through. The application can not reach this path with
 * dma-buf: the extension is not advertised.
 */

#include <android/hardware_buffer.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

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

#define WRAPPER_HT_DMA_BUF VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
#define WRAPPER_HT_AHB     VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID

/* What WSI asks for as DMA_BUF is an AHardwareBuffer here. On true, *types was
 * changed and `*saved` holds what has to be put back after the driver call. */
static bool
wsi_translate_handle_types(const struct wrapper_device *device,
                           VkExternalMemoryHandleTypeFlags *types,
                           VkExternalMemoryHandleTypeFlags *saved)
{
   if (!device->wsi_ahb || !(*types & WRAPPER_HT_DMA_BUF))
      return false;
   *saved = *types;
   *types = (*types & ~(VkExternalMemoryHandleTypeFlags)WRAPPER_HT_DMA_BUF) | WRAPPER_HT_AHB;
   return true;
}

/* the part of a gralloc native_handle_t we read (stable ABI: version, numFds,
 * numInts, then the fds) -- the NDK does not export the type */
struct wrapper_native_handle {
   int version;
   int numFds;
   int numInts;
   int data[];
};

/* AHardwareBuffer_sendHandleToUnixSocket: the fds come out of the other end */
static int
wrapper_ahb_fd_via_socket(struct AHardwareBuffer *ahb,
                          int (*send_handle)(const struct AHardwareBuffer *, int))
{
   union {
      struct cmsghdr hdr;
      char buf[CMSG_SPACE(sizeof(int) * 16)];
   } ctl;
   char data[1024];
   struct iovec iov = { .iov_base = data, .iov_len = sizeof(data) };
   struct msghdr msg = {
      .msg_iov = &iov, .msg_iovlen = 1,
      .msg_control = ctl.buf, .msg_controllen = sizeof(ctl.buf),
   };
   int sv[2], fd = -1;

   if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv) < 0)
      return -1;

   if (send_handle(ahb, sv[0]) == 0 &&
       recvmsg(sv[1], &msg, MSG_CMSG_CLOEXEC | MSG_DONTWAIT) >= 0) {
      for (struct cmsghdr *h = CMSG_FIRSTHDR(&msg); h; h = CMSG_NXTHDR(&msg, h)) {
         if (h->cmsg_level != SOL_SOCKET || h->cmsg_type != SCM_RIGHTS)
            continue;
         int n = (int)((h->cmsg_len - CMSG_LEN(0)) / sizeof(int));
         int *fds = (int *)CMSG_DATA(h);
         for (int i = 0; i < n; i++) {
            if (fd < 0)
               fd = fds[i];          /* the first one is the buffer itself */
            else
               close(fds[i]);
         }
      }
   }
   close(sv[0]);
   close(sv[1]);
   return fd;
}

/* A new fd (the caller owns it) for the memory behind an AHardwareBuffer, or -1. */
static int
wrapper_ahb_buffer_fd(struct AHardwareBuffer *ahb)
{
   const struct wrapper_native_handle *(*get_handle)(const struct AHardwareBuffer *) =
      (const struct wrapper_native_handle *(*)(const struct AHardwareBuffer *))
         dlsym(RTLD_DEFAULT, "AHardwareBuffer_getNativeHandle");           /* API 31 */
   int (*send_handle)(const struct AHardwareBuffer *, int) =
      (int (*)(const struct AHardwareBuffer *, int))
         dlsym(RTLD_DEFAULT, "AHardwareBuffer_sendHandleToUnixSocket");    /* API 26 */

   if (get_handle) {
      const struct wrapper_native_handle *h = get_handle(ahb);
      if (h && h->numFds > 0 && h->data[0] >= 0)
         return fcntl(h->data[0], F_DUPFD_CLOEXEC, 3);
   }
   return send_handle ? wrapper_ahb_fd_via_socket(ahb, send_handle) : -1;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_GetMemoryFdKHR(VkDevice _device,
                       const VkMemoryGetFdInfoKHR *pGetFdInfo,
                       int *pFd)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (pGetFdInfo->handleType == WRAPPER_HT_DMA_BUF && device->wsi_ahb &&
       device->dispatch_table.GetMemoryAndroidHardwareBufferANDROID) {
      const VkMemoryGetAndroidHardwareBufferInfoANDROID info = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
         .memory = pGetFdInfo->memory,
      };
      struct AHardwareBuffer *ahb = NULL;
      VkResult res = device->dispatch_table.GetMemoryAndroidHardwareBufferANDROID(
         device->dispatch_handle, &info, &ahb);
      int fd;

      if (res != VK_SUCCESS)
         return res;
      fd = wrapper_ahb_buffer_fd(ahb);
      AHardwareBuffer_release(ahb);        /* the call above handed us a reference */
      if (fd < 0) {
         WRAPPER_LOG(error, "no fd for the swapchain image's AHardwareBuffer");
         return VK_ERROR_TOO_MANY_OBJECTS;
      }
      *pFd = fd;
      return VK_SUCCESS;
   }

   if (!device->dispatch_table.GetMemoryFdKHR)
      return VK_ERROR_EXTENSION_NOT_PRESENT;
   return device->dispatch_table.GetMemoryFdKHR(device->dispatch_handle, pGetFdInfo, pFd);
}

/* Can the driver export a LINEAR image of this kind as an AHardwareBuffer? */
static bool
wsi_linear_exportable(struct wrapper_device *device, const VkImageCreateInfo *ci)
{
   const struct wrapper_physical_device *pd = device->physical;
   VkPhysicalDeviceExternalImageFormatInfo ext_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .handleType = WRAPPER_HT_AHB,
   };
   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &ext_info,
      .format = ci->format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = ci->usage,
      .flags = ci->flags,
   };
   VkExternalImageFormatProperties ext_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &ext_props,
   };

   if (ci->imageType != VK_IMAGE_TYPE_2D || ci->mipLevels != 1 || ci->arrayLayers != 1 ||
       ci->samples != VK_SAMPLE_COUNT_1_BIT ||
       !pd->dispatch_table.GetPhysicalDeviceImageFormatProperties2)
      return false;

   return pd->dispatch_table.GetPhysicalDeviceImageFormatProperties2(
             pd->dispatch_handle, &info, &props) == VK_SUCCESS &&
          (ext_props.externalMemoryProperties.externalMemoryFeatures &
           VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateImage(VkDevice _device,
                    const VkImageCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkImage *pImage)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkExternalMemoryImageCreateInfo *ext = (VkExternalMemoryImageCreateInfo *)
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   VkExternalMemoryHandleTypeFlags saved = 0;
   VkImageCreateInfo linear;
   const VkImageCreateInfo *ci = pCreateInfo;
   VkResult result;

   /* everything that is not a swapchain image goes straight through */
   if (!ext || !wsi_translate_handle_types(device, &ext->handleTypes, &saved))
      return device->dispatch_table.CreateImage(device->dispatch_handle, pCreateInfo,
                                                pAllocator, pImage);

   if (pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL &&
       vk_find_struct_const(pCreateInfo->pNext, WSI_IMAGE_CREATE_INFO_MESA) &&
       wsi_linear_exportable(device, pCreateInfo)) {
      linear = *pCreateInfo;
      linear.tiling = VK_IMAGE_TILING_LINEAR;
      ci = &linear;
   }

   result = device->dispatch_table.CreateImage(device->dispatch_handle, ci, pAllocator, pImage);
   ext->handleTypes = saved;
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateBuffer(VkDevice _device,
                     const VkBufferCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkExternalMemoryBufferCreateInfo *ext = (VkExternalMemoryBufferCreateInfo *)
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   VkExternalMemoryHandleTypeFlags saved = 0;
   VkResult result;

   if (!ext || !wsi_translate_handle_types(device, &ext->handleTypes, &saved))
      return device->dispatch_table.CreateBuffer(device->dispatch_handle, pCreateInfo,
                                                 pAllocator, pBuffer);

   result = device->dispatch_table.CreateBuffer(device->dispatch_handle, pCreateInfo,
                                                pAllocator, pBuffer);
   ext->handleTypes = saved;
   return result;
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
   VkExportMemoryAllocateInfo *export = (VkExportMemoryAllocateInfo *)
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   VkExternalMemoryHandleTypeFlags saved_types = 0;
   bool translated = export &&
      wsi_translate_handle_types(device, &export->handleTypes, &saved_types);
   struct wrapper_device_memory *mem = NULL;
   VkResult result;

   if (import_ahb && import_ahb->buffer) {
      mem = vk_zalloc(&device->vk.alloc, sizeof(*mem), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!mem) {
         if (translated)
            export->handleTypes = saved_types;
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
                                                  pAllocateInfo, pAllocator,
                                                  pMemory);
   if (translated)
      export->handleTypes = saved_types;
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
