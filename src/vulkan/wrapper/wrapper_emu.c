/*
 * Feature emulation.
 *
 * The Mali driver is the single source of truth: a feature it reports is
 * passed through untouched. Only when it does NOT report one of the features
 * below does the wrapper implement it, and only if the implementation is exact
 * (see struct wrapper_emu in wrapper_private.h for the list and the reasoning).
 * Everything else -- shaderCullDistance, fillModeNonSolid, depthBiasClamp --
 * stays exactly as the driver reports it.
 *
 * What is done here, per pipeline, at vkCreateGraphicsPipelines() time:
 *
 *  shaderClipDistance  the vertex shader's gl_ClipDistance[] is moved to a plain
 *                      float[N] varying and the fragment shader discards when
 *                      any element is negative -- a fragment is inside the
 *                      clipped polygon exactly when every plane distance is >= 0.
 *  depthClamp          depthClampEnable becomes depthClipEnable = false
 *                      (VK_EXT_depth_clip_enable), the application's own
 *                      depth-clip state wins when it supplied one.
 *  multiViewport       viewportCount/scissorCount are clamped to 1 and the
 *                      vkCmdSetViewport/Scissor(WithCount) calls only forward
 *                      element 0.
 *
 * A pipeline is only rewritten when it needs it; if a rewrite can not be done
 * completely the pipeline is created exactly as the application described it.
 */

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "wrapper_spirv_emu.h"
#include "vk_alloc.h"
#include "vk_util.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/simple_mtx.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ plan */

void
wrapper_emu_init_physical(struct wrapper_physical_device *pdevice)
{
   const struct vk_features *base = &pdevice->base_supported_features;
   const struct vk_device_extension_table *ext = &pdevice->base_supported_extensions;
   struct vk_features *sup = &pdevice->vk.supported_features;
   struct wrapper_emu *emu = &pdevice->emu;
   const char *env = getenv("WRAPPER_FEATURE_EMULATION");
   const char *compat = getenv("WRAPPER_COMPAT_FEATURES");

   memset(emu, 0, sizeof(*emu));

   /* what the GPU really has, so a log tells why something is (not) available */
   WRAPPER_LOG(info, "Driver features: clip=%d cull=%d depthClamp=%d depthBiasClamp=%d fillModeNonSolid=%d multiViewport=%d | geometry=%d tessellation=%d viewportIndexOut=%d viewportIndexExt=%d | multiDrawIndirect=%d drawIndirectFirstInstance=%d imageGatherExtended=%d",
               base->shaderClipDistance, base->shaderCullDistance, base->depthClamp,
               base->depthBiasClamp, base->fillModeNonSolid, base->multiViewport,
               base->geometryShader, base->tessellationShader,
               base->shaderOutputViewportIndex, ext->EXT_shader_viewport_index_layer,
               base->multiDrawIndirect, base->drawIndirectFirstInstance,
               base->shaderImageGatherExtended);

   /* WRAPPER_FEATURE_EMULATION=0: report exactly what the driver reports */
   if (env && !atoi(env))
      goto compat;

   if (!base->shaderClipDistance) {
      emu->clip_distance = true;
      sup->shaderClipDistance = true;
   }

   if (!base->depthClamp && ext->EXT_depth_clip_enable && base->depthClipEnable) {
      emu->depth_clamp = true;
      sup->depthClamp = true;
   }

   /* With geometry shaders or a viewport-index shader output a shader could
    * pick another viewport and "everything is viewport 0" would be wrong. */
   if (!base->multiViewport && !base->geometryShader &&
       !base->shaderOutputViewportIndex && !ext->EXT_shader_viewport_index_layer) {
      emu->multi_viewport = true;
      sup->multiViewport = true;
   }

   WRAPPER_LOG(info, "Feature emulation: shaderClipDistance=%s depthClamp=%s multiViewport=%s",
               emu->clip_distance ? "emulated" : (base->shaderClipDistance ? "native" : "off"),
               emu->depth_clamp ? "emulated" : (base->depthClamp ? "native" : "off"),
               emu->multi_viewport ? "emulated" : (base->multiViewport ? "native" : "off"));
   WRAPPER_LOG(info, "Not emulatable, as reported by the driver: shaderCullDistance=%d fillModeNonSolid=%d depthBiasClamp=%d",
               base->shaderCullDistance, base->fillModeNonSolid, base->depthBiasClamp);

compat:
   /*
    * WRAPPER_COMPAT_FEATURES=1 -- the old behavior, OFF by default: report the
    * six features as supported although the GPU lacks them and nothing
    * implements them. DXVK's D3D11 front end refuses every feature level
    * unless they are reported (D3D9 does not care); the driver never sees the
    * features it lacks (wrapper_CreateDevice strips them), so a game that
    * really uses one renders wrong. It exists only to get D3D11 started.
    */
   if (compat && atoi(compat)) {
      #define COMPAT(f) do { if (!sup->f) { sup->f = true; \
         WRAPPER_LOG(info, "WRAPPER_COMPAT_FEATURES: reporting " #f " although it is not implemented"); } } while (0)
      COMPAT(shaderClipDistance);
      COMPAT(shaderCullDistance);
      COMPAT(depthClamp);
      COMPAT(depthBiasClamp);
      COMPAT(fillModeNonSolid);
      COMPAT(multiViewport);
      #undef COMPAT
   }
}

void
wrapper_emu_patch_limits(const struct wrapper_physical_device *pdevice,
                         VkPhysicalDeviceLimits *limits)
{
   const struct wrapper_emu *emu = &pdevice->emu;

   /* the Vulkan minimums that go with the features */
   if (emu->clip_distance) {
      limits->maxClipDistances = MAX2(limits->maxClipDistances, 8);
      limits->maxCombinedClipAndCullDistances =
         MAX2(limits->maxCombinedClipAndCullDistances, 8);
   }
   if (emu->multi_viewport)
      limits->maxViewports = MAX2(limits->maxViewports, 16);
}

/* ------------------------------------------------------- SPIR-V registry */

struct wrapper_emu_shader {
   struct list_head link;
   VkShaderModule module;
   uint32_t *code;      /* the application's SPIR-V, before any workaround */
   size_t size;         /* bytes */
};

bool
wrapper_emu_wants_shaders(const struct wrapper_device *device)
{
   return device->emu.clip_distance;
}

void
wrapper_emu_init_device(struct wrapper_device *device)
{
   device->emu = device->physical->emu;
   if (wrapper_emu_wants_shaders(device)) {
      list_inithead(&device->shader_list);
      device->shader_table = _mesa_hash_table_u64_create(NULL);
   }
}

static void
shader_free(struct wrapper_emu_shader *s)
{
   free(s->code);
   free(s);
}

void
wrapper_emu_finish_device(struct wrapper_device *device)
{
   if (!device->shader_table)
      return;

   list_for_each_entry_safe(struct wrapper_emu_shader, s, &device->shader_list, link) {
      list_del(&s->link);
      shader_free(s);
   }
   _mesa_hash_table_u64_destroy(device->shader_table);
   device->shader_table = NULL;
}

void
wrapper_emu_add_shader(struct wrapper_device *device, VkShaderModule module,
                       const uint32_t *code, size_t size_bytes)
{
   struct wrapper_emu_shader *s, *old;

   if (!device->shader_table || module == VK_NULL_HANDLE)
      return;

   s = calloc(1, sizeof(*s));
   if (!s)
      return;
   s->code = malloc(size_bytes);
   if (!s->code) {
      free(s);
      return;
   }
   memcpy(s->code, code, size_bytes);
   s->size = size_bytes;
   s->module = module;

   simple_mtx_lock(&device->resource_mutex);
   old = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   if (old) {
      list_del(&old->link);
      _mesa_hash_table_u64_remove(device->shader_table, (uint64_t)module);
   }
   list_add(&s->link, &device->shader_list);
   _mesa_hash_table_u64_insert(device->shader_table, (uint64_t)module, s);
   simple_mtx_unlock(&device->resource_mutex);

   if (old)
      shader_free(old);
}

void
wrapper_emu_remove_shader(struct wrapper_device *device, VkShaderModule module)
{
   struct wrapper_emu_shader *s;

   if (!device->shader_table || module == VK_NULL_HANDLE)
      return;

   simple_mtx_lock(&device->resource_mutex);
   s = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   if (s) {
      list_del(&s->link);
      _mesa_hash_table_u64_remove(device->shader_table, (uint64_t)module);
   }
   simple_mtx_unlock(&device->resource_mutex);

   if (s)
      shader_free(s);
}

static const struct wrapper_emu_shader *
shader_lookup(struct wrapper_device *device, VkShaderModule module)
{
   struct wrapper_emu_shader *s;

   if (!device->shader_table)
      return NULL;
   simple_mtx_lock(&device->resource_mutex);
   s = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   simple_mtx_unlock(&device->resource_mutex);
   return s;
}

/* ------------------------------------------------------ pipeline rewrite */

static void *
rw_alloc(struct wrapper_emu_rewrite *rw, size_t size)
{
   void *p;

   if (rw->block_count == rw->block_cap) {
      uint32_t cap = rw->block_cap ? rw->block_cap * 2 : 16;
      void **n = realloc(rw->blocks, cap * sizeof(*n));
      if (!n)
         return NULL;
      rw->blocks = n;
      rw->block_cap = cap;
   }
   p = calloc(1, size);
   if (p)
      rw->blocks[rw->block_count++] = p;
   return p;
}

/* Hands two temporary shader modules to `rw` (all or nothing). */
static bool
rw_track_modules(struct wrapper_emu_rewrite *rw, VkShaderModule a, VkShaderModule b)
{
   if (rw->module_count + 2 > rw->module_cap) {
      uint32_t cap = rw->module_cap ? rw->module_cap * 2 : 8;
      VkShaderModule *n = realloc(rw->modules, cap * sizeof(*n));
      if (!n)
         return false;
      rw->modules = n;
      rw->module_cap = cap;
   }
   rw->modules[rw->module_count++] = a;
   rw->modules[rw->module_count++] = b;
   return true;
}

static void
rewrite_depth_clamp(struct wrapper_emu_rewrite *rw, VkGraphicsPipelineCreateInfo *info)
{
   const VkPipelineRasterizationStateCreateInfo *rs = info->pRasterizationState;
   VkPipelineRasterizationStateCreateInfo *n;

   if (!rs || !rs->depthClampEnable)
      return;

   n = rw_alloc(rw, sizeof(*n));
   if (!n)
      return;
   *n = *rs;
   n->depthClampEnable = VK_FALSE;

   /* clamp on means clip off, unless the application said otherwise itself */
   if (!vk_find_struct_const(rs->pNext, PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT)) {
      VkPipelineRasterizationDepthClipStateCreateInfoEXT *dc = rw_alloc(rw, sizeof(*dc));
      if (!dc)
         return;
      dc->sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT;
      dc->pNext = n->pNext;
      dc->depthClipEnable = VK_FALSE;
      n->pNext = dc;
   }

   info->pRasterizationState = n;
}

static void
rewrite_viewports(struct wrapper_emu_rewrite *rw, VkGraphicsPipelineCreateInfo *info)
{
   const VkPipelineViewportStateCreateInfo *vp = info->pViewportState;
   VkPipelineViewportStateCreateInfo *n;

   if (!vp || (vp->viewportCount <= 1 && vp->scissorCount <= 1))
      return;

   n = rw_alloc(rw, sizeof(*n));
   if (!n)
      return;
   *n = *vp;
   n->viewportCount = MIN2(n->viewportCount, 1);
   n->scissorCount = MIN2(n->scissorCount, 1);
   info->pViewportState = n;
}

static void
rewrite_clip_distance(struct wrapper_device *device, struct wrapper_emu_rewrite *rw,
                      VkGraphicsPipelineCreateInfo *info)
{
   const VkPhysicalDeviceLimits *lim = &device->physical->properties2.properties.limits;
   const struct wrapper_emu_shader *vss, *fss;
   struct wrapper_spv_stage_info vi, fi;
   VkPipelineShaderStageCreateInfo *stages;
   uint32_t *vs_code = NULL, *fs_code = NULL;
   size_t vs_size = 0, fs_size = 0;
   VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
   int vs = -1, fs = -1;
   uint32_t loc, budget;

   for (uint32_t i = 0; i < info->stageCount; i++) {
      switch (info->pStages[i].stage) {
      case VK_SHADER_STAGE_VERTEX_BIT:
         vs = i;
         break;
      case VK_SHADER_STAGE_FRAGMENT_BIT:
         fs = i;
         break;
      case VK_SHADER_STAGE_GEOMETRY_BIT:
      case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
         return; /* only vertex + fragment pipelines are lowered */
      default:
         break;
      }
   }
   if (vs < 0)
      return;

   vss = shader_lookup(device, info->pStages[vs].module);
   if (!vss || !wrapper_spv_analyze(vss->code, vss->size, WSPV_VERTEX,
                                    info->pStages[vs].pName, &vi) ||
       !vi.found || !vi.has_clip || vi.clip_count == 0)
      return;

   /* nothing is rasterized: clipping can't be seen */
   if (fs < 0 || (info->pRasterizationState &&
                  info->pRasterizationState->rasterizerDiscardEnable))
      return;

   fss = shader_lookup(device, info->pStages[fs].module);
   if (!fss || !wrapper_spv_analyze(fss->code, fss->size, WSPV_FRAGMENT,
                                    info->pStages[fs].pName, &fi) || !fi.found) {
      WRAPPER_LOG(info, "clip distance: fragment shader unknown, keeping the plain pipeline");
      return;
   }
   if (vi.slots_end == WSPV_SLOTS_UNKNOWN || fi.slots_end == WSPV_SLOTS_UNKNOWN) {
      WRAPPER_LOG(info, "clip distance: shader interface not understood, keeping the plain pipeline");
      return;
   }

   loc = MAX2(vi.slots_end, fi.slots_end);
   budget = MIN2(lim->maxVertexOutputComponents, lim->maxFragmentInputComponents) / 4;
   if (loc + vi.clip_count > budget) {
      WRAPPER_LOG(info, "clip distance: no room for %u more varyings (%u used, %u available)",
                  vi.clip_count, loc, budget);
      return;
   }

   if (!wrapper_spv_lower_clip_vs(vss->code, vss->size, info->pStages[vs].pName,
                                  loc, &vs_code, &vs_size) ||
       !wrapper_spv_lower_clip_fs(fss->code, fss->size, info->pStages[fs].pName,
                                  vi.clip_count, loc, &fs_code, &fs_size)) {
      WRAPPER_LOG(info, "clip distance: shader pattern not supported, keeping the plain pipeline");
      goto out;
   }

   wrapper_shader_driver_patch(device, vs_code, vs_size);
   wrapper_shader_driver_patch(device, fs_code, fs_size);

   {
      VkShaderModuleCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      };

      ci.codeSize = vs_size;
      ci.pCode = vs_code;
      if (device->dispatch_table.CreateShaderModule(device->dispatch_handle, &ci, NULL, &vm) != VK_SUCCESS) {
         vm = VK_NULL_HANDLE;
         goto out;
      }
      ci.codeSize = fs_size;
      ci.pCode = fs_code;
      if (device->dispatch_table.CreateShaderModule(device->dispatch_handle, &ci, NULL, &fm) != VK_SUCCESS) {
         fm = VK_NULL_HANDLE;
         goto out;
      }
   }

   stages = rw_alloc(rw, info->stageCount * sizeof(*stages));
   if (!stages || !rw_track_modules(rw, vm, fm))
      goto out;

   memcpy(stages, info->pStages, info->stageCount * sizeof(*stages));
   stages[vs].module = vm;
   stages[fs].module = fm;
   info->pStages = stages;
   vm = fm = VK_NULL_HANDLE; /* owned by rw now */

out:
   if (vm != VK_NULL_HANDLE)
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, vm, NULL);
   if (fm != VK_NULL_HANDLE)
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, fm, NULL);
   if (vs_code)
      wrapper_spv_free(vs_code);
   if (fs_code)
      wrapper_spv_free(fs_code);
}

const VkGraphicsPipelineCreateInfo *
wrapper_emu_rewrite_graphics(struct wrapper_device *device, uint32_t count,
                             const VkGraphicsPipelineCreateInfo *infos,
                             struct wrapper_emu_rewrite *rw)
{
   const struct wrapper_emu *emu = &device->emu;

   memset(rw, 0, sizeof(*rw));

   if (!emu->clip_distance && !emu->multi_viewport &&
       !(emu->depth_clamp && device->emu_depth_clip))
      return infos;

   rw->infos = malloc(count * sizeof(*infos));
   if (!rw->infos)
      return infos;
   memcpy(rw->infos, infos, count * sizeof(*infos));

   for (uint32_t i = 0; i < count; i++) {
      VkGraphicsPipelineCreateInfo *info = &rw->infos[i];

      if (emu->depth_clamp && device->emu_depth_clip)
         rewrite_depth_clamp(rw, info);
      if (emu->multi_viewport)
         rewrite_viewports(rw, info);
      if (emu->clip_distance)
         rewrite_clip_distance(device, rw, info);
   }

   return rw->infos;
}

void
wrapper_emu_rewrite_free(struct wrapper_device *device, struct wrapper_emu_rewrite *rw)
{
   for (uint32_t i = 0; i < rw->module_count; i++)
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, rw->modules[i], NULL);
   for (uint32_t i = 0; i < rw->block_count; i++)
      free(rw->blocks[i]);
   free(rw->modules);
   free(rw->blocks);
   free(rw->infos);
   memset(rw, 0, sizeof(*rw));
}

/* ------------------------------------------------- viewport / scissor cmds */

/*
 * With multiViewport emulated the driver has exactly one viewport and one
 * scissor, and no shader can select another, so only element 0 of what the
 * application records is meaningful.
 */
VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport,
                       uint32_t viewportCount, const VkViewport *pViewports)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;

   if (device->emu.multi_viewport) {
      if (firstViewport != 0 || viewportCount == 0)
         return;
      viewportCount = 1;
   }
   device->dispatch_table.CmdSetViewport(wcb->dispatch_handle, firstViewport,
                                         viewportCount, pViewports);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor,
                      uint32_t scissorCount, const VkRect2D *pScissors)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;

   if (device->emu.multi_viewport) {
      if (firstScissor != 0 || scissorCount == 0)
         return;
      scissorCount = 1;
   }
   device->dispatch_table.CmdSetScissor(wcb->dispatch_handle, firstScissor,
                                        scissorCount, pScissors);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetViewportWithCount(VkCommandBuffer commandBuffer, uint32_t viewportCount,
                                const VkViewport *pViewports)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;

   if (device->emu.multi_viewport)
      viewportCount = MIN2(viewportCount, 1);
   device->dispatch_table.CmdSetViewportWithCount(wcb->dispatch_handle,
                                                  viewportCount, pViewports);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetScissorWithCount(VkCommandBuffer commandBuffer, uint32_t scissorCount,
                               const VkRect2D *pScissors)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;

   if (device->emu.multi_viewport)
      scissorCount = MIN2(scissorCount, 1);
   device->dispatch_table.CmdSetScissorWithCount(wcb->dispatch_handle,
                                                 scissorCount, pScissors);
}
