/*
 * Feature emulation.
 *
 * The Mali driver is asked first. A feature it reports is passed through
 * untouched; a feature it lacks is implemented here, for real, and only then
 * reported to the application. Nothing is ever claimed that nothing backs.
 *
 *   feature              what the GPU does not have, and what stands in for it
 *   -------------------  ---------------------------------------------------
 *   shaderClipDistance   vertex shader: gl_ClipDistance[] -> plain float[N]
 *                        varying; fragment shader: discard when any element is
 *                        negative (a fragment is inside the clipped polygon
 *                        exactly when every plane distance is >= 0).
 *   shaderCullDistance   vertex shader: gl_CullDistance[] -> float[N] varying;
 *                        a geometry shader built from the vertex shader's own
 *                        interface drops the primitive when, for one plane,
 *                        every vertex is negative (that is the definition).
 *   fillModeNonSolid     polygonMode LINE / POINT: the same geometry shader
 *                        turns each triangle into its three edges (a closed
 *                        line strip) or its three corners.
 *   multiDrawIndirect    vkCmdDraw[Indexed]Indirect with drawCount > 1 becomes
 *                        drawCount single indirect draws (same order, same
 *                        parameters read at execution time).
 *   multiViewport        viewport / scissor arrays are accepted (up to 16) and
 *                        element 0 is the one the hardware uses. A geometry
 *                        shader that writes gl_ViewportIndex has the write
 *                        dropped (see wrapper_CreateShaderModule); that is the
 *                        one place where "everything is viewport 0" is not
 *                        exact, and it is only reached by shaders that select
 *                        a viewport.
 *   depthClamp           only when the driver lacks it: depthClampEnable
 *                        becomes depthClipEnable = false (VK_EXT_depth_clip_enable).
 *   VK_EXT_shader_viewport_index_layer
 *                        the Vulkan 1.2 core form (shaderOutputViewportIndex /
 *                        shaderOutputLayer) of the same thing; the extension
 *                        name is advertised and the SPIR-V capability is
 *                        translated.
 *
 * depthBiasClamp is native on the GPU this wrapper targets; where a driver
 * lacks it, it stays off (it can not be reproduced outside the rasterizer).
 *
 * Everything per pipeline happens in vkCreateGraphicsPipelines(). A pipeline
 * is only rewritten when it needs it, and if a rewrite can not be done
 * completely the pipeline is created exactly as the application described it
 * (apart from state the driver would refuse).
 */

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "wrapper_spirv_emu.h"
#include "spirv_patcher.hpp"
#include "vk_alloc.h"
#include "vk_util.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/simple_mtx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ plan */

static const char *
state_str(bool native, bool emulated)
{
   return native ? "native" : (emulated ? "emulated" : "off");
}

/* Which features the application will see, and how each one is backed. One
 * line, once per physical device. */
static void
log_feature_plan(const struct vk_features *base, const struct vk_device_extension_table *ext,
                 const struct vk_features *sup, const struct vk_device_extension_table *sup_ext,
                 const struct wrapper_emu *emu)
{
   WRAPPER_LOG(info,
      "Features: clip=%d(%s) cull=%d(%s) depthClamp=%d(%s) depthBiasClamp=%d(%s) "
      "fillModeNonSolid=%d(%s) multiViewport=%d(%s) viewportIndexExt=%d(%s) "
      "multiDrawIndirect=%d(%s) | geometry=%d tessellation=%d drawIndirectFirstInstance=%d imageGatherExtended=%d",
      sup->shaderClipDistance, state_str(base->shaderClipDistance, emu->clip_distance),
      sup->shaderCullDistance, state_str(base->shaderCullDistance, emu->cull_distance),
      sup->depthClamp, state_str(base->depthClamp, emu->depth_clamp),
      sup->depthBiasClamp, state_str(base->depthBiasClamp, false),
      sup->fillModeNonSolid, state_str(base->fillModeNonSolid, emu->fill_mode),
      sup->multiViewport, state_str(base->multiViewport, emu->multi_viewport),
      sup_ext->EXT_shader_viewport_index_layer,
      state_str(ext->EXT_shader_viewport_index_layer, emu->viewport_index_layer),
      sup->multiDrawIndirect, state_str(base->multiDrawIndirect, emu->multi_draw_indirect),
      sup->geometryShader, sup->tessellationShader,
      sup->drawIndirectFirstInstance, sup->shaderImageGatherExtended);
}

void
wrapper_emu_init_physical(struct wrapper_physical_device *pdevice)
{
   const struct vk_features *base = &pdevice->base_supported_features;
   const struct vk_device_extension_table *ext = &pdevice->base_supported_extensions;
   struct vk_features *sup = &pdevice->vk.supported_features;
   struct wrapper_emu *emu = &pdevice->emu;
   const char *env = getenv("WRAPPER_FEATURE_EMULATION");
   const uint32_t api = pdevice->properties2.properties.apiVersion;

   memset(emu, 0, sizeof(*emu));

   /* WRAPPER_FEATURE_EMULATION=0: report exactly what the driver reports */
   if (!env || atoi(env)) {
      if (!base->shaderClipDistance) {
         emu->clip_distance = true;
         sup->shaderClipDistance = true;
      }

      /* cull distance and wireframe are built on a geometry shader */
      if (base->geometryShader) {
         if (!base->shaderCullDistance) {
            emu->cull_distance = true;
            sup->shaderCullDistance = true;
         }
         if (!base->fillModeNonSolid) {
            emu->fill_mode = true;
            sup->fillModeNonSolid = true;
         }
      }

      if (!base->depthClamp && ext->EXT_depth_clip_enable && base->depthClipEnable) {
         emu->depth_clamp = true;
         sup->depthClamp = true;
      }

      if (!base->multiViewport) {
         emu->multi_viewport = true;
         sup->multiViewport = true;
      }

      if (!base->multiDrawIndirect) {
         emu->multi_draw_indirect = true;
         sup->multiDrawIndirect = true;
      }

      /* promoted to core in 1.2: same capability, spelled shaderOutput*  */
      if (!ext->EXT_shader_viewport_index_layer && api >= VK_API_VERSION_1_2 &&
          base->shaderOutputViewportIndex && base->shaderOutputLayer) {
         emu->viewport_index_layer = true;
         pdevice->vk.supported_extensions.EXT_shader_viewport_index_layer = true;
      }
   }

   log_feature_plan(base, ext, sup, &pdevice->vk.supported_extensions, emu);
}

void
wrapper_emu_patch_limits(const struct wrapper_physical_device *pdevice,
                         VkPhysicalDeviceLimits *limits)
{
   const struct wrapper_emu *emu = &pdevice->emu;

   /* the Vulkan minimums that go with the features */
   if (emu->clip_distance || emu->cull_distance) {
      if (emu->clip_distance)
         limits->maxClipDistances = MAX2(limits->maxClipDistances, 8);
      if (emu->cull_distance)
         limits->maxCullDistances = MAX2(limits->maxCullDistances, 8);
      limits->maxCombinedClipAndCullDistances =
         MAX2(limits->maxCombinedClipAndCullDistances, 8);
   }
   if (emu->multi_viewport)
      limits->maxViewports = MAX2(limits->maxViewports, 16);
   if (emu->multi_draw_indirect)
      limits->maxDrawIndirectCount = MAX2(limits->maxDrawIndirectCount, 65535);
}

/* ------------------------------------------------------- SPIR-V registry */

struct wrapper_emu_shader {
   struct list_head link;
   VkShaderModule module;
   uint32_t *code;      /* the application's SPIR-V, before any workaround */
   size_t size;         /* bytes */
   /* what wrapper_spv_analyze() says about the "main" vertex entry, once */
   bool vs_info_valid;
   bool vs_info_ok;
   struct wrapper_spv_stage_info vs_info;
};

/* a message that is worth one line, not one per pipeline */
#define EMU_LOG_ONCE(...) do { \
   static bool once_; \
   if (!once_) { once_ = true; WRAPPER_LOG(info, __VA_ARGS__); } \
} while (0)

bool
wrapper_emu_wants_shaders(const struct wrapper_device *device)
{
   return device->emu.clip_distance || device->emu.cull_distance || device->emu.fill_mode;
}

void
wrapper_emu_init_device(struct wrapper_device *device)
{
   device->emu = device->physical->emu;
   simple_mtx_init(&device->shader_mutex, mtx_plain);
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
   if (device->shader_table) {
      list_for_each_entry_safe(struct wrapper_emu_shader, s, &device->shader_list, link) {
         list_del(&s->link);
         shader_free(s);
      }
      _mesa_hash_table_u64_destroy(device->shader_table);
      device->shader_table = NULL;
   }
   simple_mtx_destroy(&device->shader_mutex);
}

void
wrapper_emu_add_shader(struct wrapper_device *device, VkShaderModule module,
                       const uint32_t *code, size_t size_bytes)
{
   struct wrapper_emu_shader *s, *old;
   uint32_t flags;

   if (!device->shader_table || module == VK_NULL_HANDLE)
      return;

   /* a rewrite starts from a vertex shader (clip / cull / wireframe) or, for
    * clip distance, from the fragment shader that goes with it: nothing else
    * is worth keeping */
   flags = wrapper_spv_scan(code, size_bytes);
   if (!(flags & WSPV_SCAN_VERTEX) &&
       !((flags & WSPV_SCAN_FRAGMENT) && device->emu.clip_distance))
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

   simple_mtx_lock(&device->shader_mutex);
   old = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   if (old) {
      list_del(&old->link);
      _mesa_hash_table_u64_remove(device->shader_table, (uint64_t)module);
   }
   list_add(&s->link, &device->shader_list);
   _mesa_hash_table_u64_insert(device->shader_table, (uint64_t)module, s);
   simple_mtx_unlock(&device->shader_mutex);

   if (old)
      shader_free(old);
}

void
wrapper_emu_remove_shader(struct wrapper_device *device, VkShaderModule module)
{
   struct wrapper_emu_shader *s;

   if (!device->shader_table || module == VK_NULL_HANDLE)
      return;

   simple_mtx_lock(&device->shader_mutex);
   s = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   if (s) {
      list_del(&s->link);
      _mesa_hash_table_u64_remove(device->shader_table, (uint64_t)module);
   }
   simple_mtx_unlock(&device->shader_mutex);

   if (s)
      shader_free(s);
}

static struct wrapper_emu_shader *
shader_lookup(struct wrapper_device *device, VkShaderModule module)
{
   struct wrapper_emu_shader *s;

   if (!device->shader_table || module == VK_NULL_HANDLE)
      return NULL;
   simple_mtx_lock(&device->shader_mutex);
   s = _mesa_hash_table_u64_search(device->shader_table, (uint64_t)module);
   simple_mtx_unlock(&device->shader_mutex);
   return s;
}

/* Vertex-stage facts about a registered module; "main" (what every compiler
 * of interest uses) is analysed once per module, anything else every time. */
static bool
shader_vs_info(struct wrapper_emu_shader *s, const char *entry,
               struct wrapper_spv_stage_info *out)
{
   if (entry && !strcmp(entry, "main")) {
      if (!s->vs_info_valid) {
         s->vs_info_ok = wrapper_spv_analyze(s->code, s->size, WSPV_VERTEX, "main", &s->vs_info);
         s->vs_info_valid = true;
      }
      *out = s->vs_info;
      return s->vs_info_ok && out->found;
   }
   return wrapper_spv_analyze(s->code, s->size, WSPV_VERTEX, entry, out) && out->found;
}

/*
 * SPIR-V changes every module bound for the driver needs, whichever pipeline
 * ends up using it. *words is a malloc'ed, writable module; it may be shrunk
 * or replaced (the old buffer is freed).
 *
 *  - ClipDistance / CullDistance the GPU can not compile are dropped: this is
 *    the fallback for a pipeline the emulation could not lower
 *  - SPV_EXT_shader_viewport_index_layer becomes the Vulkan 1.2 capabilities
 *  - a geometry shader writing gl_ViewportIndex without multiViewport keeps
 *    running, the index goes to a private variable (viewport 0)
 */
void
wrapper_emu_fixup_module(const struct wrapper_device *device, uint32_t **words, size_t *bytes)
{
   const struct vk_features *base = &device->physical->base_supported_features;
   const struct vk_device_extension_table *ext = &device->physical->base_supported_extensions;
   uint32_t flags;
   uint32_t *n;
   size_t nsize;

   if (*bytes < 5 * sizeof(uint32_t))
      return;
   flags = wrapper_spv_scan(*words, *bytes);
   if (!flags)
      return;

   if ((flags & WSPV_SCAN_CLIP_DISTANCE) && !base->shaderClipDistance)
      remove_ClipDistance(*words, bytes);
   if ((flags & WSPV_SCAN_CULL_DISTANCE) && !base->shaderCullDistance)
      remove_CullDistance(*words, bytes);

   if ((flags & WSPV_SCAN_VIEWPORT_LAYER_EXT) && !ext->EXT_shader_viewport_index_layer &&
       wrapper_spv_translate_viewport_layer(*words, *bytes, &n, &nsize)) {
      free(*words);
      *words = n;
      *bytes = nsize;
   }

   if ((flags & WSPV_SCAN_GEOMETRY) && (flags & WSPV_SCAN_MULTI_VIEWPORT) &&
       !base->multiViewport &&
       wrapper_spv_strip_viewport_index_gs(*words, *bytes, NULL, &n, &nsize)) {
      free(*words);
      *words = n;
      *bytes = nsize;
   }
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

/* Hands a temporary shader module to `rw`; it is destroyed with the rewrite. */
static bool
rw_track_module(struct wrapper_emu_rewrite *rw, VkShaderModule m)
{
   if (rw->module_count + 1 > rw->module_cap) {
      uint32_t cap = rw->module_cap ? rw->module_cap * 2 : 8;
      VkShaderModule *n = realloc(rw->modules, cap * sizeof(*n));
      if (!n)
         return false;
      rw->modules = n;
      rw->module_cap = cap;
   }
   rw->modules[rw->module_count++] = m;
   return true;
}

/* A module for the driver out of SPIR-V the wrapper produced or kept. */
static bool
rw_make_module(struct wrapper_device *device, struct wrapper_emu_rewrite *rw,
               const uint32_t *code, size_t size, VkShaderModule *out)
{
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
   uint32_t *words = malloc(size);
   VkResult res;

   *out = VK_NULL_HANDLE;
   if (!words)
      return false;
   memcpy(words, code, size);
   wrapper_shader_driver_patch(device, words, size);
   wrapper_emu_fixup_module(device, &words, &size);

   ci.codeSize = size;
   ci.pCode = words;
   res = device->dispatch_table.CreateShaderModule(device->dispatch_handle, &ci, NULL, out);
   free(words);
   if (res != VK_SUCCESS) {
      *out = VK_NULL_HANDLE;
      return false;
   }
   if (!rw_track_module(rw, *out)) {
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, *out, NULL);
      *out = VK_NULL_HANDLE;
      return false;
   }
   return true;
}

/* depthClamp -> depth clip control, polygonMode LINE/POINT -> FILL (the
 * geometry shader draws the lines / points; the driver must never be asked for
 * a polygon mode it lacks). One copy of the rasterization state. */
static void
rewrite_rasterization(struct wrapper_device *device, struct wrapper_emu_rewrite *rw,
                      VkGraphicsPipelineCreateInfo *info)
{
   const struct wrapper_emu *emu = &device->emu;
   const VkPipelineRasterizationStateCreateInfo *rs = info->pRasterizationState;
   const bool clamp = emu->depth_clamp && device->emu_depth_clip && rs && rs->depthClampEnable;
   const bool fill = emu->fill_mode && rs && rs->polygonMode != VK_POLYGON_MODE_FILL;
   VkPipelineRasterizationStateCreateInfo *n;

   if (!clamp && !fill)
      return;

   n = rw_alloc(rw, sizeof(*n));
   if (!n)
      return;
   *n = *rs;

   if (fill)
      n->polygonMode = VK_POLYGON_MODE_FILL;

   if (clamp) {
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

static bool
has_dynamic_state(const VkGraphicsPipelineCreateInfo *info, VkDynamicState state)
{
   const VkPipelineDynamicStateCreateInfo *ds = info->pDynamicState;

   for (uint32_t i = 0; ds && i < ds->dynamicStateCount; i++)
      if (ds->pDynamicStates[i] == state)
         return true;
   return false;
}

/* the code of a stage as the rewrite currently has it */
struct stage_code {
   const uint32_t *code;    /* registry's copy or `owned` */
   size_t size;
   uint32_t *owned;         /* malloc'ed by a lowering, or NULL */
   bool changed;
};

static void
stage_replace(struct stage_code *s, uint32_t *words, size_t size)
{
   if (s->owned)
      wrapper_spv_free(s->owned);
   s->owned = words;
   s->code = words;
   s->size = size;
   s->changed = true;
}

/*
 * The shader side of a pipeline:
 *
 *  1. clip distance      VS + FS lowered (vertex + fragment pipelines only)
 *  2. cull distance /    a geometry shader built from the vertex shader's
 *     polygonMode        outputs is added; the vertex shader first exports the
 *     LINE / POINT       cull distances as a varying the geometry shader reads
 *
 * Anything that can't be done exactly leaves that part of the pipeline as the
 * application wrote it.
 */
static void
rewrite_shaders(struct wrapper_device *device, struct wrapper_emu_rewrite *rw,
                VkGraphicsPipelineCreateInfo *info, VkPolygonMode app_polygon_mode)
{
   const struct wrapper_emu *emu = &device->emu;
   const VkPhysicalDeviceLimits *lim = &device->physical->properties2.properties.limits;
   const VkPipelineRasterizationStateCreateInfo *rs = info->pRasterizationState;
   struct wrapper_emu_shader *vss, *fss = NULL;
   struct wrapper_spv_stage_info vi, fi;
   struct stage_code vsc = {0}, fsc = {0}, vs_cull = {0}, gsc = {0};
   VkPipelineShaderStageCreateInfo *stages;
   VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE, gm = VK_NULL_HANDLE;
   const char *vs_name;
   int vs = -1, fs = -1;
   bool extra_stage = false;
   uint32_t prim = 0;

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
         extra_stage = true;   /* only vertex (+ fragment) pipelines are rewritten */
         break;
      default:
         break;
      }
   }
   /* nothing is rasterized: clipping, culling and polygon mode can't be seen */
   if (vs < 0 || extra_stage || (rs && rs->rasterizerDiscardEnable))
      return;

   vs_name = info->pStages[vs].pName;
   vss = shader_lookup(device, info->pStages[vs].module);
   if (!vss || !shader_vs_info(vss, vs_name, &vi))
      return;

   if (info->pInputAssemblyState && !has_dynamic_state(info, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY)) {
      switch (info->pInputAssemblyState->topology) {
      case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
         prim = 1;
         break;
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
         prim = 2;
         break;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
         prim = 3;
         break;
      default:
         break;
      }
   }

   const bool want_clip = emu->clip_distance && vi.has_clip && vi.clip_count > 0 && fs >= 0;
   const bool want_cull = emu->cull_distance && device->emu_gs && vi.has_cull &&
                          vi.cull_count > 0 && prim != 0;
   const bool want_fill = emu->fill_mode && device->emu_gs && prim == 3 &&
                          (app_polygon_mode == VK_POLYGON_MODE_LINE ||
                           app_polygon_mode == VK_POLYGON_MODE_POINT) &&
                          !has_dynamic_state(info, VK_DYNAMIC_STATE_POLYGON_MODE_EXT);
   if (!want_clip && !want_cull && !want_fill)
      return;

   vsc.code = vss->code;
   vsc.size = vss->size;

   /* ---- 1. clip distance ---- */
   if (want_clip) {
      uint32_t *vcode = NULL, *fcode = NULL;
      size_t vsize = 0, fsize = 0;
      uint32_t loc, budget;

      fss = shader_lookup(device, info->pStages[fs].module);
      if (!fss || !wrapper_spv_analyze(fss->code, fss->size, WSPV_FRAGMENT,
                                       info->pStages[fs].pName, &fi) || !fi.found) {
         EMU_LOG_ONCE("clip distance: fragment shader unknown, keeping the plain pipeline");
      } else if (vi.slots_end == WSPV_SLOTS_UNKNOWN || fi.slots_end == WSPV_SLOTS_UNKNOWN) {
         EMU_LOG_ONCE("clip distance: shader interface not understood, keeping the plain pipeline");
      } else {
         loc = MAX2(vi.slots_end, fi.slots_end);
         budget = MIN2(lim->maxVertexOutputComponents, lim->maxFragmentInputComponents) / 4;
         if (loc + vi.clip_count > budget) {
            EMU_LOG_ONCE("clip distance: no room for %u more varyings (%u used, %u available)",
                         vi.clip_count, loc, budget);
         } else if (!wrapper_spv_lower_clip_vs(vss->code, vss->size, vs_name, loc,
                                               &vcode, &vsize) ||
                    !wrapper_spv_lower_clip_fs(fss->code, fss->size, info->pStages[fs].pName,
                                               vi.clip_count, loc, &fcode, &fsize)) {
            EMU_LOG_ONCE("clip distance: shader pattern not supported, keeping the plain pipeline");
            if (vcode)
               wrapper_spv_free(vcode);
            if (fcode)
               wrapper_spv_free(fcode);
         } else {
            stage_replace(&vsc, vcode, vsize);
            stage_replace(&fsc, fcode, fsize);
         }
      }
   }

   /* ---- 2. geometry shader: cull distance and polygon mode ---- */
   if ((want_cull || want_fill) && !vi.has_layer && !vi.has_viewport_index) {
      struct wrapper_spv_gs_params gp = {
         .in_prim = prim,
         .out_mode = want_fill ? (app_polygon_mode == VK_POLYGON_MODE_LINE ? WSPV_GS_WIRE
                                                                           : WSPV_GS_POINTS)
                               : WSPV_GS_PASS,
         .cull_location = WSPV_GS_NO_CULL,
         .cull_count = 0,
         .forward_point_size = device->emu_gs_point_size,
      };
      struct wrapper_spv_stage_info cur;
      const struct stage_code *gs_vs = &vsc;
      uint32_t out_vertices = gp.out_mode == WSPV_GS_WIRE ? 4 : prim;
      uint32_t out_comps;

      if (!wrapper_spv_analyze(vsc.code, vsc.size, WSPV_VERTEX, vs_name, &cur) || !cur.found ||
          cur.slots_end == WSPV_SLOTS_UNKNOWN) {
         EMU_LOG_ONCE("geometry shader emulation: vertex shader interface not understood");
         goto commit;
      }

      /* what a geometry shader would have to carry per vertex */
      out_comps = cur.slots_end * 4 + 8;
      if (out_vertices > lim->maxGeometryOutputVertices ||
          out_comps > lim->maxGeometryOutputComponents ||
          out_comps * out_vertices > lim->maxGeometryTotalOutputComponents) {
         EMU_LOG_ONCE("geometry shader emulation: %u components x %u vertices exceed the geometry limits",
                      out_comps, out_vertices);
         goto commit;
      }

      if (want_cull) {
         /* the cull varying only travels from the vertex shader to the geometry
          * shader (it is not forwarded), so it only has to clear the vertex
          * shader's own outputs -- the clip varyings included */
         uint32_t loc = cur.slots_end, budget, *code = NULL;
         size_t size = 0;

         budget = MIN2(lim->maxVertexOutputComponents, lim->maxGeometryInputComponents) / 4;
         if (loc + cur.cull_count > budget) {
            EMU_LOG_ONCE("cull distance: no room for %u more varyings (%u used, %u available)",
                         cur.cull_count, loc, budget);
         } else if (!wrapper_spv_lower_cull_vs(vsc.code, vsc.size, vs_name, loc, &code, &size)) {
            EMU_LOG_ONCE("cull distance: vertex shader pattern not supported");
         } else {
            vs_cull.code = code;
            vs_cull.size = size;
            vs_cull.owned = code;
            gp.cull_location = loc;
            gp.cull_count = cur.cull_count;
            gs_vs = &vs_cull;
         }
      }

      if (gp.cull_location == WSPV_GS_NO_CULL && !want_fill)
         goto commit;               /* cull was the only reason and it could not be done */

      {
         uint32_t *gcode = NULL;
         size_t gsize = 0;

         if (!wrapper_spv_make_gs(gs_vs->code, gs_vs->size, vs_name, &gp, &gcode, &gsize)) {
            EMU_LOG_ONCE("geometry shader emulation: this vertex shader's outputs can not be forwarded "
                         "(struct varyings, layer / viewport index)");
         } else {
            gsc.code = gsc.owned = gcode;
            gsc.size = gsize;
            gsc.changed = true;
            if (gs_vs == &vs_cull) {           /* the vertex shader now exports the cull varying */
               stage_replace(&vsc, vs_cull.owned, vs_cull.size);
               vs_cull.owned = NULL;
            }
         }
      }
   }

commit:
   if (!vsc.changed && !fsc.changed && !gsc.changed)
      goto out;

   if (vsc.changed && !rw_make_module(device, rw, vsc.code, vsc.size, &vm))
      goto out;
   if (fsc.changed && !rw_make_module(device, rw, fsc.code, fsc.size, &fm))
      goto out;
   if (gsc.changed && !rw_make_module(device, rw, gsc.code, gsc.size, &gm))
      goto out;

   stages = rw_alloc(rw, (info->stageCount + 1) * sizeof(*stages));
   if (!stages)
      goto out;
   memcpy(stages, info->pStages, info->stageCount * sizeof(*stages));
   if (vm)
      stages[vs].module = vm;
   if (fm)
      stages[fs].module = fm;
   if (gm) {
      stages[info->stageCount].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[info->stageCount].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
      stages[info->stageCount].module = gm;
      stages[info->stageCount].pName = "main";
      info->stageCount++;
   }
   info->pStages = stages;

out:
   if (vsc.owned)
      wrapper_spv_free(vsc.owned);
   if (fsc.owned)
      wrapper_spv_free(fsc.owned);
   if (vs_cull.owned)
      wrapper_spv_free(vs_cull.owned);
   if (gsc.owned)
      wrapper_spv_free(gsc.owned);
}

/* Does this create info need anything from the emulation layer at all? */
static bool
info_needs_rewrite(const struct wrapper_device *device, const VkGraphicsPipelineCreateInfo *info)
{
   const struct wrapper_emu *emu = &device->emu;
   const VkPipelineRasterizationStateCreateInfo *rs = info->pRasterizationState;
   const VkPipelineViewportStateCreateInfo *vp = info->pViewportState;

   if (rs && ((emu->depth_clamp && device->emu_depth_clip && rs->depthClampEnable) ||
              (emu->fill_mode && rs->polygonMode != VK_POLYGON_MODE_FILL)))
      return true;
   if (emu->multi_viewport && vp && (vp->viewportCount > 1 || vp->scissorCount > 1))
      return true;
   /* clip / cull distance live in the shaders: only the registry can tell */
   return device->shader_table != NULL && info->stageCount > 0;
}

const VkGraphicsPipelineCreateInfo *
wrapper_emu_rewrite_graphics(struct wrapper_device *device, uint32_t count,
                             const VkGraphicsPipelineCreateInfo *infos,
                             struct wrapper_emu_rewrite *rw)
{
   const struct wrapper_emu *emu = &device->emu;

   memset(rw, 0, sizeof(*rw));

   if (!emu->clip_distance && !emu->cull_distance && !emu->fill_mode &&
       !emu->multi_viewport && !(emu->depth_clamp && device->emu_depth_clip))
      return infos;

   for (uint32_t i = 0; i < count; i++) {
      /* work on a private copy of this one create info; the array of the
       * application is only duplicated once something really changed */
      VkGraphicsPipelineCreateInfo info = infos[i];
      const VkPolygonMode mode = info.pRasterizationState ? info.pRasterizationState->polygonMode
                                                          : VK_POLYGON_MODE_FILL;

      if (!info_needs_rewrite(device, &infos[i]))
         continue;

      rewrite_rasterization(device, rw, &info);
      if (emu->multi_viewport)
         rewrite_viewports(rw, &info);
      rewrite_shaders(device, rw, &info, mode);

      if (info.pRasterizationState == infos[i].pRasterizationState &&
          info.pViewportState == infos[i].pViewportState &&
          info.pStages == infos[i].pStages && info.stageCount == infos[i].stageCount)
         continue;                       /* nothing to hand to the driver but the original */

      if (!rw->infos) {
         rw->infos = malloc(count * sizeof(*infos));
         if (!rw->infos)
            return infos;
         memcpy(rw->infos, infos, count * sizeof(*infos));
      }
      rw->infos[i] = info;
   }

   return rw->infos ? rw->infos : infos;
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

/* ----------------------------------------------------- multiDrawIndirect */

/*
 * The GPU only takes drawCount <= 1. drawCount single indirect draws read the
 * same commands, in the same order, each at the moment it executes -- the
 * result is the one a native multi draw gives.
 */
VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                        VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   const struct vk_device_dispatch_table *t = &wcb->device->dispatch_table;

   if (drawCount <= 1 || !wcb->device->emu.multi_draw_indirect) {
      t->CmdDrawIndirect(wcb->dispatch_handle, buffer, offset, drawCount, stride);
      return;
   }
   if (stride == 0)
      stride = sizeof(VkDrawIndirectCommand);
   for (uint32_t i = 0; i < drawCount; i++)
      t->CmdDrawIndirect(wcb->dispatch_handle, buffer, offset + (VkDeviceSize)i * stride, 1, 0);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                               VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   const struct vk_device_dispatch_table *t = &wcb->device->dispatch_table;

   if (drawCount <= 1 || !wcb->device->emu.multi_draw_indirect) {
      t->CmdDrawIndexedIndirect(wcb->dispatch_handle, buffer, offset, drawCount, stride);
      return;
   }
   if (stride == 0)
      stride = sizeof(VkDrawIndexedIndirectCommand);
   for (uint32_t i = 0; i < drawCount; i++)
      t->CmdDrawIndexedIndirect(wcb->dispatch_handle, buffer, offset + (VkDeviceSize)i * stride, 1, 0);
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
