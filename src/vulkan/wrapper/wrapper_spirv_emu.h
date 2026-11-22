/*
 * SPIR-V analysis / lowering used by the feature emulation layer
 * (wrapper_emu.c). Pure functions on SPIR-V words, no Vulkan dependency, so
 * they can be unit tested on a desktop.
 *
 * Every lowering is "all or nothing": on any construct it does not fully
 * understand it returns false and the caller keeps the original shader.
 */
#ifndef WRAPPER_SPIRV_EMU_H
#define WRAPPER_SPIRV_EMU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SPIR-V execution models */
#define WSPV_VERTEX    0
#define WSPV_GEOMETRY  3
#define WSPV_FRAGMENT  4

#define WSPV_SLOTS_UNKNOWN 0xffffffffu

struct wrapper_spv_stage_info {
   bool found;               /* entry point (model + name) exists                */
   bool has_clip;            /* VS: writes ClipDistance, FS: reads it            */
   uint32_t clip_count;      /* size of the ClipDistance array                   */
   bool has_cull;            /* writes CullDistance                              */
   uint32_t cull_count;      /* size of the CullDistance array                   */
   bool has_viewport_index;  /* writes ViewportIndex                             */
   bool has_layer;           /* writes Layer                                     */
   /* first user location after every non-builtin Output (VS) / Input (FS)
    * variable of the entry point, WSPV_SLOTS_UNKNOWN if it can't be computed. */
   uint32_t slots_end;
};

/* Returns false only for malformed modules. info->found tells whether the
 * requested entry point exists. */
bool wrapper_spv_analyze(const uint32_t *code, size_t size_bytes,
                         uint32_t exec_model, const char *entry,
                         struct wrapper_spv_stage_info *info);

/*
 * Vertex shader: moves gl_ClipDistance[] out of the builtin block into a plain
 * float[N] output at `location` (smooth interpolation) and drops the
 * ClipDistance capability. On success *out is a malloc'ed copy of the new
 * module (free with wrapper_spv_free).
 */
bool wrapper_spv_lower_clip_vs(const uint32_t *code, size_t size_bytes,
                               const char *entry, uint32_t location,
                               uint32_t **out, size_t *out_size_bytes);

/* Same for gl_CullDistance[] (used by the geometry-shader cull emulation). */
bool wrapper_spv_lower_cull_vs(const uint32_t *code, size_t size_bytes,
                               const char *entry, uint32_t location,
                               uint32_t **out, size_t *out_size_bytes);

/*
 * Fragment shader: adds a float[count] input at `location` and discards the
 * invocation when any element is negative (that is exactly what clipping
 * against the planes does to a fragment). A ClipDistance input the shader
 * reads itself is redirected to the same varying.
 */
bool wrapper_spv_lower_clip_fs(const uint32_t *code, size_t size_bytes,
                               const char *entry, uint32_t count,
                               uint32_t location,
                               uint32_t **out, size_t *out_size_bytes);

/*
 * Pass-through geometry shader built from the vertex shader's output
 * interface. It copies Position (PointSize), every Location varying and the
 * primitive id of each input vertex and, depending on out_mode:
 *   WSPV_GS_PASS    emits the primitive unchanged (cull-distance emulation)
 *   WSPV_GS_WIRE    triangle -> closed line strip     (polygonMode LINE)
 *   WSPV_GS_POINTS  triangle -> its three corners     (polygonMode POINT)
 * With cull_location set it first drops the primitive when, for one plane of
 * the float[cull_count] varying at that location (written by
 * wrapper_spv_lower_cull_vs), every vertex is negative -- gl_CullDistance.
 * Refuses (false) anything it can not forward exactly: struct varyings,
 * Layer / ViewportIndex / ClipDistance / CullDistance builtins.
 */
#define WSPV_GS_PASS    0
#define WSPV_GS_WIRE    1
#define WSPV_GS_POINTS  2
#define WSPV_GS_NO_CULL 0xffffffffu

struct wrapper_spv_gs_params {
   uint32_t in_prim;            /* 1 points, 2 lines, 3 triangles               */
   uint32_t out_mode;           /* WSPV_GS_*; WIRE/POINTS need in_prim == 3     */
   uint32_t cull_location;      /* WSPV_GS_NO_CULL if there is nothing to cull  */
   uint32_t cull_count;
   bool forward_point_size;     /* the device can write PointSize from a GS     */
};

bool wrapper_spv_make_gs(const uint32_t *code, size_t size_bytes,
                         const char *vs_entry,
                         const struct wrapper_spv_gs_params *params,
                         uint32_t **out, size_t *out_size_bytes);

/* One cheap pass over capabilities and entry points. */
#define WSPV_SCAN_VERTEX            (1u << 0)
#define WSPV_SCAN_FRAGMENT          (1u << 1)
#define WSPV_SCAN_GEOMETRY          (1u << 2)
#define WSPV_SCAN_TESS              (1u << 3)
#define WSPV_SCAN_COMPUTE           (1u << 4)
#define WSPV_SCAN_VIEWPORT_LAYER_EXT (1u << 5)  /* ShaderViewportIndexLayerEXT */
#define WSPV_SCAN_MULTI_VIEWPORT    (1u << 6)   /* MultiViewport               */
#define WSPV_SCAN_CLIP_DISTANCE     (1u << 7)   /* ClipDistance capability     */
#define WSPV_SCAN_CULL_DISTANCE     (1u << 8)   /* CullDistance capability     */
uint32_t wrapper_spv_scan(const uint32_t *code, size_t size_bytes);

/* ShaderViewportIndexLayerEXT (+ its extension) -> ShaderViewportIndex /
 * ShaderLayer, the Vulkan 1.2 core form of the same thing. */
bool wrapper_spv_translate_viewport_layer(const uint32_t *code, size_t size_bytes,
                                          uint32_t **out, size_t *out_size_bytes);

/* Geometry shader writing gl_ViewportIndex on a device without multiViewport:
 * the output becomes a private variable (viewport 0 is used). `entry` NULL =
 * every geometry entry point of the module. */
bool wrapper_spv_strip_viewport_index_gs(const uint32_t *code, size_t size_bytes,
                                         const char *entry,
                                         uint32_t **out, size_t *out_size_bytes);

void wrapper_spv_free(uint32_t *p);

#ifdef __cplusplus
}
#endif

#endif
