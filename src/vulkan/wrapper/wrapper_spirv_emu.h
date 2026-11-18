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

void wrapper_spv_free(uint32_t *p);

#ifdef __cplusplus
}
#endif

#endif
