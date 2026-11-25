#ifndef WRAPPER_LOG_H
#define WRAPPER_LOG_H

#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>
#include <limits.h>
#include <vulkan/vulkan.h>

#define WRAPPER_LOG_INFO (1ull << 0)
#define WRAPPER_LOG_ERROR (1ull << 1)
#define WRAPPER_LOG_SHADER (1ull << 2)
#define WRAPPER_LOG_VALIDATION (1ull << 3)
#define WRAPPER_LOG_APIDUMP (1ull << 5)

struct wrapper_log {
	char *name;
	unsigned long long value;
};

#ifdef __cplusplus
extern "C" {
#endif

/* enabled categories, parsed once from WRAPPER_LOG_LEVEL */
extern uint64_t wrapper_log_mask;

void
init_wrapper_logging(void);

/* Logs raw signal, fault address, pc and lr using an async-signal-safe
 * write path, then restores and re-delivers the signal to its previous
 * disposition. WRAPPER_CRASH_LOG=0 disables crash logging. */
void
wrapper_crash_log_init(void);

void                                                                         
dump_shader_code(const uint32_t *code, size_t size);

int 
get_wrapper_log_level(const char *option);

void
write_to_logfile(const char *fmt, const char *level, ...);

#ifdef __cplusplus
}
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL 
wrapper_debug_utils_messenger(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                              const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
                              void *userData);

/* a single AND on a global: no string compare on the hot paths */
#define WRAPPER_LOGFLAG_info       WRAPPER_LOG_INFO
#define WRAPPER_LOGFLAG_error      WRAPPER_LOG_ERROR
#define WRAPPER_LOGFLAG_shader     WRAPPER_LOG_SHADER
#define WRAPPER_LOGFLAG_validation WRAPPER_LOG_VALIDATION
#define WRAPPER_LOGFLAG_apidump    WRAPPER_LOG_APIDUMP
#define WRAPPER_LOG_LEVEL(s) ((wrapper_log_mask & WRAPPER_LOGFLAG_##s) != 0)

#define WRAPPER_LOG(level, fmt, ...) \
do {\
   if (WRAPPER_LOG_LEVEL(level)) {\
      write_to_logfile(fmt, #level, ##__VA_ARGS__); \
   }\
} while (0)

#endif
