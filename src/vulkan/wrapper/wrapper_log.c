#include <time.h>
#include <errno.h>
#include <link.h>
#include <dlfcn.h>
#include <signal.h>
#include <ucontext.h>
#include <unwind.h>
#include <pthread.h>

#include "wrapper_log.h"
#include "wrapper_util.h"

#define PATH_MAX_SIZE 1024
#define WRAPPER_DIR "/sdcard/wrapper"
#define WRAPPER_SHADER_LOG_PATH WRAPPER_DIR "/shaders"
#define WRAPPER_LOG_PATH WRAPPER_DIR "/logs"
#define WRAPPER_VALIDATION_LOG_PATH WRAPPER_DIR "/validation"

static struct wrapper_log wrapper_log_options[] = {
	{"info", WRAPPER_LOG_INFO},
	{"error", WRAPPER_LOG_ERROR},
	{"shader", WRAPPER_LOG_SHADER},
	{"validation", WRAPPER_LOG_VALIDATION},
	{"apidump", WRAPPER_LOG_APIDUMP},
	{NULL, 0}
};

uint64_t wrapper_log_mask;

static void get_formatted_date_time(char *buf, size_t length) 
{  
   time_t rawtime = time(NULL);
   struct tm *ptm = localtime(&rawtime);

   strftime(buf, 256, "%F_%H-%M-%S", ptm);
}

/* Name of the running executable (the part of /proc/self/cmdline after the
 * last '/' or '\'). Always returns a valid, NUL-terminated string. The result
 * is computed once and cached. */
char *get_executable_name() {
   static char name[PATH_MAX_SIZE];
   static int resolved = 0;

   if (resolved)
      return name;

   char cmdline[PATH_MAX_SIZE] = { 0 };
   int fd = open("/proc/self/cmdline", O_RDONLY | O_CLOEXEC);

   if (fd != -1) {
      ssize_t n = read(fd, cmdline, sizeof(cmdline) - 1);
      close(fd);
      if (n < 0)
         cmdline[0] = '\0';
   }

   const char *base = cmdline;
   char *ptr = strrchr(cmdline, '/');
   if (ptr)
      base = ptr + 1;
   ptr = strrchr(base, '\\');
   if (ptr)
      base = ptr + 1;

   snprintf(name, sizeof(name), "%s", base[0] ? base : "wrapper");
   resolved = 1;
   return name;
}

static unsigned long long get_debug_flag(const char *option) {
   int index = 0;

   while (wrapper_log_options[index].name != NULL) {
      if (!strcmp(wrapper_log_options[index].name, option))
         return wrapper_log_options[index].value;
         
      index++;
   }

   return 0;
}

static void parse_wrapper_debug_str(const char *wrapper_log_level_env) {
   char *copy, *save = NULL;

   wrapper_log_mask = 0;
   if (!wrapper_log_level_env)
      return;

   /* work on a copy: strtok would chop up the environment variable itself */
   copy = strdup(wrapper_log_level_env);
   if (!copy)
      return;

   for (char *option = strtok_r(copy, ",", &save); option != NULL;
        option = strtok_r(NULL, ",", &save))
      wrapper_log_mask |= get_debug_flag(option);

   free(copy);
}

int get_wrapper_log_level(const char *option) {
    uint64_t flag = get_debug_flag(option);

    if (wrapper_log_mask & flag)
       return 1;

    return 0;
}

void write_to_logfile(const char *fmt, const char *level, ...)  {
   static FILE *wrapper_log_file = NULL;
   va_list va_args;

   va_start(va_args, level);

   if (!wrapper_log_file) {
      char *wrapper_log_filename = getenv("WRAPPER_LOG_FILE");
      char date[256];

      get_formatted_date_time(date, 256);
      
      char default_filename[PATH_MAX_SIZE * 2];
      if (!wrapper_log_filename) {
         snprintf(default_filename, sizeof(default_filename), "%s/%s_%s",
                  WRAPPER_LOG_PATH, get_executable_name(), date);
         wrapper_log_filename = default_filename;
      }

      if (!strcmp("stdout", wrapper_log_filename)) {
         wrapper_log_file = stdout;
       }
       else {
         wrapper_log_file = fopen(wrapper_log_filename, "w");
       } 
   }

   if (wrapper_log_file) {
      fprintf(wrapper_log_file, "[%s]: ", level);
      vfprintf(wrapper_log_file, fmt, va_args);
      fprintf(wrapper_log_file, "\n");
      fflush(wrapper_log_file);
   }

   /* When WRAPPER_DIAG is on, also emit to stderr — which for a D3D process is
    * redirected into the single per-game diag file, so the wrapper's own
    * decisions (faked extensions, disabled features) join the device report and
    * the vkd3d/DXVK output in one shareable file. */
   static int diag = -1;
   if (diag == -1)
      diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   if (diag) {
      va_list va2;
      va_start(va2, level);
      fprintf(stderr, "[%s]: ", level);
      vfprintf(stderr, fmt, va2);
      fprintf(stderr, "\n");
      va_end(va2);
   }

   va_end(va_args);
}

void init_wrapper_logging()
{
   char *wrapper_log_level_env = getenv("WRAPPER_LOG_LEVEL");
   parse_wrapper_debug_str(wrapper_log_level_env);

   if (!wrapper_log_mask && !getenv("WRAPPER_LOG_FILE"))
      return;

   CREATE_FOLDER(WRAPPER_DIR, 0770);
   CREATE_FOLDER(WRAPPER_LOG_PATH, 0770);
   CREATE_FOLDER(WRAPPER_SHADER_LOG_PATH, 0770);
   CREATE_FOLDER(WRAPPER_VALIDATION_LOG_PATH, 0770);
}

void dump_shader_code(const uint32_t *code, size_t size) {
   char file[PATH_MAX_SIZE * 2];
   static int index = 0;

   snprintf(file, sizeof(file), "%s/%s_shader_%d.spv", WRAPPER_SHADER_LOG_PATH, get_executable_name(), index);

   FILE *fp = fopen(file, "wb");
   if (fp) {
      fwrite(code, 1, size, fp);
      fclose(fp);
   }

   index++;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
wrapper_debug_utils_messenger(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                              const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
                              void *userData)
{
   static FILE *vvl_log_file = NULL;
   const char* messageIdName = callbackData->pMessageIdName;
   int32_t messageIdNumber = callbackData->messageIdNumber;
   const char* message = callbackData->pMessage;

   if (!vvl_log_file) {
       char date[256];
       char vvl_log_filename[PATH_MAX_SIZE * 2];

       get_formatted_date_time(date, 256);

       snprintf(vvl_log_filename, sizeof(vvl_log_filename), "%s/%s_%s",
                WRAPPER_VALIDATION_LOG_PATH, get_executable_name(), date);
       vvl_log_file = fopen(vvl_log_filename, "w");
   }

   if (vvl_log_file) {
      fprintf(vvl_log_file, "[%s] Code %i : %s\n", messageIdName, messageIdNumber, message);
      fflush(vvl_log_file);
   }

   return 0;
}


/* ------------------------------------------------------------ crash logger */

static struct sigaction crash_old[NSIG];
static uintptr_t crash_lo, crash_hi;       /* address range of this library */
static volatile int crash_busy;

static int
crash_phdr_cb(struct dl_phdr_info *info, size_t size, void *data)
{
   uintptr_t addr = (uintptr_t)data;
   uintptr_t lo = (uintptr_t)-1, hi = 0;
   bool contains = false;

   (void)size;
   for (int i = 0; i < info->dlpi_phnum; i++) {
      const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
      if (ph->p_type != PT_LOAD)
         continue;
      uintptr_t a = info->dlpi_addr + ph->p_vaddr;
      uintptr_t b = a + ph->p_memsz;
      if (a < lo) lo = a;
      if (b > hi) hi = b;
      if (addr >= a && addr < b)
         contains = true;
   }
   if (!contains)
      return 0;
   crash_lo = lo;
   crash_hi = hi;
   return 1;
}

struct crash_bt {
   int n;
};

static _Unwind_Reason_Code
crash_bt_cb(struct _Unwind_Context *ctx, void *arg)
{
   struct crash_bt *st = arg;
   uintptr_t pc = _Unwind_GetIP(ctx);
   Dl_info info;

   if (st->n >= 40)
      return _URC_END_OF_STACK;
   if (pc && dladdr((void *)pc, &info) && info.dli_fname)
      write_to_logfile("  #%02d pc %p %s+0x%lx (%s)", "error", st->n, (void *)pc,
                       info.dli_fname, (unsigned long)(pc - (uintptr_t)info.dli_fbase),
                       info.dli_sname ? info.dli_sname : "?");
   else
      write_to_logfile("  #%02d pc %p", "error", st->n, (void *)pc);
   st->n++;
   return _URC_NO_REASON;
}

static void
crash_describe(const char *what, uintptr_t addr)
{
   Dl_info info;

   if (addr && dladdr((void *)addr, &info) && info.dli_fname)
      write_to_logfile("  %s %p = %s+0x%lx (%s)", "error", what, (void *)addr, info.dli_fname,
                       (unsigned long)(addr - (uintptr_t)info.dli_fbase),
                       info.dli_sname ? info.dli_sname : "?");
   else
      write_to_logfile("  %s %p", "error", what, (void *)addr);
}

static void
crash_handler(int sig, siginfo_t *si, void *uc)
{
   uintptr_t pc = 0, lr = 0, fault = si ? (uintptr_t)si->si_addr : 0;

#if defined(__aarch64__)
   pc = ((ucontext_t *)uc)->uc_mcontext.pc;
   lr = ((ucontext_t *)uc)->uc_mcontext.regs[30];
#elif defined(__x86_64__)
   pc = ((ucontext_t *)uc)->uc_mcontext.gregs[REG_RIP];
#endif

   /* Wine and the emulator fault on purpose all the time (guard pages, code
    * invalidation); only report faults that look like a real bug: a NULL
    * dereference / jump through NULL, or a fault inside this library. */
   if (!crash_busy && (fault < 0x10000 || pc < 0x1000 || (pc >= crash_lo && pc < crash_hi))) {
      struct crash_bt st = { 0 };

      crash_busy = 1;
      write_to_logfile("CRASH: signal %d, fault address %p", "error", sig, (void *)fault);
      crash_describe("pc", pc);
      crash_describe("lr", lr);
      _Unwind_Backtrace(crash_bt_cb, &st);
      crash_busy = 0;
   }

   /* the handler that was there before us (Wine, debuggerd, the emulator) */
   const struct sigaction *old = &crash_old[sig];
   if (old->sa_flags & SA_SIGINFO) {
      old->sa_sigaction(sig, si, uc);
   } else if (old->sa_handler == SIG_DFL) {
      signal(sig, SIG_DFL);
   } else if (old->sa_handler != SIG_IGN) {
      old->sa_handler(sig);
   }
}

static void
crash_install(void)
{
   static const int signals[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE };
   struct sigaction sa;
   const char *env = getenv("WRAPPER_CRASH_LOG");

   if (env && !atoi(env))
      return;

   dl_iterate_phdr(crash_phdr_cb, (void *)(uintptr_t)&crash_install);

   memset(&sa, 0, sizeof(sa));
   sa.sa_sigaction = crash_handler;
   sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
   sigemptyset(&sa.sa_mask);
   for (unsigned i = 0; i < sizeof(signals) / sizeof(signals[0]); i++)
      sigaction(signals[i], &sa, &crash_old[signals[i]]);
}

void
wrapper_crash_log_init(void)
{
   static pthread_once_t once = PTHREAD_ONCE_INIT;

   pthread_once(&once, crash_install);
}
