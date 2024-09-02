#include "hetu/common/logging.h"
#include <algorithm>
#include <cctype>

namespace hetu {
namespace logging {

LOG_LEVEL __HT_INTERNAL_LOG_LEVEL = HT_DEFAULT_LOG_LEVEL;

namespace {
struct LogLevelInitializer {
  explicit LogLevelInitializer() {
    LOG_LEVEL log_level = HT_DEFAULT_LOG_LEVEL;
    char* env = std::getenv("HETU_INTERNAL_LOG_LEVEL");
    if (env != nullptr) {
      std::string level = env;
      std::transform(level.begin(), level.end(), level.begin(), ::toupper);
      if (level == "TRACE")
        log_level = LOG_LEVEL::TRACE;
      else if (level == "DEBUG")
        log_level = LOG_LEVEL::DEBUG;
      else if (level == "INFO")
        log_level = LOG_LEVEL::INFO;
      else if (level == "WARN")
        log_level = LOG_LEVEL::WARN;
      else if (level == "ERROR")
        log_level = LOG_LEVEL::ERROR;
      else if (level == "FATAL")
        log_level = LOG_LEVEL::FATAL;
      else
        throw std::runtime_error("Unknown logging level: " + level);
    }
    set_log_level(log_level);
  }
};
static LogLevelInitializer _log_level_initializer;
} // namespace

} // namespace logging
} // namespace hetu
