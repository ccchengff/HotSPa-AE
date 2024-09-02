#pragma once

#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include "hetu/common/collection_streaming.h"

namespace hetu {
namespace logging {

enum class LOG_LEVEL : int8_t {
  TRACE = 0,
  DEBUG,
  INFO,
  WARN,
  ERROR,
  FATAL
};

#define HT_DEFAULT_LOG_LEVEL (LOG_LEVEL::INFO)

extern LOG_LEVEL __HT_INTERNAL_LOG_LEVEL;

inline LOG_LEVEL get_log_level() {
  return __HT_INTERNAL_LOG_LEVEL;
}

inline void set_log_level(LOG_LEVEL level) {
  __HT_INTERNAL_LOG_LEVEL = level;
  // TODO: update env?
}

inline bool log_level_enabled(LOG_LEVEL level) {
  return level >= __HT_INTERNAL_LOG_LEVEL;
}

inline std::ostream& operator<<(std::ostream& os, LOG_LEVEL level) {
  switch (level) {
    case LOG_LEVEL::TRACE: os << "TRACE"; break;
    case LOG_LEVEL::DEBUG: os << "DEBUG"; break;
    case LOG_LEVEL::INFO: os << "INFO"; break;
    case LOG_LEVEL::WARN: os << "WARN"; break;
    case LOG_LEVEL::ERROR: os << "ERROR"; break;
    case LOG_LEVEL::FATAL: os << "FATAL"; break;
    default:
      throw std::runtime_error("Unknown logging level: " +
                               std::to_string(static_cast<int>(level)));
  }
  return os;
}

class Logger {
 public:
  Logger(const char* file, int line, LOG_LEVEL level,
         std::ostream& os = std::cout)
  : _os(os), _enabled{log_level_enabled(level)} {
    if (_enabled) {
      auto now = std::chrono::system_clock::now();
      auto time = std::chrono::system_clock::to_time_t(now);
      _ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X") << " ("
          << file << ":" << line << ")] [" << level << "] ";
    }
  }
  ~Logger() {
    if (_enabled) {
      _ss << "\n";
      _os << _ss.str();
    }
  }
  inline std::ostringstream& stream() {
    return _ss;
  }

 protected:
  std::ostringstream _ss;
  std::ostream& _os;
  bool _enabled;

 private:
  Logger(const Logger&);
  void operator=(const Logger&);
};

template <typename Exception = std::runtime_error>
class FatalLogger {
 public:
  FatalLogger(const char* file, int line, bool verbose,
              std::ostream& os = std::cerr)
  : _os(os), _verbose(verbose), _file(file), _line(line) {
    static_assert(std::is_base_of<std::exception, Exception>::value,
                  "Exception must be derived from hetu_exception");
    if (_verbose) {
      auto now = std::chrono::system_clock::now();
      auto time = std::chrono::system_clock::to_time_t(now);
      _verbose_ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X")
                  << " (" << file << ":" << line << ")] [" << LOG_LEVEL::FATAL
                  << "] ";
    }
  }
  ~FatalLogger() noexcept(false) {
    auto err_msg = _ss.str();
    if (_verbose) {
      _verbose_ss << err_msg << "\n";
      _os << _verbose_ss.str();
      _os.flush();
    }
    std::ostringstream exception_ss;
    exception_ss << " (" << _file << ":" << _line << ") " << err_msg;
    throw Exception(exception_ss.str());
  }
  inline std::ostringstream& stream() {
    return _ss;
  }

 protected:
  std::ostringstream _ss;
  std::ostream& _os;
  bool _verbose;
  std::ostringstream _verbose_ss;
  const char* const _file;
  const int _line;

 private:
  FatalLogger(const FatalLogger&);
  void operator=(const FatalLogger&);
};

} // namespace logging
} // namespace hetu

/******************************************************
 * Logging Utils
 ******************************************************/

#define __FILENAME__                                                           \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define quote(x) #x

#define __HT_LOG_TRACE                                                         \
  hetu::logging::Logger(__FILENAME__, __LINE__,                                \
                        hetu::logging::LOG_LEVEL::TRACE, std::cout)
#define __HT_LOG_DEBUG                                                         \
  hetu::logging::Logger(__FILENAME__, __LINE__,                                \
                        hetu::logging::LOG_LEVEL::DEBUG, std::cout)
#define __HT_LOG_INFO                                                          \
  hetu::logging::Logger(__FILENAME__, __LINE__,                                \
                        hetu::logging::LOG_LEVEL::INFO, std::cout)
#define __HT_LOG_WARN                                                          \
  hetu::logging::Logger(__FILENAME__, __LINE__,                                \
                        hetu::logging::LOG_LEVEL::WARN, std::cout)
#define __HT_LOG_ERROR                                                         \
  hetu::logging::Logger(__FILENAME__, __LINE__,                                \
                        hetu::logging::LOG_LEVEL::ERROR, std::cerr)

#define HT_LOG(severity)                                                       \
  if (log_level_enabled(hetu::logging::LOG_LEVEL::severity))                   \
  __HT_LOG_##severity.stream()
#define HT_LOG_IF(severity, cond)                                              \
  if (log_level_enabled(hetu::logging::LOG_LEVEL::severity) && (cond))         \
  __HT_LOG_##severity.stream()
#define HT_LOG_TRACE HT_LOG(TRACE)
#define HT_LOG_TRACE_IF(cond) HT_LOG_IF(TRACE, cond)
#define HT_LOG_DEBUG HT_LOG(DEBUG)
#define HT_LOG_DEBUG_IF(cond) HT_LOG_IF(DEBUG, cond)
#define HT_LOG_INFO HT_LOG(INFO)
#define HT_LOG_INFO_IF(cond) HT_LOG_IF(INFO, cond)
#define HT_LOG_WARN HT_LOG(WARN)
#define HT_LOG_WARN_IF(cond) HT_LOG_IF(WARN, cond)
#define HT_LOG_ERROR HT_LOG(ERROR)
#define HT_LOG_ERROR_IF(cond) HT_LOG_IF(ERROR, cond)

// Calling the following macros will throw a corresponding exception.
// Using the verbose variant will log the error message to std::cerr
// before throwing the exception.
// However, it is recommended to call the macros inside `except.h`
// to throw specific exceptions.
#define __HT_FATAL_VERBOSE(Exception)                                          \
  hetu::logging::FatalLogger<Exception>(__FILENAME__, __LINE__, true).stream()
#define HT_FATAL_VERBOSE __HT_FATAL_VERBOSE(hetu::hetu_exception)
#define __HT_FATAL_SILENT(Exception)                                           \
  hetu::logging::FatalLogger<Exception>(__FILENAME__, __LINE__, false).stream()
#define HT_FATAL_SILENT __HT_FATAL_SILENT(hetu::hetu_exception)
