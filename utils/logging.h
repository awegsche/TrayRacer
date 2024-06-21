#ifndef LOGGING_H
#define LOGGING_H

#ifdef NO_SPDLOG
#include <format>
#include <string>
#include <vector>
#else
#include <spdlog/spdlog.h>
#endif


namespace logging {
#ifdef NO_SPDLOG
enum class LogMsgType { Error, Warning, Info, Debug };

struct LogMsg
{
    LogMsgType type = LogMsgType::Info;
    std::string msg;
};

extern std::vector<LogMsg> log_msg;

template<typename... Args> void logv(LogMsgType type, const char *fmt, Args... args)
{
    std::string full_message = std::vformat(fmt, std::make_format_args(args...));
    auto it = full_message.cbegin();

    while (true) {
        auto end = std::find(it, full_message.cend(), '\n');
        log_msg.push_back({ type, std::string(it, end) });
        if (end == full_message.cend()) break;
        it = end + 1;
    }
}

template<typename... Args> void info(const char *fmt, Args... args) { logv(LogMsgType::Info, fmt, args...); }
template<typename... Args> void error(const char *fmt, Args... args) { logv(LogMsgType::Error, fmt, args...); }

#else

template<typename... Args> void info(spdlog::format_string_t<Args...> fmt, Args &&...args)
{
    spdlog::info(fmt, std::forward<Args>(args)...);
}

template<typename... Args> void error(spdlog::format_string_t<Args...> fmt, Args &&...args)
{
    spdlog::error(fmt, std::forward<Args>(args)...);
}

template<typename... Args> void warn(spdlog::format_string_t<Args...> fmt, Args &&...args)
{
    spdlog::warn(fmt, std::forward<Args>(args)...);
}

#endif

}// namespace logging

#endif
