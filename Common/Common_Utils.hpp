#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

//===================================================================================================================//

namespace Common
{
  namespace Utils
  {
      //-- Formatting --//

      // Format current time as ISO 8601 string (e.g., "2026-06-11T14:30:00+02:00")
      inline std::string formatISO8601()
      {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm* localTime = std::localtime(&time);

        std::ostringstream oss;
        oss << std::put_time(localTime, "%Y-%m-%dT%H:%M:%S");

        // Add UTC offset in ISO 8601 format (e.g., +01:00)
        // %z gives +0100, we need to insert the colon
        char tzOffset[8];
        std::strftime(tzOffset, sizeof(tzOffset), "%z", localTime);

        // Insert colon: "+0100" -> "+01:00"
        std::string offset(tzOffset);

        if (offset.length() >= 5) {
          offset.insert(3, ":");
        }

        oss << offset;

        return oss.str();
      }

      //===================================================================================================================//

       // Format current time as a human-readable string using system locale (e.g., "11/06/2026 14:30:00")
      inline std::string formatHumanReadable()
      {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm* localTime = std::localtime(&time);

        std::ostringstream oss;
         oss << std::put_time(localTime, "%x %X");

        return oss.str();
      }

      //===================================================================================================================//

      // Format duration in seconds as human-readable string (e.g., "1y 2mo 3d 4h 5m 6s")
      inline std::string formatDuration(double totalSeconds)
      {
        // Handle negative or zero durations
        if (totalSeconds <= 0) {
          return "0s";
        }

        // Constants for time calculations
        constexpr int SECONDS_PER_MINUTE = 60;
        constexpr int SECONDS_PER_HOUR = 3600;
        constexpr int SECONDS_PER_DAY = 86400;
        constexpr int DAYS_PER_MONTH = 30; // Approximate
        constexpr int DAYS_PER_YEAR = 365; // Approximate

        // Calculate each unit
        long long totalSecs = static_cast<long long>(totalSeconds);

        long long years = totalSecs / (DAYS_PER_YEAR * SECONDS_PER_DAY);
        totalSecs %= (DAYS_PER_YEAR * SECONDS_PER_DAY);

        long long months = totalSecs / (DAYS_PER_MONTH * SECONDS_PER_DAY);
        totalSecs %= (DAYS_PER_MONTH * SECONDS_PER_DAY);

        long long days = totalSecs / SECONDS_PER_DAY;
        totalSecs %= SECONDS_PER_DAY;

        long long hours = totalSecs / SECONDS_PER_HOUR;
        totalSecs %= SECONDS_PER_HOUR;

        long long minutes = totalSecs / SECONDS_PER_MINUTE;
        long long seconds = totalSecs % SECONDS_PER_MINUTE;

        // Build the formatted string, only including non-zero units
        std::ostringstream oss;

        if (years > 0) {
          oss << years << "y ";
        }

        if (months > 0) {
          oss << months << "mo ";
        }

        if (days > 0) {
          oss << days << "d ";
        }

        if (hours > 0) {
          oss << hours << "h ";
        }

        if (minutes > 0) {
          oss << minutes << "m ";
        }

        if (seconds > 0 || oss.str().empty()) {
          oss << seconds << "s";
        }

        std::string result = oss.str();

        // Trim trailing space if present
        if (!result.empty() && result.back() == ' ') {
          result.pop_back();
        }

        return result;
      }
  } // namespace Utils
} // namespace Common

//===================================================================================================================//

#endif // COMMON_UTILS_HPP
