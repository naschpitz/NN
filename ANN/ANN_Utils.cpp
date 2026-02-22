#include "ANN_Utils.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace ANN;

//===================================================================================================================//

template <typename T>
std::string Utils<T>::formatISO8601() {
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

template <typename T>
std::string Utils<T>::formatDuration(double totalSeconds) {
  // Handle negative or zero durations
  if (totalSeconds <= 0) {
    return "0s";
  }

  // Constants for time calculations
  constexpr int SECONDS_PER_MINUTE = 60;
  constexpr int SECONDS_PER_HOUR = 3600;
  constexpr int SECONDS_PER_DAY = 86400;
  constexpr int DAYS_PER_MONTH = 30;  // Approximate
  constexpr int DAYS_PER_YEAR = 365;  // Approximate

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

//===================================================================================================================//

// Explicit template instantiations.
template class ANN::Utils<int>;
template class ANN::Utils<double>;
template class ANN::Utils<float>;
