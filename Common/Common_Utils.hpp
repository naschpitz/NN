#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <string>

//===================================================================================================================//

namespace Common
{
  namespace Utils
  {
    //-- Formatting --//

    // Format current time as ISO 8601 string (e.g., "2026-06-11T14:30:00+02:00")
    std::string formatISO8601();

    //===================================================================================================================//

    // Format current time as a human-readable string using system locale (e.g., "11/06/2026 14:30:00")
    std::string formatHumanReadable();

    //===================================================================================================================//

    // Format duration in seconds as human-readable string (e.g., "1y 2mo 3d 4h 5m 6s")
    std::string formatDuration(double totalSeconds);
  } // namespace Utils
} // namespace Common

//===================================================================================================================//

#endif // COMMON_UTILS_HPP
