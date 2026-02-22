#ifndef NN_CLI_DATATYPE_HPP
#define NN_CLI_DATATYPE_HPP

#include <string>

namespace NN_CLI {

// Whether input/output data is a flat numeric vector or an image file path
enum class DataType { VECTOR, IMAGE };

// Conversion helpers
DataType dataTypeFromString(const std::string& name);
std::string dataTypeToString(DataType type);

} // namespace NN_CLI

#endif // NN_CLI_DATATYPE_HPP

