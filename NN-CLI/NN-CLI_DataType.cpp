#include "NN-CLI_DataType.hpp"

#include <stdexcept>

namespace NN_CLI {

//===================================================================================================================//

DataType dataTypeFromString(const std::string& name) {
    if (name == "vector") return DataType::VECTOR;
    if (name == "image")  return DataType::IMAGE;
    throw std::runtime_error("Unknown data type: '" + name + "'. Expected 'vector' or 'image'.");
}

//===================================================================================================================//

std::string dataTypeToString(DataType type) {
    switch (type) {
        case DataType::VECTOR: return "vector";
        case DataType::IMAGE:  return "image";
    }
    return "vector";
}

//===================================================================================================================//

} // namespace NN_CLI

