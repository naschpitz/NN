#include "NN-CLI_ModelPackage.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- isPackage --//
  //===================================================================================================================//

  bool ModelPackage::isPackage(const std::string& path)
  {
    // Accept .nnmodel (legacy) or .nnmodel.tar extensions
    const std::string ext1 = ".nnmodel";
    const std::string ext2 = ".nnmodel.tar";

    if (path.size() < ext1.size()) {
      return false;
    }

    std::string suffix = path.substr(path.size() - ext1.size());
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (suffix == ext1) {
      return true;
    }

    if (path.size() < ext2.size()) {
      return false;
    }

    suffix = path.substr(path.size() - ext2.size());
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return suffix == ext2;
  }

  //===================================================================================================================//
  //-- initUstarHeader --//
  //===================================================================================================================//

  void ModelPackage::initUstarHeader(UstarHeader& hdr, const std::string& fileName, size_t fileSize)
  {
    std::memset(&hdr, 0, sizeof(hdr));

    // name: filename, null-terminated (max 99 chars + NUL)
    std::strncpy(hdr.name, fileName.c_str(), sizeof(hdr.name) - 1);

    // mode: "0000644\0" (7 octal digits + NUL)
    std::snprintf(hdr.mode, sizeof(hdr.mode), "%07o", 0644);

    // uid: "0000000\0"
    std::snprintf(hdr.uid, sizeof(hdr.uid), "%07o", 0);

    // gid: "0000000\0"
    std::snprintf(hdr.gid, sizeof(hdr.gid), "%07o", 0);

    // size: 11 octal digits + NUL
    std::snprintf(hdr.size, sizeof(hdr.size), "%011o", static_cast<unsigned int>(fileSize));

    // mtime: 11 octal digits + NUL
    std::snprintf(hdr.mtime, sizeof(hdr.mtime), "%011o", 0);

    // chksum: 8 spaces initially (will be filled after calculation)
    std::memset(hdr.chksum, ' ', sizeof(hdr.chksum));

    // typeflag: '0' for regular file
    hdr.typeflag = '0';

    // magic: "ustar\0"
    std::memcpy(hdr.magic, "ustar", 6);

    // version: "00"
    std::memcpy(hdr.version, "00", 2);

    // Calculate and set checksum
    unsigned int chksum = calculateChecksum(reinterpret_cast<const char*>(&hdr));
    std::snprintf(hdr.chksum, sizeof(hdr.chksum), "%06o", chksum);
    hdr.chksum[6] = '\0';
    hdr.chksum[7] = ' ';
  }

  //===================================================================================================================//
  //-- calculateChecksum --//
  //===================================================================================================================//

  unsigned int ModelPackage::calculateChecksum(const char* header)
  {
    unsigned int sum = 0;

    for (int i = 0; i < 512; i++) {
      if (i >= 148 && i < 156) {
        // Treat chksum field (bytes 148-155) as spaces
        sum += static_cast<unsigned char>(' ');
      } else {
        sum += static_cast<unsigned char>(header[i]);
      }
    }

    return sum;
  }

  //===================================================================================================================//
  //-- createFromMemory --//
  //===================================================================================================================//

  void ModelPackage::createFromMemory(const std::string& packagePath,
                                       const std::string& jsonStr,
                                       const std::vector<char>& binData)
  {
    // Write to a temporary file first, then atomically rename to final path.
    // This prevents partial writes from corrupting an existing model file.
    const std::string tmpPath = packagePath + ".tmp";

    std::ofstream ofs(tmpPath, std::ios::binary);

    if (!ofs) {
      throw std::runtime_error("Failed to create package file: " + tmpPath);
    }

    //-- Write model.json entry --//

    UstarHeader jsonHdr;
    initUstarHeader(jsonHdr, "model.json", jsonStr.size());
    ofs.write(reinterpret_cast<const char*>(&jsonHdr), sizeof(UstarHeader));

    ofs.write(jsonStr.data(), static_cast<std::streamsize>(jsonStr.size()));

    // Pad to 512-byte boundary
    size_t jsonPadding = (512 - (jsonStr.size() % 512)) % 512;

    if (jsonPadding > 0) {
      std::vector<char> jsonPadBuf(jsonPadding, '\0');
      ofs.write(jsonPadBuf.data(), static_cast<std::streamsize>(jsonPadding));
    }

    //-- Write params.bin entry --//

    UstarHeader binHdr;
    initUstarHeader(binHdr, "params.bin", binData.size());
    ofs.write(reinterpret_cast<const char*>(&binHdr), sizeof(UstarHeader));

    ofs.write(binData.data(), static_cast<std::streamsize>(binData.size()));

    // Pad to 512-byte boundary
    size_t binPadding = (512 - (binData.size() % 512)) % 512;

    if (binPadding > 0) {
      std::vector<char> binPadBuf(binPadding, '\0');
      ofs.write(binPadBuf.data(), static_cast<std::streamsize>(binPadding));
    }

    //-- End-of-archive marker: two 512-byte zero blocks --//

    std::vector<char> zeroBlock(512, '\0');
    ofs.write(zeroBlock.data(), static_cast<std::streamsize>(zeroBlock.size()));
    ofs.write(zeroBlock.data(), static_cast<std::streamsize>(zeroBlock.size()));

    ofs.close();

    // Atomically replace the target file with the completed temp file
    if (std::rename(tmpPath.c_str(), packagePath.c_str()) != 0) {
      std::remove(tmpPath.c_str());
      throw std::runtime_error("Failed to rename temporary package file to: " + packagePath);
    }
  }

  //===================================================================================================================//
  //-- readJsonFromPackage --//
  //===================================================================================================================//

  std::string ModelPackage::readJsonFromPackage(const std::string& packagePath)
  {
    std::vector<char> data = readFileFromTarFile(packagePath, "model.json");
    return std::string(data.begin(), data.end());
  }

  //===================================================================================================================//
  //-- readBinaryFromPackage --//
  //===================================================================================================================//

  std::vector<char> ModelPackage::readBinaryFromPackage(const std::string& packagePath)
  {
    return readFileFromTarFile(packagePath, "params.bin");
  }

  //===================================================================================================================//
  //-- readFileFromTar --//
  //===================================================================================================================//

  std::vector<char> ModelPackage::readFileFromTar(const std::vector<char>& tarData,
                                                   const std::string& fileName)
  {
    const size_t blockSize = 512;
    size_t pos = 0;

    while (pos + blockSize <= tarData.size()) {
      const char* headerPtr = tarData.data() + pos;

      // Check for empty block (end-of-archive)
      if (headerPtr[0] == '\0') {
        break;
      }

      // Validate USTAR magic at offset 257
      if (std::strncmp(headerPtr + 257, "ustar", 5) != 0) {
        break;
      }

      // Parse filename from header
      std::string entryName(headerPtr, strnlen(headerPtr, 100));

      // Parse size from header (octal at offset 124, 12 bytes)
      unsigned int entrySize = 0;
      std::string sizeStr(headerPtr + 124, 11);
      std::istringstream sizeStream(sizeStr);
      sizeStream >> std::oct >> entrySize;

      pos += blockSize;

      // Check if this is the file we want
      if (entryName == fileName) {
        std::vector<char> result(tarData.begin() + pos,
                                 tarData.begin() + pos + entrySize);
        return result;
      }

      // Skip data (round up to 512-byte boundary)
      size_t dataBlocks = (entrySize + blockSize - 1) / blockSize;
      pos += dataBlocks * blockSize;
    }

    // File not found — return empty vector
    return std::vector<char>();
  }

  //===================================================================================================================//
  //-- readFileFromTarFile --//
  //===================================================================================================================//

  std::vector<char> ModelPackage::readFileFromTarFile(const std::string& tarPath,
                                                       const std::string& fileName)
  {
    std::ifstream ifs(tarPath, std::ios::binary | std::ios::ate);

    if (!ifs) {
      throw std::runtime_error("Failed to open package file: " + tarPath);
    }

    std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> tarData(static_cast<size_t>(fileSize));

    if (!ifs.read(tarData.data(), fileSize)) {
      throw std::runtime_error("Failed to read package file: " + tarPath);
    }

    ifs.close();

    return readFileFromTar(tarData, fileName);
  }

} // namespace NN_CLI
