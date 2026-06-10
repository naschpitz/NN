#ifndef NN_CLI_MODELPACKAGE_HPP
#define NN_CLI_MODELPACKAGE_HPP

#include <string>
#include <vector>

namespace NN_CLI
{

  class ModelPackage
  {
    public:
      // Check if filename ends with .nnmodel or .nnmodel.tar (legacy format)
      static bool isPackage(const std::string& path);

      // Create a .nnmodel.tar package from in-memory buffers
      static void createFromMemory(const std::string& packagePath, const std::string& jsonStr,
                                   const std::vector<char>& binData);

      // Read model.json content from package (returns JSON string)
      static std::string readJsonFromPackage(const std::string& packagePath);

      // Read params.bin content from package (returns raw binary)
      static std::vector<char> readBinaryFromPackage(const std::string& packagePath);

      // Extract file by name from a tar archive into a buffer
      static std::vector<char> readFileFromTar(const std::vector<char>& tarData, const std::string& fileName);
      static std::vector<char> readFileFromTarFile(const std::string& tarPath, const std::string& fileName);

    private:
      struct UstarHeader {
          char name[100];
          char mode[8];
          char uid[8];
          char gid[8];
          char size[12];
          char mtime[12];
          char chksum[8];
          char typeflag;
          char linkname[100];
          char magic[6];
          char version[2];
          char uname[32];
          char gname[32];
          char devmajor[8];
          char devminor[8];
          char prefix[155];
          char padding[12];
      };
      static_assert(sizeof(UstarHeader) == 512, "UstarHeader must be 512 bytes");

      static void initUstarHeader(UstarHeader& hdr, const std::string& fileName, size_t fileSize);
      static unsigned int calculateChecksum(const char* header);
  };

} // namespace NN_CLI

#endif // NN_CLI_MODELPACKAGE_HPP
