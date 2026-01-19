#include <cstdint>
#ifndef __IDX_PARSER_H__

#include <fstream>
#include <string>
#include <vector>
#define __IDX_PARSER_H__
class IdxParser {
public:
  long long element_size = 1;
  long long length = 24;
  long long position;
  std::vector<int> dims;
  IdxParser(const std::string &file_path);
  ~IdxParser();

  template <typename T> T read();

  void read(std::vector<float> &v, int count);

private:
  std::ifstream file;
};
#endif
