#ifndef __MODEL_PARSER_H__
#define __MODEL_PARSER_H__

#include <cassert>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

class ModelParser {
public:
  long long length = 1;
  long long position;
  std::vector<std::pair<int, int>> dims;
  ModelParser(const std::string &file_path);
  ~ModelParser();

  template <typename T> T read();
  std::vector<float> read(int layer);

private:
  std::ifstream file;
};
#endif
