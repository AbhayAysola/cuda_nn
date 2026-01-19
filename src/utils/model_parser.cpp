#include "include/model_parser.h"

#include <cstdint>
#include <iostream>

ModelParser::ModelParser(const std::string &file_path) {
  position = 0;
  file.open(file_path, std::ios::binary);
  assert(file.is_open());
  dims.resize(read<uint32_t>());
  for (int i = 0; i < dims.size(); i++) {
    int rows = read<uint16_t>();
    int cols = read<uint16_t>();
    dims[i] = std::make_pair(rows, cols);
    length += rows * (cols + 1) * sizeof(float);
  }
  position = 0;
};
ModelParser::~ModelParser() { file.close(); }

template <typename T> T ModelParser::read() {
  assert(position < length);
  T res;
  file.read(reinterpret_cast<char *>(&res), sizeof(res));
  return res;
}

std::vector<float> ModelParser::read(int layer) {
  assert(position < length);
  int rows = dims[layer].first;
  int cols = dims[layer].second;
  
  std::vector<float> params(rows * (cols + 1));
  file.read(reinterpret_cast<char *>(params.data()),
            rows * (cols + 1) * sizeof(float));


  position += (rows * (cols + 1) * sizeof(float));
  return params;
};
