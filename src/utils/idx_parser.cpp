#include <cassert>
#include <cstdint>

#include "./include/idx_parser.h"

uint32_t swap_endian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
  return (val << 16) | (val >> 16);
}

IdxParser::IdxParser(const std::string &file_path) {
  position = 0;
  file.open(file_path, std::ios::binary);
  assert(file.is_open());
  assert(read<uint16_t>() == 0);
  assert(read<uint8_t>() == 8);
  int dims_length = read<uint8_t>();
  length = swap_endian(read<uint32_t>());
  dims.push_back(length);
  for (int dim = 1; dim < dims_length; dim++) {
    dims.push_back(swap_endian(read<uint32_t>()));
    element_size *= dims[dim];
  }
  position = 0;
};
IdxParser::~IdxParser() { file.close(); }

template <typename T> T IdxParser::read() {
  assert(position < length);
  T res;
  file.read(reinterpret_cast<char *>(&res), sizeof(res));
  position += (sizeof(res) / element_size);
  return res;
}

void IdxParser::read(std::vector<float> &v, int count) {
  assert(position < length);
  v.resize(count);
  std::vector<uint8_t> v_tmp(count);
  file.read(reinterpret_cast<char *>(v_tmp.data()), count * sizeof(uint8_t));
  for (int i = 0; i < count; i++)
    v[i] = (v_tmp[i]) / 255.0f;
  position += (count * sizeof(uint8_t)) / element_size;
};

void IdxParser::read(std::vector<uint8_t> &v, int count) {
  assert(position < length);
  v.resize(count);
  file.read(reinterpret_cast<char *>(v.data()), count * sizeof(uint8_t));
  position += (count * sizeof(uint8_t)) / element_size;
}
