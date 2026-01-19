#include "./utils/include/idx_parser.h"
#include "./utils/include/model.h"
#include "./utils/include/model_parser.h"
#include <iostream>

int main() {
  const std::string CACHE_DIR = "./.cache";
  const std::string DATA_DIR = "./data";
  IdxParser data_parser(DATA_DIR + "/t10k-images.idx3-ubyte");
  ModelParser model_parser(CACHE_DIR + "/params.bin");

  NeuralNetwork network(model_parser.dims);
  network.load_params(&model_parser);

  std::vector<float> v(data_parser.element_size);

  for (int i = 0; i < 100; i++) {

    data_parser.read(v, v.size());
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        if (v[i * 28 + j])
          std::cout << 1;
        else
          std::cout << 0;
      }
      std::cout << std::endl;
    }

    auto res = network.forward(v);
    float m = 0;
    int idx = 0;
    for (int i = 0; i < 10; i++) {
      std::cout << res[i] << ' ';
      if (m < res[i]) {
        m = res[i];
        idx = i;
      }
    }
    std::cout << std::endl;
    std::cout << idx << std::endl;
  }
}
