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

  int batch_size = 16;
  std::vector<float> v(data_parser.element_size * batch_size);

  data_parser.read(v, v.size());
  auto res = network.batched_forward(v, batch_size);
  for (int k = 0; k < batch_size; k++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        if (v[k * data_parser.element_size + i * 28 + j])
          std::cout << 1;
        else
          std::cout << 0;
      }
      std::cout << std::endl;
    }

    float m = 0;
    int idx = 0;
    for (int i = 0; i < 10; i++) {
      std::cout << res[i + 10 * k] << ' ';
      if (m < res[i + 10 * k]) {
        m = res[i + 10 * k];
        idx = i;
      }
    }
    std::cout << std::endl;
    std::cout << idx << std::endl;
  }
}
