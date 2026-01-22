#include "./utils/include/idx_parser.h"
#include "./utils/include/model.h"
#include "./utils/include/model_parser.h"
#include <iostream>

int main() {
  const std::string CACHE_DIR = "./.cache";
  const std::string DATA_DIR = "./data";
  IdxParser images_parser(DATA_DIR + "/t10k-images.idx3-ubyte");
  IdxParser labels_parser(DATA_DIR + "/t10k-labels.idx1-ubyte");
  ModelParser model_parser(CACHE_DIR + "/params_cpp.bin");

  NeuralNetwork network(model_parser.dims);
  network.load_params(&model_parser);

  int batch_size = 64;
  std::vector<float> images(images_parser.element_size * batch_size);
  std::vector<uint8_t> labels(labels_parser.element_size * batch_size);

  images_parser.read(images, images.size());
  labels_parser.read(labels, labels.size());
  auto res = network.batched_forward(images, batch_size);
  int num_correct = 0;
  for (int k = 0; k < batch_size; k++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        if (images[k * images_parser.element_size + i * 28 + j])
          std::cout << 1;
        else
          std::cout << 0;
      }
      std::cout << std::endl;
    }

    float m = 0;
    int idx = 0;
    for (int i = 0; i < 10; i++) {
      if (m < res[i + 10 * k]) {
        m = res[i + 10 * k];
        idx = i;
      }
    }
    if (labels[k] == idx)
      num_correct++;
    std:: cout << idx << " " << (int)labels[k] << std::endl;
    std::cout << std::endl;
  }
  std::cout << "Accuracy: " << 100*(float)num_correct / (float)batch_size << std::endl;
}
