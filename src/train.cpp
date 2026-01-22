#include "./utils/include/idx_parser.h"
#include "./utils/include/model.h"
#include "./utils/include/model_parser.h"
#include <iostream>
#include <utility>

int main() {
  const std::string CACHE_DIR = "./.cache";
  const std::string DATA_DIR = "./data";
  IdxParser images_parser(DATA_DIR + "/train-images.idx3-ubyte");
  IdxParser labels_parser(DATA_DIR + "/train-labels.idx1-ubyte");
  ModelParser model_parser(CACHE_DIR + "/params_cpp.bin");

  NeuralNetwork network(model_parser.dims);
  // for (auto dim : model_parser.dims)
  //   std::cout << dim.first << '\n';
  network.load_params(&model_parser);

  // std::vector<std::pair<int, int>> dims = {std::make_pair(512, 28 * 28), std::make_pair(512, 512), std::make_pair(10, 512)};
  // NeuralNetwork network(dims);
  // network.initialize_params();

  int batch_size = 64;
  int num_batches = 10;
  int num_epochs = 50;
  // std::vector<float> images(images_parser.element_size * batch_size * num_batches);
  // std::vector<uint8_t> labels(labels_parser.element_size * batch_size * num_batches);

  std::vector<float> images(images_parser.element_size * images_parser.length);
  std::vector<uint8_t> labels(labels_parser.element_size * labels_parser.length);
  images_parser.read(images, images.size());
  labels_parser.read(labels, labels.size());

  network.gradient_descent(images, labels, batch_size, num_epochs);
  network.save_params(CACHE_DIR + "/params_cpp.bin");
}
