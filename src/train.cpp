#include "./utils/include/idx_parser.h"
#include "./utils/include/model.h"
#include "./utils/include/model_parser.h"

int main() {
  const std::string CACHE_DIR = "./.cache";
  const std::string DATA_DIR = "./data";
  IdxParser images_parser(DATA_DIR + "/train-images.idx3-ubyte");
  IdxParser labels_parser(DATA_DIR + "/train-labels.idx1-ubyte");
  ModelParser model_parser(CACHE_DIR + "/params.bin");

  NeuralNetwork network(model_parser.dims);
  // network.load_params(&model_parser);
  network.initialize_params();

  int batch_size = 32;
  int num_batches = 10;
  int num_epochs = 50;
  // std::vector<float> images(images_parser.element_size * batch_size * num_batches);
  // std::vector<uint8_t> labels(labels_parser.element_size * batch_size * num_batches);

  std::vector<float> images(images_parser.element_size * images_parser.length);
  std::vector<uint8_t> labels(labels_parser.element_size * labels_parser.length);
  images_parser.read(images, images.size());
  labels_parser.read(labels, labels.size());

  network.gradient_descent(images, labels, batch_size, num_epochs);
}
