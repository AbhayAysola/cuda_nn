#include "./utils/include/idx_parser.h"
#include "./utils/include/model.h"
#include "./utils/include/model_parser.h"
#include <algorithm>
#include <cmath>
#include <iostream>

float cross_entropy_loss(std::vector<float> &pred, int k, int t_val) {
  float offset = 1e-16;
  float max_logit = -1e9;

  // we use the max_logit for numerical stability
  // or else the e^logit values can shoot up till infinity
  for (int i = k * 10; i < (k + 1) * 10; i++)
    max_logit = std::max(max_logit, pred[i]);

  float sum_exp = 0;
  for (int i = k * 10; i < (k + 1) * 10; i++)
    sum_exp += std::exp(pred[i] - max_logit);

  float log_prob = (pred[10 * k + t_val] - max_logit) - std::log(sum_exp);
  return -log_prob;
}

void batched_test(IdxParser *images_parser, IdxParser *labels_parser,
                  NeuralNetwork *network, int batch_size) {

  std::vector<float> images(images_parser->element_size * batch_size);
  std::vector<uint8_t> labels(labels_parser->element_size * batch_size);

  images_parser->read(images, images.size());
  labels_parser->read(labels, labels.size());
  auto pred = network->gradient_descent(images, batch_size);

  float loss = 0;
  int num_correct = 0;
  for (int k = 0; k < batch_size; k++) {
    // for (int i = 0; i < 28; i++) {
    //   for (int j = 0; j < 28; j++) {
    //     if (images[k * images_parser->element_size + i * 28 + j])
    //       std::cout << 1;
    //     else
    //       std::cout << 0;
    //   }
    //   std::cout << std::endl;
    // }

    float m = 0;
    int idx = 0;
    for (int i = 0; i < 10; i++) {
      if (m < pred[i + 10 * k]) {
        m = pred[i + 10 * k];
        idx = i;
      }
    }
    if (idx == labels[k])
      num_correct++;
    loss += cross_entropy_loss(pred, k, labels[k]);
  }
  loss /= batch_size;

  std::cout << "Batch Size: " << batch_size << std::endl;
  std::cout << "Loss: " << loss << std::endl;
  std::cout << "Accuracy: "
            << (float)num_correct / (float)batch_size * float(100) << std::endl;
}

int main() {
  const std::string CACHE_DIR = "./.cache";
  const std::string DATA_DIR = "./data";
  IdxParser images_parser(DATA_DIR + "/train-images.idx3-ubyte");
  IdxParser labels_parser(DATA_DIR + "/train-labels.idx1-ubyte");
  ModelParser model_parser(CACHE_DIR + "/params.bin");

  NeuralNetwork network(model_parser.dims);
  network.load_params(&model_parser);

  int batch_size = 64;
  batched_test(&images_parser, &labels_parser, &network, batch_size);
}
