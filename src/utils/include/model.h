#ifndef __MODEL_H__
#define __MODEL_H__
#include "./model_parser.h"
#include <vector>

struct Layer {
  int input_size;
  int output_size;
  float *weights_d;
  float *biases_d;
  float *activations_d;
};

class NeuralNetwork {
private:
  std::vector<Layer> layers;
  int max_dim;
  float *input_d[2];

public:
  NeuralNetwork(std::vector<std::pair<int, int>> l);

  void load_params(ModelParser *parser);

  std::vector<float> forward(std::vector<float> &input);

  std::vector<float> batched_forward(std::vector<float> &input, int batch_size);

  std::vector<float> gradient_descent(std::vector<float> &input,
                                      int batch_size);


  ~NeuralNetwork();
};
#endif
