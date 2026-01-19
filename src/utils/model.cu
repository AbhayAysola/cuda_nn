#include "./include/model.h"
#include <iostream>

__global__ void forward_kernel(Layer layer, float *input, float *output) {
  int tx = threadIdx.x;
  float sum = layer.biases_d[tx];
  for (int i = 0; i < layer.input_size; i++) {
    sum += layer.weights_d[tx * layer.input_size + i] * input[i];
  }
  if (sum < 0)
    output[tx] = 0;
  else
    output[tx] = sum;
}

NeuralNetwork::NeuralNetwork(std::vector<std::pair<int, int>> l) {
  max_dim = 0;
  for (auto x : l) {
    max_dim = std::max(max_dim, std::max(x.first, x.second));
    Layer layer;
    layer.input_size = x.second;
    layer.output_size = x.first;
    cudaMalloc((void **)&layer.weights_d,
               layer.input_size * layer.output_size * sizeof(float));
    cudaMalloc((void **)&layer.biases_d, layer.output_size * sizeof(float));
    layers.push_back(layer);
  }
  cudaMalloc((void **)&input_d[0], max_dim * sizeof(float));
  cudaMalloc((void **)&input_d[1], max_dim * sizeof(float));
};

void NeuralNetwork::load_params(ModelParser *parser) {
  for (int i = 0; i < layers.size(); i++) {
    Layer layer = layers[i];
    std::vector<float> params = parser->read(i);
    cudaMemcpy(layer.weights_d, params.data(),
               layer.input_size * layer.output_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(layer.biases_d,
               &params.data()[layer.input_size * layer.output_size],
               layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
  }
}

std::vector<float> NeuralNetwork::forward(std::vector<float> input) {
  cudaMemcpy(input_d[0], input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  for (int i = 0; i < layers.size(); i++) {
    float *in_d = input_d[i % 2];
    float *out_d = input_d[(i + 1) % 2];
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(layers[i].output_size);
    forward_kernel<<<grid_dim, block_dim>>>(layers[i], in_d, out_d);
    cudaDeviceSynchronize();
  }
  std::vector<float> out(layers.back().output_size);
  cudaMemcpy(out.data(), input_d[layers.size() % 2], out.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return out;
}

NeuralNetwork::~NeuralNetwork() {
  cudaFree(input_d[0]);
  cudaFree(input_d[1]);
}
