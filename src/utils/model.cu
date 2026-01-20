#include "./include/model.h"
#include <cassert>
#include <iostream>
#include <ostream>

__device__ __host__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// forward for one input (28*28, 1)
__global__ void forward_kernel(Layer layer, float *input, float *output,
                               bool relu) {
  int tx = threadIdx.x;
  float sum = layer.biases_d[tx];
  for (int i = 0; i < layer.input_size; i++) {
    sum += layer.weights_d[tx * layer.input_size + i] * input[i];
  }
  if (sum < 0 and relu)
    output[tx] = 0;
  else
    output[tx] = sum;
}

// input shape (batch_size, layer.input_size)
// weights shape (layer.output_size, layer.input_size)
// output_shape (batch_size, layer.output_size)
__global__ void batched_forward_kernel(Layer layer, int batch_size,
                                       float *input, float *output, bool relu) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  if (row < layer.output_size and col < batch_size) {
    float sum = layer.biases_d[row];
    // input[i][col]
    for (int i = 0; i < layer.input_size; i++) {
      sum += layer.weights_d[row * layer.input_size + i] *
             input[col * layer.input_size + i];
    }
    if (sum < 0 and relu)
      output[col * layer.output_size + row] = 0;
    else
      output[col * layer.output_size + row] = sum;
  }
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

std::vector<float> NeuralNetwork::forward(std::vector<float> &input) {
  cudaMemcpy(input_d[0], input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  for (int i = 0; i < layers.size(); i++) {
    float *in_d = input_d[i % 2];
    float *out_d = input_d[(i + 1) % 2];
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(layers[i].output_size);
    forward_kernel<<<grid_dim, block_dim>>>(
        layers[i], in_d, out_d, (i == layers.size() - 1) ? false : true);
    cudaDeviceSynchronize();
  }
  std::vector<float> out(layers.back().output_size);
  cudaMemcpy(out.data(), input_d[layers.size() % 2], out.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return out;
}

const int BLOCK_SIZE = 16;
std::vector<float> NeuralNetwork::batched_forward(std::vector<float> &input,
                                                  int batch_size) {
  float *batched_input_d[2];
  cudaMalloc((void **)&batched_input_d[0],
             max_dim * batch_size * sizeof(float));
  cudaMalloc((void **)&batched_input_d[1],
             max_dim * batch_size * sizeof(float));

  cudaMemcpy(batched_input_d[0], input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  for (int i = 0; i < layers.size(); i++) {
    float *in_d = batched_input_d[i % 2];
    float *out_d = batched_input_d[(i + 1) % 2];
    dim3 grid_dim(cdiv(batch_size, BLOCK_SIZE),
                  cdiv(layers[i].output_size, BLOCK_SIZE), 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    batched_forward_kernel<<<grid_dim, block_dim>>>(
        layers[i], batch_size, in_d, out_d,
        (i == layers.size() - 1) ? false : true);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cout << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
  }
  std::vector<float> out(layers.back().output_size * batch_size);
  cudaMemcpy(out.data(), batched_input_d[layers.size() % 2],
             out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(batched_input_d[0]);
  cudaFree(batched_input_d[1]);
  return out;
}

std::vector<float> NeuralNetwork::gradient_descent(std::vector<float> &input,
                                                   int batch_size) {
  float *batched_input_d;
  cudaMalloc((void **)&batched_input_d,
             layers[0].input_size * batch_size * sizeof(float));
  std::vector<float> out(layers.back().output_size * batch_size);

  for (Layer &layer : layers) {
    cudaMalloc((void **)&layer.activations_d,
               layer.output_size * batch_size * sizeof(float));
  }

  for (int k = 0; k < input.size(); k += batch_size * layers[0].input_size) {
    cudaMemcpy(batched_input_d, input.data() + k,
               batch_size * layers[0].input_size * sizeof(float),
               cudaMemcpyHostToDevice);

    for (int layer = 0; layer < layers.size(); layer++) {
      dim3 grid_dim(cdiv(batch_size, BLOCK_SIZE),
                    cdiv(layers[layer].output_size, BLOCK_SIZE), 1);
      dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
      batched_forward_kernel<<<grid_dim, block_dim>>>(
          layers[layer], batch_size,
          (layer == 0 ? batched_input_d : layers[layer - 1].activations_d),
          layers[layer].activations_d,
          (layer == layers.size() - 1) ? false : true);
      cudaDeviceSynchronize();
    }
    cudaMemcpy(out.data(), layers[layers.size() - 1].activations_d,
               out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(batched_input_d);
  for (Layer layer : layers)
    cudaFree(layer.activations_d);

  return out;
}

NeuralNetwork::~NeuralNetwork() {
  for (Layer layer : layers) {
    cudaFree(layer.biases_d);
    cudaFree(layer.weights_d);
  }
  cudaFree(input_d[0]);
  cudaFree(input_d[1]);
}
