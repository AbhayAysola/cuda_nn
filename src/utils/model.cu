#include "./include/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <ostream>
#include <random>

__device__ __host__ int cdiv(int a, int b) { return (a + b - 1) / b; }

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

  float log_prob = (pred[10 * k + t_val] - max_logit) - std::log(sum_exp + offset);
  return -log_prob;
}

// forward for one input (28*28, 1)
__global__ void forward_kernel(Layer layer, float *input, float *output, bool relu) {
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

// input shape batch_size X layer.input_size
// weights shape layer.output_size X layer.input_size
// output_shape batch_size X layer.output_size
__global__ void batched_forward_kernel(Layer layer, int batch_size, float *input, float *output, bool relu) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  if (row < layer.output_size and col < batch_size) {
    float sum = layer.biases_d[row];
    // input[i][col]
    for (int i = 0; i < layer.input_size; i++) {
      sum += layer.weights_d[row * layer.input_size + i] * input[col * layer.input_size + i];
    }
    if (sum < 0 and relu)
      output[col * layer.output_size + row] = 0;
    else
      output[col * layer.output_size + row] = sum;
  }
}

// delta.shape batch_size X output_size
// activations.shape batch_size X input_size
// weights.shape output_size X input_size
__global__ void param_update_kernel(Layer layer, int batch_size, float *delta, float learning_rate) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  // grad weights = delta^T * layer.activations_d
  // grad biases = delta reduction
  // weights -= learning_rate * grad weights
  // biases -= learning_rate * grad biases
  if (row < layer.output_size and col < layer.input_size) {
    float weights_sum = 0;
    float biases_sum = 0;
    for (int i = 0; i < batch_size; i++) {
      weights_sum += delta[i * layer.output_size + row] * layer.activations_d[i * layer.input_size + col];
      if (col == 0)
        biases_sum += delta[i * layer.output_size + row];
    }
    layer.weights_d[row * layer.input_size + col] -= (learning_rate * weights_sum) / batch_size;
    if (col == 0)
      layer.biases_d[row] -= (learning_rate * biases_sum) / batch_size;
  }
}

// delta.shape batch_size X output_size
// weights.shape output_size X input_size
// delta_new.shape batch_size X input_size
__global__ void delta_update_kernel(Layer layer, int batch_size, float *delta_in, float *delta_out, bool relu) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // delta_new = [delta dot relu'(z)] * weights
  if (row < batch_size and col < layer.input_size) {
    float sum = 0;
    for (int i = 0; i < layer.output_size; i++) {
      sum += delta_in[row * layer.output_size + i] * layer.weights_d[i * layer.input_size + col];
    }
    if (relu and layer.activations_d[row * layer.input_size + col] <= 0) {
      sum = 0;
    }
    delta_out[row * layer.input_size + col] = sum;
  }
}

NeuralNetwork::NeuralNetwork(std::vector<std::pair<int, int>> l) {
  max_dim = 0;
  for (auto x : l) {
    max_dim = std::max(max_dim, std::max(x.first, x.second));
    Layer layer;
    layer.input_size = x.second;
    layer.output_size = x.first;
    cudaMalloc((void **)&layer.weights_d, layer.input_size * layer.output_size * sizeof(float));
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
    cudaMemcpy(layer.weights_d, params.data(), layer.input_size * layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(layer.biases_d, &params.data()[layer.input_size * layer.output_size], layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
  }
}

void NeuralNetwork::initialize_params() {
  for (int i = 0; i < layers.size(); i++) {
    Layer layer = layers[i];
    std::vector<float> weights(layer.input_size * layer.output_size);
    std::vector<float> biases(layer.output_size, 0.0);
    std::default_random_engine generator;
    float std_dev = std::sqrt(2.0f / layer.input_size);
    std::normal_distribution<float> distribution(0.0, std_dev);
    for (int i = 0; i < layer.input_size * layer.output_size; ++i) {
      weights[i] = distribution(generator);
    }
    cudaMemcpy(layer.weights_d, weights.data(), layer.input_size * layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(layer.biases_d, biases.data(), layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
  }
}

std::vector<float> NeuralNetwork::forward(std::vector<float> &input) {
  cudaMemcpy(input_d[0], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
  for (int i = 0; i < layers.size(); i++) {
    float *in_d = input_d[i % 2];
    float *out_d = input_d[(i + 1) % 2];
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(layers[i].output_size);
    forward_kernel<<<grid_dim, block_dim>>>(layers[i], in_d, out_d, (i == layers.size() - 1) ? false : true);
    cudaDeviceSynchronize();
  }
  std::vector<float> out(layers.back().output_size);
  cudaMemcpy(out.data(), input_d[layers.size() % 2], out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return out;
}

const int BLOCK_SIZE = 16;
std::vector<float> NeuralNetwork::batched_forward(std::vector<float> &input, int batch_size) {
  float *batched_input_d[2];
  cudaMalloc((void **)&batched_input_d[0], max_dim * batch_size * sizeof(float));
  cudaMalloc((void **)&batched_input_d[1], max_dim * batch_size * sizeof(float));

  cudaMemcpy(batched_input_d[0], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
  for (int i = 0; i < layers.size(); i++) {
    float *in_d = batched_input_d[i % 2];
    float *out_d = batched_input_d[(i + 1) % 2];
    dim3 grid_dim(cdiv(batch_size, BLOCK_SIZE), cdiv(layers[i].output_size, BLOCK_SIZE), 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    batched_forward_kernel<<<grid_dim, block_dim>>>(layers[i], batch_size, in_d, out_d, (i == layers.size() - 1) ? false : true);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cout << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
  }
  std::vector<float> out(layers.back().output_size * batch_size);
  cudaMemcpy(out.data(), batched_input_d[layers.size() % 2], out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(batched_input_d[0]);
  cudaFree(batched_input_d[1]);
  return out;
}

void NeuralNetwork::gradient_descent(std::vector<float> &input, std::vector<uint8_t> &labels, int batch_size, int num_epochs) {
  float *batched_output_d;
  float *delta_d[2];
  for (int i = 0; i < num_epochs; i++) {
    std::cout << "Epoch Number: " << i + 1 << std::endl;

    int input_size = layers[0].input_size;
    int out_size = layers.back().output_size;
    cudaMalloc((void **)&batched_output_d, out_size * batch_size * sizeof(float));
    std::vector<float> out(out_size * batch_size);

    cudaMalloc((void **)&delta_d[0], max_dim * batch_size * sizeof(float));
    cudaMalloc((void **)&delta_d[1], max_dim * batch_size * sizeof(float));

    for (Layer &layer : layers)
      cudaMalloc((void **)&layer.activations_d, layer.input_size * batch_size * sizeof(float));

    int num_batches = 0;
    float epoch_loss = 0;
    float epoch_accuracy = 0;
    // batch wise computation
    for (int idx = 0; idx < input.size() / input_size; idx += batch_size) {
      num_batches++;
      if (idx + batch_size >= input.size() / input_size)
        batch_size = input.size() / input_size - idx;
      cudaMemcpy(layers[0].activations_d, input.data() + idx * input_size, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

      // forward pass
      for (int layer = 0; layer < layers.size(); layer++) {
        dim3 grid_dim(cdiv(batch_size, BLOCK_SIZE), cdiv(layers[layer].output_size, BLOCK_SIZE), 1);
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
        batched_forward_kernel<<<grid_dim, block_dim>>>(layers[layer], batch_size, layers[layer].activations_d, (layer == layers.size() - 1) ? batched_output_d : layers[layer + 1].activations_d,
                                                        (layer == layers.size() - 1) ? false : true);
        cudaDeviceSynchronize();
      }
      cudaMemcpy(out.data(), batched_output_d, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

      // calculate loss and accuracy
      float loss = 0;
      int num_correct = 0;
      for (int k = 0; k < batch_size; k++) {
        float max_logit = -1e9;
        int pred = 0;
        for (int i = 0; i < out_size; i++) {
          if (max_logit < out[i + k * out_size]) {
            max_logit = out[i + k * out_size];
            pred = i;
          }
        }

        float sum_exp = 0;
        for (int i = 0; i < out_size; i++) {
          out[i + k * out_size] = std::exp(out[i + k * out_size] - max_logit);
          sum_exp += out[i + k * out_size];
        }

        for (int i = 0; i < out_size; i++) {
          out[i + k * out_size] /= sum_exp;

          if (i == labels[idx + k]) {
            out[i + k * out_size] -= 1.0f;
          }
        }
        if (pred == labels[idx + k])
          num_correct++;
        loss += cross_entropy_loss(out, k, labels[idx + k]);
      }

      loss /= batch_size;
      epoch_loss += loss;
      float accuracy = (float)num_correct / (float)batch_size * float(100);
      epoch_accuracy += accuracy;
      float learning_rate = 5 * 1e-4 / (loss - 2);
      // std::cout << "Batch Size: " << batch_size << std::endl;
      // std::cout << "Batch Loss: " << loss << std::endl;
      // std::cout << "Batch Accuracy: " << accuracy << "%" << std::endl;
      // std::cout << std::endl;

      // backward pass
      cudaMemcpy(delta_d[(layers.size() - 1) % 2], out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << ' ' << idx << std::endl;
      }
      for (int i = layers.size() - 1; i >= 0; i--) {
        float *in_delta = delta_d[i % 2];
        float *out_delta = delta_d[(i + 1) % 2];

        dim3 grid_dim_delta(cdiv(batch_size, BLOCK_SIZE), cdiv(layers[i].input_size, BLOCK_SIZE), 1);
        dim3 block_dim_delta(BLOCK_SIZE, BLOCK_SIZE, 1);
        delta_update_kernel<<<grid_dim_delta, block_dim_delta>>>(layers[i], batch_size, in_delta, out_delta, (i == layers.size() - 1) ? false : true);

        dim3 grid_dim_params(cdiv(layers[i].output_size, BLOCK_SIZE), cdiv(layers[i].input_size, BLOCK_SIZE), 1);
        dim3 block_dim_params(BLOCK_SIZE, BLOCK_SIZE, 1);
        param_update_kernel<<<grid_dim_params, block_dim_params>>>(layers[i], batch_size, in_delta, learning_rate);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cout << cudaGetErrorString(err) << ' ' << idx << std::endl;
        }
      }
    }
    std::cout << "Number of Batches: " << num_batches << std::endl;
    std::cout << "Epoch Loss: " << epoch_loss / num_batches << std::endl;
    std::cout << "Epoch Accuracy: " << epoch_accuracy / (float)num_batches << "%" << std::endl;
  }
  cudaFree(batched_output_d);
  cudaFree(delta_d[0]);
  cudaFree(delta_d[1]);
  for (Layer layer : layers)
    cudaFree(layer.activations_d);
}

NeuralNetwork::~NeuralNetwork() {
  for (Layer layer : layers) {
    cudaFree(layer.biases_d);
    cudaFree(layer.weights_d);
  }
  cudaFree(input_d[0]);
  cudaFree(input_d[1]);
}
