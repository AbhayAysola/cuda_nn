# CUDA Neural Network
A "from-scratch" implementation of a feedforward neural network built using **C++** and **NVIDIA CUDA**.

The goal of this project is to explore the "atoms" of machine learning—moving beyond high-level abstractions like PyTorch or TensorFlow to understand how gradient descent and backpropagation operate at the hardware level.



## Features
* **Built from Scratch:** No high-level ML libraries; logic is implemented in raw C++ and CUDA kernels.
* **GPU Accelerated:** Leverages CUDA for parallelized training and inference.
* **Full Pipeline:** Includes implementation for both forward passes and backpropagation.

> **Note:** This project is a work in progress. While functional, I am currently working on 1-to-1 parity with a Python reference implementation and further optimizing the GPU kernels for better performance.

---

## Prerequisites

Before building, ensure you have the following installed:
* **CUDA Toolkit:** [Installation Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
* **C++ Compiler:** (e.g., GCC, Clang, or MSVC)
* **CMake:** Version 3.10 or higher
PS: You'll also need an nvidia gpu is that wasn't obvious by now :)

---

## Building the Project

Follow these steps to compile the source code:

```bash
# Configure the build directory
cmake -S . -B build

# Compile the project
cmake --build ./build
```

# Usage
```bash
# training
./build/train
#inference
./build/cuda_nn
```

# Features
* add alternate hardware support - amd, intel, etc.
* flexible network configuration
* alternate activation functions and loss functions
* autograd support
* optimized kernels

# References
[MNIST Dataset](https://github.com/cvdfoundation/mnist)

[Python Reference Implementation](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

[Backpropagation 3b1b](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
