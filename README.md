# Godot Neural Network Engine

**GPU-Accelerated Deep Learning Inside Godot Using Compute Shaders**

# Overview

This project implements a fully GPU-accelerated feedforward neural network inside the Godot Engine using SPIR-V compute shaders. It is designed for high-performance training and inference, leveraging Godot’s RenderingDevice API to execute parallel computations directly on the GPU.

The system is modular, scalable ideal for experimentation, technical showcases, or integration into simulation pipelines. It supports mini-batching, gradient normalization, and NaN/exploding gradient detection, with a clean separation between model logic, training flow, and GPU dispatch.

Whether you're building AI-driven simulations, experimenting with shader-based learning, or showcasing GPU compute capabilities in Godot, this project offers a robust and extensible foundation.

# Features

- GPU-Accelerated Training and Inference Leverages SPIR-V compute shaders and Godot’s RenderingDevice API for high-speed parallel computation.
- Modular Architecture Clean separation between model structure, training logic, shader dispatch, and utility functions — ideal for extension and experimentation.
- Mini-Batch Support Processes multiple input vectors in parallel for efficient gradient accumulation and stable training.
- Gradient Normalization and Stability Checks Includes NaN detection, exploding gradient handling, and activation clamping to ensure numerical robustness.
- Shader-Driven Backpropagation Custom GLSL shaders compute weight and bias gradients using atomic-safe accumulation and activation derivatives.
- Test Bench Integration Includes a configurable test node for synthetic data generation, training execution, and performance evaluation.
- Utility Toolkit Provides tensor reshaping, byte/float conversion, batch flattening, and diagnostic helpers via `TensorUtils`.

# Architecture

This project is built around a clean separation of responsibilities between model structure, GPU dispatch, training logic, and utility functions. Each component is designed to be modular, extensible, and performance-conscious

## Core components

- `NeuralNetwork`
  Manages the overall structure of the network and performs forward propagation using GPU shaders. Caches intermediate outputs for use during training.
- `NetworkLayer`
  Encapsulates weights and biases for a single layer. Provides flat and matrix representations for GPU dispatch and gradient updates.
- `Trainer`
  Handles mini-batch training, loss computation, backpropagation, and gradient application. Implements NaN detection and gradient explosion handling.
- `ShaderRunner`
  Manages GPU resources and dispatches SPIR-V compute shaders. Handles buffer creation, uniform binding, pipeline execution, and synchronization.
- `TensorUtils`
  Provides tensor reshaping, byte/float conversion, batch flattening, and diagnostic utilities.

### GLSL Compute Shaders

- `Forward Shader`
  Computes gradients using activation derivatives and error signals. Uses atomic operations for safe parallel accumulation.
- `Backward Shader`
  Computes gradients using activation derivatives and error signals. Uses atomic operations for safe parallel accumulation.

# Benchmarks

The benchmark approximates the nonlinear function:
$z = sin(3x).cos(2y) + 0.5xy$
This function is chosen for its complexity and smoothness, making it ideal for testing activation behavior, gradient flow, and generalization.

## Default Configuration

- **Network Architecture**: [2, 8, 8, 1]
- **Activation Function**: tanh
- **Training Samples**: 70000 input vectors
- **Epochs**: 50
- **Batch Size**: 256
- **Learning Rate**: 0.01

## Reported Metrics

- **Training time**: 1m 47s
- **Average Absolute Error**: 0.202485
- **Initial Loss**: 0.175282
- **Final Loss**: 0.060117

#### output:

```
Epoch 0 - Avg Loss: 0.175282
Epoch 1 - Avg Loss: 0.141020
Epoch 2 - Avg Loss: 0.131166
Epoch 3 - Avg Loss: 0.125177
Epoch 4 - Avg Loss: 0.118417
Epoch 5 - Avg Loss: 0.109540
Epoch 6 - Avg Loss: 0.100434
Epoch 7 - Avg Loss: 0.093290
Epoch 8 - Avg Loss: 0.088716
Epoch 9 - Avg Loss: 0.086314
Epoch 10 - Avg Loss: 0.084325
Epoch 11 - Avg Loss: 0.082766
Epoch 12 - Avg Loss: 0.081314
Epoch 13 - Avg Loss: 0.079678
Epoch 14 - Avg Loss: 0.078286
Epoch 15 - Avg Loss: 0.076895
Epoch 16 - Avg Loss: 0.075618
Epoch 17 - Avg Loss: 0.074455
Epoch 18 - Avg Loss: 0.073529
Epoch 19 - Avg Loss: 0.072553
Epoch 20 - Avg Loss: 0.071750
Epoch 21 - Avg Loss: 0.071086
Epoch 22 - Avg Loss: 0.070459
Epoch 23 - Avg Loss: 0.069843
Epoch 24 - Avg Loss: 0.069298
Epoch 25 - Avg Loss: 0.068835
Epoch 26 - Avg Loss: 0.068325
Epoch 27 - Avg Loss: 0.067898
Epoch 28 - Avg Loss: 0.067501
Epoch 29 - Avg Loss: 0.067072
Epoch 30 - Avg Loss: 0.066703
Epoch 31 - Avg Loss: 0.066325
Epoch 32 - Avg Loss: 0.065890
Epoch 33 - Avg Loss: 0.065547
Epoch 34 - Avg Loss: 0.065168
Epoch 35 - Avg Loss: 0.064815
Epoch 36 - Avg Loss: 0.064480
Epoch 37 - Avg Loss: 0.064156
Epoch 38 - Avg Loss: 0.063804
Epoch 39 - Avg Loss: 0.063458
Epoch 40 - Avg Loss: 0.063120
Epoch 41 - Avg Loss: 0.062810
Epoch 42 - Avg Loss: 0.062491
Epoch 43 - Avg Loss: 0.062133
Epoch 44 - Avg Loss: 0.061750
Epoch 45 - Avg Loss: 0.061466
Epoch 46 - Avg Loss: 0.061108
Epoch 47 - Avg Loss: 0.060827
Epoch 48 - Avg Loss: 0.060408
Epoch 49 - Avg Loss: 0.060117
Training time: 107508 ms
Average absolute error: 0.202485
```

# Coding Conventions

1. **Explicit Typing** All variables, parameters, and return types are explicitly declared to improve readability and reduce ambiguity.=
2. **Clear and Purposeful Comments** Each function and section includes descriptive comments explaining its role and behavior. Comments are used to clarify intent, not restate code.
3. **Docstrings for Classes and Methods** Every class and method begins with a concise docstring describing its purpose and scope. While GDScript does not support structured annotations, natural-language summaries are used consistently.
4. **Single-Responsibility Functions** Functions are designed to perform one task only. If a function grows too long or handles multiple concerns, it is refactored into helper methods.
5. **Modular and Scalable Design** Code is organized into reusable components with clear boundaries — separating model logic, training flow, GPU dispatch, and utilities.
6. **Refactoring When Needed** Redundant or unclear logic is regularly revisited and improved. Naming, structure, and performance are refined as the project evolves.
7. **Proper and Consistent Naming** Variables, functions, and classes use descriptive, context-appropriate names. Naming reflects intent and avoids ambiguity.
8. **Tab Indentation** All scripts use tab-based indentation for consistency across the codebase.
