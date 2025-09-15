## Defines supported activation function types for neural network layers.

class_name Activations
extends RefCounted

## Enum representing available activation function types.
## Used to configure behavior of neural network layers.
enum Type {
    LINEAR,       ## Linear activation: f(x) = x
    SIGMOID,      ## Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    TANH,         ## Hyperbolic tangent: f(x) = tanh(x)
    RELU,         ## Rectified Linear Unit: f(x) = max(0, x)
    LEAKY_RELU,   ## Leaky ReLU: f(x) = x if x > 0 else αx
    SOFTMAX       ## Softmax: f(x_i) = exp(x_i) / Σ exp(x_j)
}