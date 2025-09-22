class_name Activations
extends RefCounted

enum Type {
	LINEAR,      # f(x) = x
	SIGMOID,     # f(x) = 1 / (1 + exp(-x))
	TANH,        # f(x) = tanh(x)
	RELU,        # f(x) = max(0, x)
	LEAKY_RELU,  # f(x) = x>0 ? x : αx
	SOFTMAX      # f(xi) = exp(xi) / Σ exp(xj)
}

static func activation_derivative(
	activation: float, activation_type: Type
) -> float:
	# For output-layer derivative we use the *activated value*
	# This avoids recomputing expensive functions from z
	match activation_type:
		Type.TANH:
			return compute_derivative_tanh(activation)
		Type.SIGMOID:
			return compute_derivative_sigmoid(activation)
		Type.RELU:
			return compute_derivative_relu(activation)
		Type.LEAKY_RELU:
			return compute_derivative_leaky_relu(activation)
		_:
			return 1.0  # Linear, Softmax not derived here

static func compute_derivative_tanh(activation: float) -> float:
	# tanh' = 1 - a²
	return 1.0 - activation * activation

static func compute_derivative_sigmoid(activation: float) -> float:
	# σ'(a) = a(1-a) when a is already σ(z)
	return activation * (1.0 - activation)

static func compute_derivative_relu(activation: float) -> float:
	# ReLU' = 1 if a>0, else 0
	return 1.0 if activation > 0.0 else 0.0

static func compute_derivative_leaky_relu(activation: float) -> float:
	# Leaky ReLU' = 1 if a>0 else small slope (α)
	return 1.0 if activation > 0.0 else 0.01
