# LayerData.gd
# Resource container for neural network layer parameters.
# Why: Enables easy serialization and reuse of trained weights.

extends Resource
class_name LayerData

@export var input_size: int
@export var output_size: int

# Weight matrix: indexed [output][input], stores connection weights.
@export var weight_matrix: Array[Array]  # Each sub-array is Array[float].

# Bias vector: stores bias term for each output neuron.
@export var bias_vector: Array[float]
