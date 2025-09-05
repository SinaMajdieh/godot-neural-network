extends RefCounted
class_name NetworkLayer

##
## Represents a single layer in a feedforward neural network.
## Stores weights and biases, and provides accessors and update logic.
##

# Number of input and output nodes for this layer
var input_size: int
var output_size: int

# 2D array of weights: [output_index][input_index]
var weight_matrix: Array[Array] = []

# Bias vector for each output node
var bias_vector: Array[float] = []

##
## Constructor: initializes weights and biases with random values in range [-1, 1].
##
func _init(input_size_: int, output_size_: int) -> void:
    input_size = input_size_
    output_size = output_size_
    _initialize_weights()
    _initialize_biases()

##
## Initializes the weight matrix with random values.
##
func _initialize_weights() -> void:
    weight_matrix.resize(output_size)
    for out_idx: int in range(output_size):
        weight_matrix[out_idx].resize(input_size)
        for in_idx: int in range(input_size):
            weight_matrix[out_idx][in_idx] = randf_range(-1.0, 1.0)

##
## Initializes the bias vector with random values.
##
func _initialize_biases() -> void:
    bias_vector.resize(output_size)
    for out_idx: int in range(output_size):
        bias_vector[out_idx] = randf_range(-1.0, 1.0)

##
## Returns weights as a flat array for GPU upload.
##
func get_flat_weights() -> Array[float]:
    var flat_weights: Array[float] = []
    for row: Array[float] in weight_matrix:
        flat_weights.append_array(row)
    return flat_weights

##
## Returns the full weight matrix.
##
func get_weight_matrix() -> Array[Array]:
    return weight_matrix

##
## Returns the bias vector.
##
func get_bias_vector() -> Array[float]:
    return bias_vector

##
## Replaces the weight matrix with new values.
##
func set_weight_matrix(new_weights: Array[Array]) -> void:
    weight_matrix = new_weights

##
## Replaces the bias vector with new values.
##
func set_bias_vector(new_biases: Array[float]) -> void:
    bias_vector = new_biases

##
## Applies computed gradients to update weights and biases of this layer.
## Performs gradient scaling and NaN/explosion detection.
##
func apply_gradients(weight_grads: PackedFloat32Array, bias_grads: PackedFloat32Array, lr: float, batch_size: int) -> void:
    var reshaped_weights: Array[Array] = TensorUtils.reshape_weights(weight_grads, input_size, output_size)

    for i: int in range(output_size):
        for j: int in range(input_size):
            var grad: float = reshaped_weights[i][j]
            if TensorUtils.is_nan_or_exploding(grad, 1e6):
                push_error("NaN or exploding gradient detected in weight update at [%d][%d]" % [i, j])
            weight_matrix[i][j] -= (lr / batch_size) * grad

        var bias_grad: float = bias_grads[i]
        if TensorUtils.is_nan_or_exploding(bias_grad, 1e6):
            push_error("NaN or exploding gradient detected in bias update at [%d]" % i)
        bias_vector[i] -= (lr / batch_size) * bias_grad
