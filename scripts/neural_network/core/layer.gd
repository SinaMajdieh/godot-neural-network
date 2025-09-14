extends RefCounted
class_name NetworkLayer

##
## Represents a fully connected layer in a feedforward neural network.
## Stores and updates weights and biases, and provides accessors for GPU upload and training.
##

const GRADIENT_EXPLOSION_THRESHOLD: float = 1e6

var input_size: int
var output_size: int

var weight_matrix: Array[Array] = []
var bias_vector: Array[float] = []

##
## Constructs a layer with randomly initialized weights and biases.
##
func _init(input_size_: int, output_size_: int) -> void:
    input_size = input_size_
    output_size = output_size_
    _initialize_parameters()

##
## Initializes weights and biases using uniform values in [-1, 1].
## Replace with proper randomization when needed.
##
func _initialize_parameters() -> void:
    weight_matrix = _generate_weight_matrix(input_size, output_size)
    bias_vector = _generate_bias_vector(output_size)

##
## Generates a weight matrix of shape [output_size][input_size].
##
static func _generate_weight_matrix(in_size: int, out_size: int) -> Array[Array]:
    var matrix: Array[Array] = []
    for out_idx in range(out_size):
        var row: Array[float] = []
        for in_idx in range(in_size):
            row.append(randf_range(-1.0, 1.0)) 
        matrix.append(row)
    return matrix

##
## Generates a bias vector of length output_size.
##
static func _generate_bias_vector(out_size: int) -> Array[float]:
    var biases: Array[float] = []
    for out_idx in range(out_size):
        biases.append(randf_range(-1.0, 1.0))
    return biases

##
## Returns a flat array of weights for GPU upload.
##
func get_flat_weights() -> Array[float]:
    var flat: Array[float] = []
    for row in weight_matrix:
        flat.append_array(row)
    return flat

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
## Sets the weight matrix to new values.
##
func set_weight_matrix(new_weights: Array[Array]) -> void:
    weight_matrix = new_weights

##
## Sets the bias vector to new values.
##
func set_bias_vector(new_biases: Array[float]) -> void:
    bias_vector = new_biases

##
## Applies gradients to update weights and biases using learning rate and batch size.
## Includes NaN and explosion detection for stability.
##
func update_parameters_with_gradients(weight_grads: PackedFloat32Array, bias_grads: PackedFloat32Array, lr: float, batch_size: int) -> void:
    var reshaped: Array[Array] = TensorUtils.reshape_weights(weight_grads, input_size, output_size)

    for i in range(output_size):
        for j in range(input_size):
            var grad: float = reshaped[i][j]
            if TensorUtils.is_nan_or_exploding(grad, GRADIENT_EXPLOSION_THRESHOLD):
                push_error("Exploding gradient in weight [%d][%d]" % [i, j])
            weight_matrix[i][j] -= (lr / batch_size) * grad

        var bias_grad := bias_grads[i]
        if TensorUtils.is_nan_or_exploding(bias_grad, GRADIENT_EXPLOSION_THRESHOLD):
            push_error("Exploding gradient in bias [%d]" % i)
        bias_vector[i] -= (lr / batch_size) * bias_grad
