## Represents a fully connected layer in a feedforward neural network.
## Handles weight and bias initialization, gradient updates, and GPU-ready accessors.

extends RefCounted
class_name NetworkLayer

## Thresholds for gradient stability checks.
const GRADIENT_EXPLOSION_THRESHOLD: float = 1e6
const GRADIENT_THRESHOLD: float = 1.0

## Layer dimensions.
var input_size: int
var output_size: int

## Trainable parameters.
var weight_matrix: Array[Array] = []  ## Shape: [output_size][input_size]
var bias_vector: Array[float] = []    ## Shape: [output_size]

## Initializes the layer with given input/output sizes and randomized parameters.
func _init(input_size_: int, output_size_: int) -> void:
    input_size = input_size_
    output_size = output_size_
    _initialize_parameters()

## Initializes weights and biases using scaled uniform distribution.
func _initialize_parameters() -> void:
    weight_matrix = _generate_weight_matrix(input_size, output_size)
    bias_vector = _generate_bias_vector(output_size)

## Generates a weight matrix of shape [output_size][input_size] with values in [-scale, scale].
static func _generate_weight_matrix(in_size: int, out_size: int) -> Array[Array]:
    var matrix: Array[Array] = []
    var scale: float = sqrt(6.0 / (in_size + out_size))  ## Xavier-like scaling
    for out_idx in range(out_size):
        var row: Array[float] = []
        for in_idx in range(in_size):
            row.append(randf_range(-scale, scale))
        matrix.append(row)
    return matrix

## Generates a bias vector of shape [output_size] with values in [-1, 1].
static func _generate_bias_vector(out_size: int) -> Array[float]:
    var biases: Array[float] = []
    for _i in range(out_size):
        biases.append(randf_range(-1.0, 1.0))
    return biases

## Returns a flattened weight array suitable for GPU upload.
func get_flat_weights() -> Array[float]:
    var flat: Array[float] = []
    for row in weight_matrix:
        flat.append_array(row)
    return flat

## Accessor for full weight matrix.
func get_weight_matrix() -> Array[Array]:
    return weight_matrix

## Accessor for bias vector.
func get_bias_vector() -> Array[float]:
    return bias_vector

## Updates the weight matrix with new values.
func set_weight_matrix(new_weights: Array[Array]) -> void:
    weight_matrix = new_weights

## Updates the bias vector with new values.
func set_bias_vector(new_biases: Array[float]) -> void:
    bias_vector = new_biases

## Applies gradients to weights and biases using learning rate and batch size.
## Includes NaN and explosion detection for numerical stability.
func update_parameters_with_gradients(
        weight_grads: PackedFloat32Array,
        bias_grads: PackedFloat32Array,
        lr: float,
        batch_size: int
) -> void:
    var reshaped: Array[Array] = TensorUtils.reshape_weights(weight_grads, input_size, output_size)
    for i in range(output_size):
        for j in range(input_size):
            var grad: float = reshaped[i][j]
            if TensorUtils.is_nan_or_exploding(grad, GRADIENT_EXPLOSION_THRESHOLD):
                push_error("Exploding gradient in weight [%d][%d]" % [i, j])
                grad = clamp(grad, -GRADIENT_THRESHOLD, GRADIENT_THRESHOLD)
            weight_matrix[i][j] -= (lr / batch_size) * grad

        var bias_grad: float = bias_grads[i]
        if TensorUtils.is_nan_or_exploding(bias_grad, GRADIENT_EXPLOSION_THRESHOLD):
            push_error("Exploding gradient in bias [%d]" % i)
            bias_grad = clamp(bias_grad, -GRADIENT_THRESHOLD, GRADIENT_THRESHOLD)
        bias_vector[i] -= (lr / batch_size) * bias_grad
