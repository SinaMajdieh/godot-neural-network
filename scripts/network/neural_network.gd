extends RefCounted
class_name NeuralNetwork

##
## Represents a feedforward neural network composed of multiple layers.
## Handles initialization, forward propagation, and GPU-based execution.
##

# List of layers in the neural network
var layers: Array[NetworkLayer] = []

# Shader runner for GPU dispatch
var runner: ShaderRunner

# Stores intermediate outputs from each layer during forward pass
var cached_layer_outputs: Array[PackedFloat32Array] = []

##
## Initializes the network with the given layer sizes and shader runner.
##
func _init(layer_sizes: Array[int], runner_: ShaderRunner) -> void:
    runner = runner_
    _initialize_layers(layer_sizes)

##
## Creates and initializes layers based on the provided architecture.
##
func _initialize_layers(layer_sizes: Array[int]) -> void:
    layers.resize(layer_sizes.size() - 1)
    for i: int in range(layers.size()):
        var input_size: int = layer_sizes[i]
        var output_size: int = layer_sizes[i + 1]
        layers[i] = NetworkLayer.new(input_size, output_size)

##
## Returns all weights in the network as a single flat array.
##
func get_all_flat_weights() -> PackedFloat32Array:
    var flat_weights: PackedFloat32Array = PackedFloat32Array()
    for layer: NetworkLayer in layers:
        var weights: Array[float] = layer.get_flat_weights()
        flat_weights.append_array(weights)
    return flat_weights

##
## Returns all biases in the network as a single flat array.
##
func get_all_biases() -> PackedFloat32Array:
    var flat_biases: PackedFloat32Array = PackedFloat32Array()
    for layer: NetworkLayer in layers:
        var biases: Array[float] = layer.get_bias_vector()
        flat_biases.append_array(biases)
    return flat_biases

##
## Performs a forward pass through the network using compute shaders.
## Returns the final output vector.
##
func forward_pass(input_batch: Array[PackedFloat32Array]) -> PackedFloat32Array:
    cached_layer_outputs.clear()
    var flat_inputs: PackedFloat32Array = TensorUtils.flatten_batch(input_batch)
    var current_buffer: RID = runner.create_buffer(flat_inputs)

    for layer_index: int in range(layers.size()):
        current_buffer = _dispatch_forward_layer(layers[layer_index], current_buffer, input_batch.size())
        var output_bytes: PackedByteArray = runner.rd.buffer_get_data(current_buffer)
        var output: PackedFloat32Array = TensorUtils.bytes_to_floats(output_bytes)
        _clamp_activation(output, layer_index)
        cached_layer_outputs.append(output)

    runner.rd.free_rid(current_buffer)
    return cached_layer_outputs[-1]

##
## Clamps activation values to prevent overflow and logs errors.
##
func _clamp_activation(activation: PackedFloat32Array, layer_index: int) -> void:
    for i: int in range(activation.size()):
        if TensorUtils.is_nan_or_exploding(activation[i], 100.0):
            push_error("Activation overflow detected at layer %d, index %d" % [layer_index, i])
            activation[i] = clamp(activation[i], -10.0, 10.0)

##
## Dispatches the forward pass for a single layer using compute shaders.
##
func _dispatch_forward_layer(layer: NetworkLayer, input_buf: RID, batch_size: int) -> RID:
    var weights: PackedFloat32Array = PackedFloat32Array(layer.get_flat_weights())
    var biases: PackedFloat32Array = PackedFloat32Array(layer.get_bias_vector())
    return runner.dispatch_forward(
        input_buf,
        weights,
        biases,
        layer.input_size,
        layer.output_size,
        batch_size
    )
