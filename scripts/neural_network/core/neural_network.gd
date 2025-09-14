extends RefCounted
class_name NeuralNetwork

##
## Represents a feedforward neural network composed of multiple layers.
## Handles layer initialization, forward propagation, and GPU-based execution.
##

var layers: Array[NetworkLayer] = []
var runner: ShaderRunner
var cached_layer_outputs: Array[PackedFloat32Array] = []

##
## Constructs the network using the given layer sizes and GPU shader runner.
##
func _init(layer_sizes: Array[int], runner_: ShaderRunner) -> void:
    runner = runner_
    _initialize_layers(layer_sizes)

##
## Creates and stores NetworkLayer instances based on the architecture.
##
func _initialize_layers(layer_sizes: Array[int]) -> void:
    for i: int in range(layer_sizes.size() - 1):
        var input_size: int = layer_sizes[i]
        var output_size: int = layer_sizes[i + 1]
        var layer: NetworkLayer = NetworkLayer.new(input_size, output_size)
        layers.append(layer)

##
## Returns all weights across layers as a single flat array.
##
func get_all_flat_weights() -> PackedFloat32Array:
    var flat: PackedFloat32Array = PackedFloat32Array()
    for layer: NetworkLayer in layers:
        var weights: Array[float] = layer.get_flat_weights()
        flat.append_array(weights)
    return flat

##
## Returns all biases across layers as a single flat array.
##
func get_all_biases() -> PackedFloat32Array:
    var flat: PackedFloat32Array = PackedFloat32Array()
    for layer: NetworkLayer in layers:
        var biases: Array[float] = layer.get_bias_vector()
        flat.append_array(biases)
    return flat

##
## Performs a forward pass on the input batch using GPU acceleration.
## Returns the final output from the last layer.
##
func forward_pass(input_batch: Array[PackedFloat32Array]) -> PackedFloat32Array:
    cached_layer_outputs.clear()

    var batch_size: int = input_batch.size()
    var flat_inputs: PackedFloat32Array = TensorUtils.flatten_batch(input_batch)
    var flat_weights: PackedFloat32Array = get_all_flat_weights()
    var flat_biases: PackedFloat32Array = get_all_biases()

    var metadata: NetworkUtils.LayerMetadata = NetworkUtils.compute_layer_metadata(layers, batch_size)
    var meta_bytes: PackedByteArray = ShaderRunner.build_meta_buffer(
        layers.size(),
        batch_size,
        metadata.input_sizes,
        metadata.output_sizes,
        metadata.weight_offsets,
        metadata.bias_offsets,
        metadata.interm_offsets
    )

    var threads: int = batch_size * metadata.output_sizes.max()
    var output_buffer: RID = runner.dispatch_full_network(
        flat_inputs,
        flat_weights,
        flat_biases,
        meta_bytes,
        metadata.total_intermediates,
        threads
    )

    var output_floats: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(output_buffer))
    runner.rd.free_rid(output_buffer)

    cached_layer_outputs = NetworkUtils.split_intermediates_to_layers(
        output_floats,
        metadata.interm_offsets,
        metadata.output_sizes,
        batch_size
    )

    return cached_layer_outputs[-1]