## NeuralNetwork.gd
## Represents a feedforward neural network composed of multiple layers.
## Handles layer initialization, forward propagation, and GPU-based execution.

extends RefCounted
class_name NeuralNetwork

const KEYS: Dictionary = {
	LAYER_SIZES    = "layer_sizes",
	RUNNER         = "runner",
	HIDDEN_ACT     = "hidden_act",
	OUTPUT_ACT     = "output_act",
	WEIGHT_INIT    = "weight_init_method"
}

## Network architecture and execution context.
var layers: Array[NetworkLayer] = []                         ## Ordered list of network layers
var layers_activation: Array[Activations.Type] = []          ## Activation type per layer
var runner: ForwardPassRunner                                     ## GPU shader runner
var cached_post_act_layer_outputs: Array[PackedFloat32Array] = []     ## Cached outputs from each layer
var cached_pre_act_layer_outputs: Array[PackedFloat32Array] = []     ## Cached outputs from each layer

## Constructs the network using the given layer sizes and GPU shader runner.
## @param layer_sizes List of neuron counts per layer
## @param runner_ ForwardPassRunner instance for GPU dispatch
## @param hidden_act Activation type for hidden layers
## @param output_act Activation type for output layer
func _init(config: Dictionary) -> void:
	var defaults: Dictionary = {
		KEYS.LAYER_SIZES: [] as Array[int],
		KEYS.RUNNER: null,
		KEYS.HIDDEN_ACT: Activations.Type.RELU,
		KEYS.OUTPUT_ACT: Activations.Type.SIGMOID,
		KEYS.WEIGHT_INIT: NetworkLayer.WeightInitialization.XAVIER
	}

	config = defaults.merged(config, true)

	runner = config[KEYS.RUNNER]
	_initialize_layers(config[KEYS.LAYER_SIZES], config[KEYS.HIDDEN_ACT], config[KEYS.OUTPUT_ACT], config[KEYS.WEIGHT_INIT])

## Creates and stores NetworkLayer instances based on the architecture.
## Assigns appropriate activation functions to each layer.
func _initialize_layers(
		layer_sizes: Array[int],
		hidden_act: Activations.Type,
		output_act: Activations.Type,
		weight_init_method: NetworkLayer.WeightInitialization
) -> void:
	var num_layers: int = layer_sizes.size() - 1
	for i: int in range(num_layers):
		var input_size: int = layer_sizes[i]
		var output_size: int = layer_sizes[i + 1]
		var layer: NetworkLayer = NetworkLayer.new(input_size, output_size, weight_init_method)
		layers.append(layer)

		var is_output_layer: bool = (i == num_layers - 1)
		var act: Activations.Type = output_act if is_output_layer else hidden_act
		layers_activation.append(act)

## Returns all weights across layers as a single flat array.
## Used for GPU upload and shader execution.
func get_all_flat_weights() -> PackedFloat32Array:
	var flat: PackedFloat32Array = PackedFloat32Array()
	for layer: NetworkLayer in layers:
		flat.append_array(layer.get_flat_weights())
	return flat

## Returns all biases across layers as a single flat array.
## Used for GPU upload and shader execution.
func get_all_biases() -> PackedFloat32Array:
	var flat: PackedFloat32Array = PackedFloat32Array()
	for layer: NetworkLayer in layers:
		flat.append_array(layer.get_bias_vector())
	return flat

## Performs a forward pass on the input batch using GPU acceleration.
## @param input_batch Array of PackedFloat32Array inputs (one per sample)
## @return Final output from the last layer
func forward_pass(input_batch: Array[PackedFloat32Array], reutrn_pre_act: bool = false) -> PackedFloat32Array:
	cached_post_act_layer_outputs.clear()

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
	var output_buffers: Dictionary = runner.dispatch_full_network(
		flat_inputs,
		flat_weights,
		flat_biases,
		layers_activation,
		meta_bytes,
		metadata.total_intermediates,
		threads
	)

	var post_act_output_floats: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(output_buffers.POST_ACT))
	var pre_act_output_floats: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(output_buffers.PRE_ACT))

	runner.rd.free_rid(output_buffers.POST_ACT)
	runner.rd.free_rid(output_buffers.PRE_ACT)

	cached_post_act_layer_outputs = NetworkUtils.split_intermediates_to_layers(
		post_act_output_floats,
		metadata.interm_offsets,
		metadata.output_sizes,
		batch_size
	)
	cached_pre_act_layer_outputs = NetworkUtils.split_intermediates_to_layers(
		pre_act_output_floats,
		metadata.interm_offsets,
		metadata.output_sizes,
		batch_size
	)

	if reutrn_pre_act:
		return cached_pre_act_layer_outputs[-1]
	else:
		return cached_post_act_layer_outputs[-1]
