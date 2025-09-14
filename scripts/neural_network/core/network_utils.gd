class_name NetworkUtils

##
## Utility class for computing metadata and slicing intermediate buffers
## during neural network execution. Used for GPU dispatch coordination.
##

class LayerMetadata:
	##
	## Stores metadata for each layer in the network.
	## Includes input/output sizes, buffer offsets, and total intermediate size.
	##
	var input_sizes: Array[int] = []
	var output_sizes: Array[int] = []
	var weight_offsets: Array[int] = []
	var bias_offsets: Array[int] = []
	var interm_offsets: Array[int] = []
	var total_intermediates: int = 0

##
## Computes metadata required for GPU shader execution.
##
## @param layers Array of NetworkLayer instances
## @param batch_size Number of samples per batch
## @return LayerMetadata containing offsets and sizes
##
static func compute_layer_metadata(layers: Array[NetworkLayer], batch_size: int) -> LayerMetadata:
	var meta: LayerMetadata = LayerMetadata.new()
	var w_off: int = 0
	var b_off: int = 0
	var total: int = 0

	for layer: NetworkLayer in layers:
		meta.input_sizes.append(layer.input_size)
		meta.output_sizes.append(layer.output_size)
		meta.weight_offsets.append(w_off)
		meta.bias_offsets.append(b_off)
		meta.interm_offsets.append(total)

		w_off += layer.input_size * layer.output_size
		b_off += layer.output_size
		total += batch_size * layer.output_size

	meta.total_intermediates = total
	return meta

##
## Splits a flat intermediate buffer into per-layer outputs.
##
## @param interm_floats Flattened output buffer from GPU
## @param interm_offsets Start indices for each layer's output
## @param output_sizes Number of outputs per layer
## @param batch_size Number of samples in batch
## @return Array of PackedFloat32Array outputs per layer
##
static func split_intermediates_to_layers(
	interm_floats: PackedFloat32Array,
	interm_offsets: Array[int],
	output_sizes: Array[int],
	batch_size: int
) -> Array[PackedFloat32Array]:
	var outputs: Array[PackedFloat32Array] = []

	for i: int in range(output_sizes.size()):
		var block_size: int = batch_size * output_sizes[i]
		var start: int = interm_offsets[i]
		var layer_output: PackedFloat32Array = interm_floats.slice(start, start + block_size)
		outputs.append(layer_output)

	return outputs
