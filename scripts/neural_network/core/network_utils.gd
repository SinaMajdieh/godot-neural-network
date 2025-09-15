## NetworkUtils.gd
## Utility class for computing metadata and slicing intermediate buffers
## during neural network execution. Supports GPU dispatch coordination.

class_name NetworkUtils

## Stores metadata for each layer in the network.
## Includes input/output sizes, buffer offsets, and total intermediate size.
class LayerMetadata:
    var input_sizes: Array[int] = []         ## Number of inputs per layer
    var output_sizes: Array[int] = []        ## Number of outputs per layer
    var weight_offsets: Array[int] = []      ## Start index of each layer's weights in flat buffer
    var bias_offsets: Array[int] = []        ## Start index of each layer's biases in flat buffer
    var interm_offsets: Array[int] = []      ## Start index of each layer's intermediate output
    var total_intermediates: int = 0         ## Total size of intermediate buffer across all layers

## Computes metadata required for GPU shader execution.
## @param layers Array of NetworkLayer instances
## @param batch_size Number of samples per batch
## @return LayerMetadata containing offsets and sizes
static func compute_layer_metadata(
        layers: Array[NetworkLayer],
        batch_size: int
) -> LayerMetadata:
    var meta: LayerMetadata = LayerMetadata.new()
    var weight_offset: int = 0
    var bias_offset: int = 0
    var interm_total: int = 0

    for layer: NetworkLayer in layers:
        meta.input_sizes.append(layer.input_size)
        meta.output_sizes.append(layer.output_size)
        meta.weight_offsets.append(weight_offset)
        meta.bias_offsets.append(bias_offset)
        meta.interm_offsets.append(interm_total)

        weight_offset += layer.input_size * layer.output_size
        bias_offset += layer.output_size
        interm_total += batch_size * layer.output_size

    meta.total_intermediates = interm_total
    return meta

## Splits a flat intermediate buffer into per-layer outputs.
## @param interm_floats Flattened output buffer from GPU
## @param interm_offsets Start indices for each layer's output
## @param output_sizes Number of outputs per layer
## @param batch_size Number of samples in batch
## @return Array of PackedFloat32Array outputs per layer
static func split_intermediates_to_layers(
        interm_floats: PackedFloat32Array,
        interm_offsets: Array[int],
        output_sizes: Array[int],
        batch_size: int
) -> Array[PackedFloat32Array]:
    var outputs: Array[PackedFloat32Array] = []

    for layer_index: int in range(output_sizes.size()):
        var block_size: int = batch_size * output_sizes[layer_index]
        var start_index: int = interm_offsets[layer_index]
        var layer_output: PackedFloat32Array = interm_floats.slice(start_index, start_index + block_size)
        outputs.append(layer_output)

    return outputs
