extends RefCounted
class_name TensorUtils

##
## Utility class for tensor operations and data conversions.
## Provides batch flattening, reshaping, byte/float conversions, and safety checks.
##

##
## Flattens a batch of PackedFloat32Array vectors into a single PackedFloat32Array.
##
## @param batch Array of PackedFloat32Array vectors
## @return Flattened PackedFloat32Array
##
static func flatten_batch(batch: Array[PackedFloat32Array]) -> PackedFloat32Array:
    var flat: PackedFloat32Array = PackedFloat32Array()
    for vector: PackedFloat32Array in batch:
        flat.append_array(vector)
    return flat

##
## Reshapes a flat PackedFloat32Array into an array of PackedFloat32Array vectors.
##
## @param flat Flattened data
## @param vector_size Number of elements per vector
## @return Array of reshaped PackedFloat32Array vectors
##
static func unflatten_batch(flat: PackedFloat32Array, vector_size: int) -> Array[PackedFloat32Array]:
    var batch: Array[PackedFloat32Array] = []
    var total_vectors: int = flat.size() / vector_size
    for i: int in range(total_vectors):
        var vector: PackedFloat32Array = PackedFloat32Array()
        for j: int in range(vector_size):
            vector.append(flat[i * vector_size + j])
        batch.append(vector)
    return batch

##
## Converts a PackedByteArray to a PackedFloat32Array.
##
## @param bytes Byte array containing encoded float data
## @return Decoded float array
##
static func bytes_to_floats(bytes: PackedByteArray) -> PackedFloat32Array:
    var floats: PackedFloat32Array = PackedFloat32Array()
    var total_floats: int = bytes.size() / 4
    for i: int in range(total_floats):
        floats.append(bytes.decode_float(i * 4))
    return floats

##
## Converts a PackedFloat32Array to a PackedByteArray.
##
## @param floats Float array to encode
## @return Encoded byte array
##
static func floats_to_bytes(floats: PackedFloat32Array) -> PackedByteArray:
    return floats.to_byte_array()

##
## Reshapes a flat gradient array into a 2D weight matrix.
##
## @param flat Flat gradient array
## @param input_size Number of input neurons
## @param output_size Number of output neurons
## @return 2D weight matrix [output][input]
##
static func reshape_weights(flat: PackedFloat32Array, input_size: int, output_size: int) -> Array[Array]:
    var expected_size: int = input_size * output_size
    if flat.size() < expected_size:
        push_error("reshape_weights: Expected %d elements, got %d" % [expected_size, flat.size()])
        return []

    var matrix: Array[Array] = []
    for i: int in range(output_size):
        var row: Array[float] = []
        for j: int in range(input_size):
            row.append(flat[i * input_size + j])
        matrix.append(row)
    return matrix

##
## Splits input and target data into mini-batches.
##
## @param inputs Array of input vectors
## @param targets Array of target vectors
## @param batch_size Number of samples per batch
## @return Array of dictionaries with "inputs" and "targets"
##
static func create_batches(inputs: Array[PackedFloat32Array], targets: Array[PackedFloat32Array], batch_size: int) -> Array[Dictionary]:
    var batches: Array[Dictionary] = []
    var total_samples: int = inputs.size()

    for i: int in range(0, total_samples, batch_size):
        var input_batch: Array[PackedFloat32Array] = inputs.slice(i, i + min(batch_size, total_samples - i))
        var target_batch: Array[PackedFloat32Array] = targets.slice(i, i + min(batch_size, total_samples - i))
        batches.append({
            "inputs": input_batch,
            "targets": target_batch
        })
    return batches

##
## Shuffles input and target data in-place using a shared index permutation.
##
## @param inputs Array of input vectors
## @param targets Array of target vectors
##
static func shuffle_data(inputs: Array[PackedFloat32Array], targets: Array[PackedFloat32Array]) -> void:
    var indices: Array[int] = []
    for i: int in range(inputs.size()):
        indices.append(i)
    indices.shuffle()

    var shuffled_inputs: Array[PackedFloat32Array] = []
    var shuffled_targets: Array[PackedFloat32Array] = []
    for i: int in indices:
        shuffled_inputs.append(inputs[i])
        shuffled_targets.append(targets[i])

    inputs.clear()
    targets.clear()
    inputs.append_array(shuffled_inputs)
    targets.append_array(shuffled_targets)

##
## Checks if a value is NaN or exceeds a specified threshold.
##
## @param value Float value to check
## @param exploding_threshold Threshold for explosion detection
## @return True if value is NaN or exceeds threshold
##
static func is_nan_or_exploding(value: float, exploding_threshold: float) -> bool:
    return is_nan(value) or abs(value) > exploding_threshold
