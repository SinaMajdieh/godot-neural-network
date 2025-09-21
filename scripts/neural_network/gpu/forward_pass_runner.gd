# ForwardPassRunner.gd
# Why: Runs the forward-pass compute shader for the full network.
# Handles resource allocation, uniform binding, and dispatching.

extends BaseShaderRunner
class_name ForwardPassRunner

const DEFAULT_SHADER_PATH: String = "res://scripts/neural_network/gpu/shaders/forward_pass.spv"

var forward_shader: RID
var forward_pipeline: RID

func _init(shader_path: String = DEFAULT_SHADER_PATH) -> void:
	super()
	forward_shader = _load_shader(shader_path)
	forward_pipeline = rd.compute_pipeline_create(forward_shader)

func dispatch_full_network(
	flat_inputs: PackedFloat32Array,
	flat_weights: PackedFloat32Array,
	flat_biases: PackedFloat32Array,
	layer_activations: PackedInt32Array,
	meta_data: PackedByteArray,
	intermediate_size: int,
	total_threads: int
) -> RID:
	var input_buf: RID = create_buffer(flat_inputs)
	var weight_buf: RID = create_buffer(flat_weights)
	var bias_buf: RID = create_buffer(flat_biases)
	var activation_buf: RID = create_buffer(layer_activations)
	var interm_buf: RID = create_empty_buffer(intermediate_size)
	var meta_buf: RID = rd.storage_buffer_create(
		meta_data.size(),
		meta_data
	)

	var uniform_set: RID = _create_uniform_set([
		_create_uniform(input_buf, 0),
		_create_uniform(weight_buf, 1),
		_create_uniform(bias_buf, 2),
		_create_uniform(activation_buf, 3),
		_create_uniform(interm_buf, 4),
		_create_uniform(meta_buf, 5)
	], forward_shader)

	_dispatch_compute(forward_pipeline, uniform_set, total_threads)

	rd.free_rid(input_buf)
	rd.free_rid(weight_buf)
	rd.free_rid(bias_buf)
	rd.free_rid(meta_buf)

	return interm_buf

static func build_meta_buffer(
	layer_count: int,
	batch_size: int,
	input_sizes: Array[int],
	output_sizes: Array[int],
	weight_offsets: Array[int],
	bias_offsets: Array[int],
	interm_offsets: Array[int]
) -> PackedByteArray:
	const MAX_LAYERS: int = 32
	const HEADER_UINTS: int = 2
	const ARRAYS_PER_META: int = 5

	var meta_bytes: PackedByteArray = PackedByteArray()
	meta_bytes.resize(
		(HEADER_UINTS + ARRAYS_PER_META * MAX_LAYERS) * 4
	)

	# Encode header
	meta_bytes.encode_u32(0, layer_count)
	meta_bytes.encode_u32(4, batch_size)

	# Encode layer arrays
	_encode_array_to_meta(meta_bytes, input_sizes, 0, MAX_LAYERS)
	_encode_array_to_meta(meta_bytes, output_sizes, 1, MAX_LAYERS)
	_encode_array_to_meta(meta_bytes, weight_offsets, 2, MAX_LAYERS)
	_encode_array_to_meta(meta_bytes, bias_offsets, 3, MAX_LAYERS)
	_encode_array_to_meta(meta_bytes, interm_offsets, 4, MAX_LAYERS)

	return meta_bytes

static func _encode_array_to_meta(
	meta_bytes: PackedByteArray,
	arr: Array[int],
	slot_index: int,
	max_len: int
) -> void:
	var base: int = 8
	for i: int in range(max_len):
		var val: int = arr[i] if i < arr.size() else 0
		var byte_pos: int = (
			base + ((slot_index * max_len + i) * 4)
		)
		meta_bytes.encode_u32(byte_pos, val)
