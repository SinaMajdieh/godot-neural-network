# BackwardPassRunner.gd
# Why: Runs GPU backward-pass shaders for weight and bias gradient calc.

extends BaseShaderRunner
class_name BackwardPassRunner

const DEFAULT_SHADER_PATH: String = "res://scripts/neural_network/gpu/shaders/backward_pass.spv"

var backward_shader: RID
var backward_pipeline: RID

func _init(shader_path: String = DEFAULT_SHADER_PATH) -> void:
	super()
	backward_shader = _load_shader(shader_path)
	backward_pipeline = rd.compute_pipeline_create(backward_shader)

func dispatch_backward(
	activation: PackedFloat32Array,
	error: PackedFloat32Array,
	input: PackedFloat32Array,
	layer_activation: PackedInt32Array,
	input_size: int,
	output_size: int,
	num_vectors: int
) -> Array[RID]:
	# Create GPU buffers for inputs, activations, and gradients
	var activation_buf: RID = create_buffer(activation)
	var error_buf: RID = create_buffer(error)
	var input_buf: RID = create_buffer(input)
	var layer_activation_buf: RID = create_buffer(layer_activation)

	var weight_grad_buf: RID = create_empty_buffer(
		input_size * output_size
	)
	var bias_grad_buf: RID = create_empty_buffer(output_size)
	var meta_buf: RID = _create_meta_buffer(
		input_size,
		output_size,
		num_vectors
	)

	# Bind uniform set
	var uniform_set: RID = _create_uniform_set([
		_create_uniform(activation_buf, 0),
		_create_uniform(error_buf, 1),
		_create_uniform(input_buf, 2),
		_create_uniform(layer_activation_buf, 3),
		_create_uniform(weight_grad_buf, 4),
		_create_uniform(bias_grad_buf, 5),
		_create_uniform(meta_buf, 6, true)
	], backward_shader)

	# Dispatch compute
	_dispatch_compute(
		backward_pipeline,
		uniform_set,
		output_size * num_vectors
	)

	# Clean up
	rd.free_rid(activation_buf)
	rd.free_rid(error_buf)
	rd.free_rid(input_buf)
	rd.free_rid(meta_buf)

	return [weight_grad_buf, bias_grad_buf]

func _create_meta_buffer(
	input_size: int,
	output_size: int,
	num_vectors: int
) -> RID:
	var meta_data: PackedByteArray = PackedByteArray()
	meta_data.resize(16)
	meta_data.encode_u32(0, input_size)
	meta_data.encode_u32(4, output_size)
	meta_data.encode_u32(8, num_vectors)
	return rd.uniform_buffer_create(meta_data.size(), meta_data)
