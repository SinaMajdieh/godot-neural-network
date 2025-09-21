# GradClipRunner.gd
# Why: Combines gradient L2 norm computation and scaling on GPU for
# clipping. Uses grad_clip_norm.comp to sum squared values into a single
# float buffer, then grad_clip_scale.comp to scale gradients by the
# clipping factor — all on GPU to avoid CPU round‑trips.

extends BaseShaderRunner
class_name GradClipRunner

const DEFAULT_SHADER_PATHS: Dictionary = {
	NORM_SHADER_PATH = "res://scripts/neural_network/gpu/shaders/grad_clip_norm.spv",
	SCALE_SHADER_PATH = "res://scripts/neural_network/gpu/shaders/grad_clip_scale.spv"
}

var norm_shader: RID
var norm_pipeline: RID

var scale_shader: RID
var scale_pipeline: RID

func _init(
	norm_shader_path: String = DEFAULT_SHADER_PATHS.NORM_SHADER_PATH, 
	scale_shader_path: String = DEFAULT_SHADER_PATHS.SCALE_SHADER_PATH
) -> void:
	super()
	norm_shader = _load_shader(norm_shader_path)
	norm_pipeline = rd.compute_pipeline_create(norm_shader)
	scale_shader = _load_shader(scale_shader_path)
	scale_pipeline = rd.compute_pipeline_create(scale_shader)

func clip_gradients(grad_buf: RID, grad_count: int, clip_norm: float) -> void:
	var norm_buf: RID = dispatch_calc_norm(grad_buf, grad_count)
	dispatch_scale(grad_buf, norm_buf, grad_count, clip_norm)
	rd.free_rid(norm_buf)

func dispatch_calc_norm(grad_buf: RID, grad_count: int) -> RID:
	assert(grad_count > 0, "grad_count must be > 0")

	var norm_buf: RID = create_empty_buffer(1)

	# Encode grad_count as push constant (u32)
	var push_consts: PackedByteArray = PackedByteArray()
	push_consts.resize(16)
	push_consts.encode_u32(0, grad_count)

	# Bind buffers to shader
	var uniform_set: RID = _create_uniform_set([
		_create_uniform(grad_buf, 0),
		_create_uniform(norm_buf, 1)
	], norm_shader)

	# Integer ceiling for groups: covers all gradients
	const THREADS_PER_GROUP: int = 256
	var total_groups: int = ceil(grad_count / float(THREADS_PER_GROUP))
	# Dispatch compute
	var cl: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, norm_pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, push_consts, push_consts.size())
	rd.compute_list_dispatch(cl, total_groups, 1, 1)
	rd.compute_list_end()

	rd.submit()
	rd.sync()

	return norm_buf

func dispatch_scale(
	grad_buf: RID,
	norm_buf: RID,
	grad_count: int,
	clip_norm: float
) -> void:
	assert(grad_count > 0, "grad_count must be > 0")

	# Push constants (padded to 16 bytes for Vulkan alignment)
	var push_consts: PackedByteArray = PackedByteArray()
	push_consts.resize(16)
	push_consts.encode_u32(0, grad_count)
	push_consts.encode_float(4, clip_norm)

	var uniform_set: RID = _create_uniform_set([
		_create_uniform(grad_buf, 0),  # storage (read/write)
		_create_uniform(norm_buf, 1)   # storage (read-only)
	], scale_shader)

	# Integer ceiling for groups
	const THREADS_PER_GROUP: int = 256
	var total_groups: int = ceil(grad_count / float(THREADS_PER_GROUP))

	# Dispatch compute
	var cl: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, scale_pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, push_consts, push_consts.size())
	rd.compute_list_dispatch(cl, total_groups, 1, 1)
	rd.compute_list_end()

	rd.submit()
	rd.sync()
