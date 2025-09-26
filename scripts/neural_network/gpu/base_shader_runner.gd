# BaseShaderRunner.gd
# Why: Provides common GPU utilities for shader runners.
# Manages rendering device creation, shader loading, buffer utils,
# uniform creation, and compute dispatching.

extends RefCounted
class_name BaseShaderRunner

var rd: RenderingDevice

func _init() -> void:
	rd = RenderingServer.create_local_rendering_device()

func _load_shader(path: String) -> RID:
	var spirv_bytes: PackedByteArray = (
		FileAccess.get_file_as_bytes(path)
	)
	var spirv: RDShaderSPIRV = RDShaderSPIRV.new()
	spirv.set_stage_bytecode(
		RenderingDevice.SHADER_STAGE_COMPUTE,
		spirv_bytes
	)
	return rd.shader_create_from_spirv(spirv)

func create_buffer(data: Variant) -> RID:
	return rd.storage_buffer_create(
		data.size() * 4,
		data.to_byte_array()
	)

func create_uniform_buffer(data: Variant) -> RID:
	return rd.uniform_buffer_create(
		data.size() * 4,
		data.to_byte_array()
	)

func create_empty_buffer(size_in_floats: int) -> RID:
	var data: PackedFloat32Array = PackedFloat32Array()
	data.resize(size_in_floats)
	data.fill(0.0)
	return create_buffer(data)

func get_buffer_data(buffer: RID) -> PackedByteArray:
	return rd.buffer_get_data(buffer)

func _create_uniform(
	buffer: RID,
	binding: int,
	is_uniform: bool = false
) -> RDUniform:
	var u: RDUniform = RDUniform.new()
	u.uniform_type = (
		RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER if is_uniform
		else RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	)
	u.binding = binding
	u.add_id(buffer)
	return u

func _create_uniform_set(
	uniforms: Array[RDUniform],
	shader: RID
) -> RID:
	return rd.uniform_set_create(uniforms, shader, 0)

func _dispatch_compute(
	pipeline: RID,
	uniform_set: RID,
	total_threads: int,
	threads_per_group: int = 64
) -> void:
	var cl: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	var workgroups: int = int(
		ceil(total_threads / float(threads_per_group))
	)
	rd.compute_list_dispatch(cl, workgroups, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
# New method for 2D grid dispatch
func _dispatch_compute_2d(
	pipeline: RID,
	uniform_set: RID,
	total_threads_x: int,
	total_threads_y: int,
	threads_per_group_x: int = 64,
	threads_per_group_y: int = 64
) -> void:
	var cl: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	var workgroups_x: int = int(ceil(total_threads_x / float(threads_per_group_x)))
	var workgroups_y: int = int(ceil(total_threads_y / float(threads_per_group_y)))
	rd.compute_list_dispatch(cl, workgroups_x, workgroups_y, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
