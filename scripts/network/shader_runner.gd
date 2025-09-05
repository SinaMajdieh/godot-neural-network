extends RefCounted
class_name ShaderRunner

##
## Manages GPU resources and dispatches compute shaders for neural network operations.
## Handles shader compilation, buffer creation, uniform binding, and pipeline execution.
##

# Rendering device and shader pipelines
var rd: RenderingDevice
var forward_shader: RID
var forward_pipeline: RID
var backward_shader: RID
var backward_pipeline: RID

##
## Initializes the rendering device and compiles both forward and backward shaders.
##
## @param forward_shader_path Path to forward shader SPIR-V bytecode
## @param backward_shader_path Path to backward shader SPIR-V bytecode
##
func _init(forward_shader_path: String, backward_shader_path: String) -> void:
    rd = RenderingServer.create_local_rendering_device()
    forward_shader = _load_shader(forward_shader_path)
    forward_pipeline = rd.compute_pipeline_create(forward_shader)
    backward_shader = _load_shader(backward_shader_path)
    backward_pipeline = rd.compute_pipeline_create(backward_shader)

##
## Loads and compiles a compute shader from SPIR-V bytecode.
##
## @param path File path to SPIR-V bytecode
## @return Compiled shader RID
##
func _load_shader(path: String) -> RID:
    var spirv_bytes: PackedByteArray = FileAccess.get_file_as_bytes(path)
    var spirv: RDShaderSPIRV = RDShaderSPIRV.new()
    spirv.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, spirv_bytes)
    return rd.shader_create_from_spirv(spirv)

##
## Creates a GPU buffer from a PackedFloat32Array.
##
## @param data Float array to upload
## @return Storage buffer RID
##
func create_buffer(data: PackedFloat32Array) -> RID:
    var byte_data: PackedByteArray = data.to_byte_array()
    return rd.storage_buffer_create(byte_data.size(), byte_data)

##
## Creates an empty GPU buffer of specified float size.
##
## @param size_in_floats Number of floats to allocate
## @return Empty storage buffer RID
##
func create_empty_buffer(size_in_floats: int) -> RID:
    var byte_data: PackedByteArray = PackedByteArray()
    byte_data.resize(size_in_floats * 4)
    return rd.storage_buffer_create(byte_data.size(), byte_data)

##
## Retrieves data from a GPU buffer.
##
## @param buffer Buffer RID to read from
## @return Byte array containing buffer contents
##
func get_buffer_data(buffer: RID) -> PackedByteArray:
    return rd.buffer_get_data(buffer)

##
## Dispatches the forward compute shader with input, weights, and biases.
##
## @param input_buf Input buffer RID
## @param weights Flat weight array
## @param biases Flat bias array
## @param input_size Number of input neurons
## @param output_size Number of output neurons
## @param num_vectors Number of input samples
## @return Output buffer RID
##
func dispatch_forward(
    input_buf: RID,
    weights: PackedFloat32Array,
    biases: PackedFloat32Array,
    input_size: int,
    output_size: int,
    num_vectors: int
) -> RID:
    var weight_buf: RID = create_buffer(weights)
    var bias_buf: RID = create_buffer(biases)
    var output_buf: RID = create_empty_buffer(output_size * num_vectors)
    var meta_buf: RID = _create_meta_buffer(input_size, output_size, num_vectors)

    var uniform_set: RID = _create_uniform_set([
        _create_uniform(input_buf, 0),
        _create_uniform(weight_buf, 1),
        _create_uniform(bias_buf, 2),
        _create_uniform(output_buf, 3),
        _create_uniform(meta_buf, 4, true)
    ], forward_shader)

    _dispatch_compute(forward_pipeline, uniform_set, output_size * num_vectors)

    rd.free_rid(weight_buf)
    rd.free_rid(bias_buf)
    rd.free_rid(meta_buf)

    return output_buf

##
## Dispatches the backward compute shader with activations, errors, and inputs.
##
## @param activation_buf Activation buffer RID
## @param error_buf Error buffer RID
## @param input_buf Input buffer RID
## @param input_size Number of input neurons
## @param output_size Number of output neurons
## @param num_vectors Number of input samples
## @return Array containing weight and bias gradient buffer RIDs
##
func dispatch_backward(
    activation_buf: RID,
    error_buf: RID,
    input_buf: RID,
    input_size: int,
    output_size: int,
    num_vectors: int
) -> Array[RID]:
    var weight_grad_buf: RID = create_empty_buffer(input_size * output_size)
    var bias_grad_buf: RID = create_empty_buffer(output_size)
    var meta_buf: RID = _create_meta_buffer(input_size, output_size, num_vectors)

    var uniform_set: RID = _create_uniform_set([
        _create_uniform(activation_buf, 0),
        _create_uniform(error_buf, 1),
        _create_uniform(input_buf, 2),
        _create_uniform(weight_grad_buf, 3),
        _create_uniform(bias_grad_buf, 4),
        _create_uniform(meta_buf, 5, true)
    ], backward_shader)

    _dispatch_compute(backward_pipeline, uniform_set, output_size * num_vectors)

    rd.free_rid(activation_buf)
    rd.free_rid(error_buf)
    rd.free_rid(input_buf)
    rd.free_rid(meta_buf)

    return [weight_grad_buf, bias_grad_buf]

##
## Creates a uniform buffer containing metadata (input/output sizes, batch size).
##
## @param input_size Number of input neurons
## @param output_size Number of output neurons
## @param num_vectors Number of input samples
## @return Uniform buffer RID
##
func _create_meta_buffer(input_size: int, output_size: int, num_vectors: int) -> RID:
    var meta_data: PackedByteArray = PackedByteArray()
    meta_data.resize(16)
    meta_data.encode_u32(0, input_size)
    meta_data.encode_u32(4, output_size)
    meta_data.encode_u32(8, num_vectors)
    return rd.uniform_buffer_create(meta_data.size(), meta_data)

##
## Creates a single RDUniform for a buffer.
##
## @param buffer Buffer RID
## @param binding Binding index
## @param is_uniform Whether the buffer is a uniform buffer
## @return RDUniform instance
##
func _create_uniform(buffer: RID, binding: int, is_uniform: bool = false) -> RDUniform:
    var uniform: RDUniform = RDUniform.new()
    uniform.uniform_type = (
        RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER if is_uniform
        else RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
    )
    uniform.binding = binding
    uniform.add_id(buffer)
    return uniform

##
## Creates a uniform set from a list of RDUniforms and a shader.
##
## @param uniforms Array of RDUniforms
## @param shader Shader RID
## @return Uniform set RID
##
func _create_uniform_set(uniforms: Array[RDUniform], shader: RID) -> RID:
    return rd.uniform_set_create(uniforms, shader, 0)

##
## Dispatches a compute shader with the given pipeline and uniform set.
##
## @param pipeline Compute pipeline RID
## @param uniform_set Uniform set RID
## @param total_threads Total number of threads to dispatch
##
func _dispatch_compute(pipeline: RID, uniform_set: RID, total_threads: int) -> void:
    var compute_list: int = rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
    rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
    var workgroups: int = int(ceil(total_threads / 64.0))
    rd.compute_list_dispatch(compute_list, workgroups, 1, 1)
    rd.compute_list_end()
    rd.submit()
    rd.sync()