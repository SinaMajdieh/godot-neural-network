extends RefCounted
class_name ShaderRunner

##
## Manages GPU resources and dispatches compute shaders for neural network operations.
## Handles shader compilation, buffer creation, uniform binding, and pipeline execution.
##

var rd: RenderingDevice
var forward_shader: RID
var forward_pipeline: RID
var backward_shader: RID
var backward_pipeline: RID

##
## Initializes the rendering device and compiles both forward and backward shaders.
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
func _load_shader(path: String) -> RID:
    var spirv_bytes: PackedByteArray = FileAccess.get_file_as_bytes(path)
    var spirv: RDShaderSPIRV = RDShaderSPIRV.new()
    spirv.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, spirv_bytes)
    return rd.shader_create_from_spirv(spirv)

##
## Creates a GPU buffer from a float array.
##
func create_buffer(data: PackedFloat32Array) -> RID:
    var byte_data: PackedByteArray = data.to_byte_array()
    return rd.storage_buffer_create(byte_data.size(), byte_data)

##
## Creates an empty GPU buffer of specified float size.
##
func create_empty_buffer(size_in_floats: int) -> RID:
    var byte_data: PackedByteArray = PackedByteArray()
    byte_data.resize(size_in_floats * 4)
    return rd.storage_buffer_create(byte_data.size(), byte_data)

##
## Retrieves data from a GPU buffer.
##
func get_buffer_data(buffer: RID) -> PackedByteArray:
    return rd.buffer_get_data(buffer)

##
## Dispatches the forward shader with input, weights, biases, and metadata.
##
func dispatch_full_network(
    flat_inputs: PackedFloat32Array,
    flat_weights: PackedFloat32Array,
    flat_biases: PackedFloat32Array,
    meta_data: PackedByteArray,
    intermediate_size: int,
    total_threads: int
) -> RID:
    var input_buf: RID = create_buffer(flat_inputs)
    var weight_buf: RID = create_buffer(flat_weights)
    var bias_buf: RID = create_buffer(flat_biases)
    var interm_buf: RID = create_empty_buffer(intermediate_size)
    var meta_buf: RID = rd.storage_buffer_create(meta_data.size(), meta_data)

    var uniform_set: RID = _create_uniform_set([
        _create_uniform(input_buf, 0),
        _create_uniform(weight_buf, 1),
        _create_uniform(bias_buf, 2),
        _create_uniform(interm_buf, 3),
        _create_uniform(meta_buf, 4)
    ], forward_shader)

    _dispatch_compute(forward_pipeline, uniform_set, total_threads)

    rd.free_rid(input_buf)
    rd.free_rid(weight_buf)
    rd.free_rid(bias_buf)
    rd.free_rid(meta_buf)

    return interm_buf

##
## Dispatches the backward shader to compute gradients.
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
## Creates a uniform buffer for backward shader metadata.
##
func _create_meta_buffer(input_size: int, output_size: int, num_vectors: int) -> RID:
    var meta_data: PackedByteArray = PackedByteArray()
    meta_data.resize(16)
    meta_data.encode_u32(0, input_size)
    meta_data.encode_u32(4, output_size)
    meta_data.encode_u32(8, num_vectors)
    return rd.uniform_buffer_create(meta_data.size(), meta_data)

##
## Builds a std430-compatible metadata buffer for forward shader.
##
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
    meta_bytes.resize((HEADER_UINTS + ARRAYS_PER_META * MAX_LAYERS) * 4)

    meta_bytes.encode_u32(0, layer_count)
    meta_bytes.encode_u32(4, batch_size)

    _encode_array_to_meta(meta_bytes, input_sizes, 0, MAX_LAYERS)
    _encode_array_to_meta(meta_bytes, output_sizes, 1, MAX_LAYERS)
    _encode_array_to_meta(meta_bytes, weight_offsets, 2, MAX_LAYERS)
    _encode_array_to_meta(meta_bytes, bias_offsets, 3, MAX_LAYERS)
    _encode_array_to_meta(meta_bytes, interm_offsets, 4, MAX_LAYERS)

    return meta_bytes

##
## Writes an integer array into a specific slot in the std430 layout.
##
static func _encode_array_to_meta(meta_bytes: PackedByteArray, arr: Array[int], slot_index: int, max_len: int) -> void:
    var base: int = 8
    for i: int in range(max_len):
        var value: int = arr[i] if i < arr.size() else 0
        var byte_pos: int = base + ((slot_index * max_len + i) * 4)
        meta_bytes.encode_u32(byte_pos, value)

##
## Creates a single RDUniform for a buffer.
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
func _create_uniform_set(uniforms: Array[RDUniform], shader: RID) -> RID:
    return rd.uniform_set_create(uniforms, shader, 0)

##
## Dispatches a compute shader with the given pipeline and uniform set.
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
