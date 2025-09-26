extends BaseShaderRunner
class_name ColorMapRunner

# === Constants ===
const DEFAULT_SHADER_PATH: String = "res://examples/visual_training/shaders/color_map.spv"	# Default SPIR-V shader path

# === Internal State ===
var shader_rid: RID								# Compiled shader resource ID
var pipeline_rid: RID							# Compute pipeline resource ID

# === Constructor ===
func _init(shader_path: String = DEFAULT_SHADER_PATH) -> void:
	super()
	shader_rid = _load_shader(shader_path)		# Compile or load shader
	pipeline_rid = rd.compute_pipeline_create(shader_rid)	# Create compute pipeline from shader

# === Meta Buffer Builder ===
static func build_meta_buffer(texture_width: int, texture_height: int) -> PackedByteArray:
	# Storage buffer (std430) containing texture dimensions
	var meta_bytes: PackedByteArray = PackedByteArray()
	meta_bytes.resize(8)						# Two u32 values (4 bytes each)
	meta_bytes.encode_u32(0, texture_width)		# Encoded width at byte offset 0
	meta_bytes.encode_u32(4, texture_height)	# Encoded height at byte offset 4
	return meta_bytes

# === Storage Texture Creation ===
func _create_storage_texture(texture_width: int, texture_height: int) -> RID:
	# Define texture format
	var format: RDTextureFormat = RDTextureFormat.new()
	format.width = texture_width
	format.height = texture_height
	format.depth = 1
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM	# RGBA8 UNORM format
	
	# Allow compute shader storage, sampling, and CPU readback
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	
	format.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	
	var view: RDTextureView = RDTextureView.new()
	return rd.texture_create(format, view)

# === Dispatch Shader ===
func dispatch_color_map(
	predictions_buffer_rid: RID,
	texture_width: int,
	texture_height: int,
	colors: Array[Color],
	threads_per_group_x: int = 64,
	threads_per_group_y: int = 64
) -> RID:
	# Create output texture for computed colors
	var output_texture: RID = _create_storage_texture(texture_width, texture_height)
	
	# Meta buffer with texture dimensions
	var meta_buffer: RID = rd.storage_buffer_create(8, build_meta_buffer(texture_width, texture_height))

	var colors_array: PackedFloat32Array = _colors_to_rgba_array(colors)
	var colors_buffer: RID = create_uniform_buffer(colors_array)
	
	# Bind output texture to shader image slot
	var uniform_output_image: RDUniform = RDUniform.new()
	uniform_output_image.binding = 1										# Slot 1 -> image2D in shader
	uniform_output_image.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform_output_image.add_id(output_texture)


	# Create uniform set binding all required resources
	# Slot 0: predictions SSBO
	# Slot 1: output image2D
	# Slot 2: dimensions meta buffer
	var uniform_set_rid: RID = _create_uniform_set([
		_create_uniform(predictions_buffer_rid, 0),
		uniform_output_image,
		_create_uniform(meta_buffer, 2),
		_create_uniform(colors_buffer, 3, true)
	], shader_rid)
	
	# Run compute shader over 2D work groups
	_dispatch_compute_2d(
		pipeline_rid,
		uniform_set_rid,
		texture_width,
		texture_height,
		threads_per_group_x,
		threads_per_group_y
	)
	
	return output_texture

func _colors_to_rgba_array(colors: Array[Color]) -> PackedFloat32Array:
	var colors_array: PackedFloat32Array = PackedFloat32Array()
	for color: Color in colors:
		colors_array.append_array([color.r, color.g, color.b, color.a])
	return colors_array