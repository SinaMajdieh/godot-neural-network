class_name ImageUtils
extends RefCounted

# Handles image loading, preprocessing, and reconstruction.
# Why: Keeps image operations isolated so model logic stays independent.

# Loads and preprocesses an image to grayscale input vector in [-1.0, 1.0].
# Params:
#   path (String): Path to image file.
#   scale (float): Scale factor to resize image.
# Returns:
#   PackedFloat32Array: Flattened grayscale pixel values.
static func read_image(path: String, scale: float, invert: bool = false) -> PackedFloat32Array:
	var result: PackedFloat32Array = []
	var image: Image = Image.new()
	if image.load(path) != OK:
		push_error("Failed to load image: %s" % path)
		return result

	image.convert(Image.FORMAT_RGBA8)
	var width: int = int(image.get_width() * scale)
	var height: int = int(image.get_height() * scale)
	image.resize(width, height, Image.INTERPOLATE_LANCZOS)

	for y: int in range(height):
		for x: int in range(width):
			var color: Color = image.get_pixel(x, y)
			var value: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			var scaled_value: float = (value - 0.5) * 2.0
			if invert:
				result.append(1.0 - scaled_value)
			else:
				result.append(scaled_value)
	return result


# Loads and preprocesses all images in a directory.
# Params:
#   path (String): Directory path containing images.
#   scale (float): Scale factor for each image.
# Returns:
#   Array[PackedFloat32Array]: List of processed image data.
static func read_images(path: String, scale: float, invert: bool = false) -> Array[PackedFloat32Array]:
	var results: Array[PackedFloat32Array] = []
	for file: String in FileUtils.list_files(path):
		var img_data: PackedFloat32Array = read_image(file, scale, invert) 
		results.append(img_data)
	return results


# Converts grayscale float data to an Image.
# Params:
#   gray_floats (PackedFloat32Array): Flattened pixel intensities.
#   width (int): Output image width.
#   height (int): Output image height.
# Returns:
#   Image: Output grayscale image with alpha.
static func image_from_f32_array(
	gray_floats: PackedFloat32Array,
	width: int,
	height: int
) -> Image:
	var img: Image = Image.create(width, height, false, Image.FORMAT_RGBA8)
	for y: int in range(height):
		for x: int in range(width):
			var idx: int = y * width + x
			var v: float = clamp(gray_floats[idx] / 2.0 + 0.5, 0.0, 1.0)
			img.set_pixel(x, y, Color(v, v, v, 1.0))
	return img

static func invert_image(data: PackedFloat32Array) -> void:
	for i: int in range(data.size()):
		data[i] = 1.0 - data[i]
