extends TextureRect
class_name ClassificationBoundary

# === Assigns network output bytes to this TextureRect ===
func set_texture_bytes(raw_bytes: PackedByteArray, texture_width: int, texture_height: int) -> void:
	# Why: Needs RGBA8 format so GPU shader output maps directly to displayable pixels
	var output_image: Image = Image.create_from_data(
		texture_width,
		texture_height,
		false,	# No mipmaps — avoids unwanted blur
		Image.FORMAT_RGBA8,
		raw_bytes
	)

	# Why: Convert raw Image to Texture2D so Godot can render it
	var output_texture: Texture2D = ImageTexture.create_from_image(output_image)

	# Why: Set the generated Texture2D to be displayed in the TextureRect
	texture = output_texture


# === Generates a full-resolution input grid for network prediction ===
func generate_full_plot_input_array(
	plot: PointPlot,
	plot_size: Vector2i,
	plot_scale: float = 1.0
) -> Array[PackedFloat32Array]:
	# Why: Sampling density is proportional to `plot_scale`
	var width: int = int(plot_size.x * plot_scale)
	var height: int = int(plot_size.y * plot_scale)

	var x_axis: PointPlot.AxisRange = plot._calculate_x_range()
	var y_axis: PointPlot.AxisRange = plot._calculate_y_range()

	var output: Array[PackedFloat32Array] = []

	# Why: Map each pixel (px, py) to its corresponding data-space coordinate
	for py: int in range(height):
		for px: int in range(width):
			var data_x: float = ((float(px) / width) * x_axis.range_size) + x_axis.min_value
			var data_y: float = ((1.0 - (float(py) / height)) * y_axis.range_size) + y_axis.min_value
			output.append(PackedFloat32Array([data_x, data_y]))

	return output
