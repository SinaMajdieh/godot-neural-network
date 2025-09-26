extends TextureRect
class_name ClassificationBoundary

func set_texture_bytes(raw_bytes: PackedByteArray, texture_width: int, texture_height: int) -> void:
    # === Create Image from raw RGBA8 bytes ===
    var output_image: Image = Image.create_from_data(
        texture_width,
        texture_height,
        false,								# No mipmaps
        Image.FORMAT_RGBA8,
        raw_bytes
    )
    

    # === Convert Image to Texture2D for display ===
    var output_texture: Texture2D = ImageTexture.create_from_image(output_image)

    # === Assign generated texture to TextureRect ===
    texture = output_texture

## Generates a full network input array where each pixel's coordinates are stored
## as a PackedFloat32Array [x, y]
## Generates a full network input array with sampling density proportional to `scale`
## Each pixel's coordinates are stored as a PackedFloat32Array [x, y]
func generate_full_plot_input_array(
	plot: PointPlot,
	plot_size: Vector2i,
	sampling_scale: float = 1.0
) -> Array[PackedFloat32Array]:
    var step_x: int = max(1, int(1.0 / sampling_scale))
    var step_y: int = max(1, int(1.0 / sampling_scale))

    var x_axis: PointPlot.AxisRange = plot._calculate_x_range()
    var y_axis: PointPlot.AxisRange = plot._calculate_y_range()

    var output: Array[PackedFloat32Array] = []

    for py: int in range(0, plot_size.y, step_y):
        for px: int in range(0, plot_size.x, step_x):
            var data_x: float = ((float(px) / float(plot_size.x)) * x_axis.range_size) + x_axis.min_value
            var data_y: float = ((1.0 - (float(py) / float(plot_size.y))) * y_axis.range_size) + y_axis.min_value
            output.append(PackedFloat32Array([data_x, data_y]))

    return output

