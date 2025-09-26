#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// === Buffers and Images ===

// Prediction values per pixel (float array)
layout(std430, set = 0, binding = 0) buffer PredictionBuffer {
	float prediction_values[];
};

// Output image storage (write-only RGBA8)
layout(rgba8, set = 0, binding = 1) uniform writeonly image2D output_image;

// Texture dimensions buffer for safe indexing
layout(std430, set = 0, binding = 2) buffer TextureDimensions {
	uint texture_width;
	uint texture_height;
};

// Colors for classes: 0 and 1
layout(set = 0, binding = 3) uniform ClassColorMap {
	vec4 class_colors[2];
};

void main() {
	uint px_x = gl_GlobalInvocationID.x;
	uint px_y = gl_GlobalInvocationID.y;

	// === Check bounds to avoid out-of-range
	if (px_x >= texture_width || px_y >= texture_height)
		return;

	// === Calculate linear index in predictions array
	uint linear_idx = px_y * texture_width + px_x;
	float pred_value = prediction_values[linear_idx];

	// === Choose color based on prediction threshold
	vec4 out_color = (pred_value < 0.5) ? class_colors[0] : class_colors[1];

	// === Store resulting color to output image at current pixel
	imageStore(output_image, ivec2(int(px_x), int(px_y)), out_color);
}
