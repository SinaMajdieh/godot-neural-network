extends Node

##
## Test bench for verifying:
## 1. grad_clip_norm.comp computes correct squared L2 norm on GPU.
## 2. grad_clip_scale.comp scales gradients in-place correctly without CPU readbacks.
##

@export_file var NORM_SHADER_PATH: String = "res://scripts/neural_network/gpu/shaders/grad_clip_norm.spv"
@export_file var SCALE_SHADER_PATH: String = "res://scripts/neural_network/gpu/shaders/grad_clip_scale.spv"

func _ready() -> void:
	# --- Test data ----------------------------------------------------------
	var gradients: PackedFloat32Array = PackedFloat32Array([1.0, -2.0, 3.0, -4.0])
	var clip_threshold: float = 3.5
	var grad_count: int = gradients.size()

	# --- CPU reference ------------------------------------------------------
	var cpu_sum_sq: float = 0.0
	for g: float in gradients:
		cpu_sum_sq += g * g
	print("CPU norm²: ", cpu_sum_sq)

	var cpu_scaled: PackedFloat32Array = gradients.duplicate()
	if cpu_sum_sq > clip_threshold * clip_threshold and cpu_sum_sq != 0.0:
		var scale: float = clip_threshold / sqrt(cpu_sum_sq)
		for i: int in range(cpu_scaled.size()):
			cpu_scaled[i] *= scale
	print("CPU clipped gradients: ", cpu_scaled)

	# --- GPU norm calculation and scaling -----------------------------------
	var clip_runner: GradClipRunner = GradClipRunner.new(NORM_SHADER_PATH, SCALE_SHADER_PATH)
	var grad_buf: RID = clip_runner.create_buffer(gradients)
	clip_runner.clip_gradients(grad_buf, grad_count, clip_threshold)

	var gpu_scaled: PackedFloat32Array = TensorUtils.bytes_to_floats(
		clip_runner.get_buffer_data(grad_buf)
	)

	print("GPU clipped gradients: ", gpu_scaled)

	# --- Check correctness ---------------------------------------------------
	var scale_pass: bool = true
	for i: int in range(cpu_scaled.size()):
		if not is_equal_approx(cpu_scaled[i], gpu_scaled[i]):
			scale_pass = false
			break

	if  scale_pass:
		print_rich("[color=green][PASS] Scale shaders match CPU reference.")
	else:
		print_rich("[color=red][FAIL] Scaling shader mismatch.")

	# --- Cleanup -------------------------------------------------------------
	clip_runner.rd.free_rid(grad_buf)
