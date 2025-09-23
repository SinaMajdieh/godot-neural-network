class_name GradientClipOptimizer
extends RefCounted

var clip_threshold: float
var clip_enabled: bool
var clip_runner: GradClipRunner

func _init(
	clip_enabled_: bool = false,
	clip_threshold_: float = 1.0,
	clip_runner_: GradClipRunner = GradClipRunner.new()
) -> void:
	clip_enabled = clip_enabled_
	clip_threshold = clip_threshold_
	clip_runner = clip_runner_

func run(
	layer: NetworkLayer,
	w_grad: PackedFloat32Array,
	b_grad: PackedFloat32Array
) -> Dictionary:
	var w_final: PackedFloat32Array = w_grad
	var b_final: PackedFloat32Array = b_grad

	# Only apply clipping if enabled
	if clip_enabled:
		var combined: PackedFloat32Array = PackedFloat32Array(w_grad)
		combined.append_array(b_grad)
		var buf: RID = clip_runner.create_buffer(combined)
		clip_runner.clip_gradients(
			buf, combined.size(), clip_threshold
		)
		var clipped: PackedFloat32Array = TensorUtils.bytes_to_floats(
			clip_runner.get_buffer_data(buf)
		)
		var w_size: int = layer.input_size * layer.output_size
		w_final = clipped.slice(0, w_size)
		b_final = clipped.slice(w_size, combined.size())

	return {"weight": w_final, "bias": b_final}
