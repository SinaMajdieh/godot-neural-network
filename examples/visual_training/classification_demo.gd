extends Control

# === Test Bench Settings ===
@export_category("Test Bench Settings")
@export var animation_duration: float = 1.5  # Delay for smooth visual updates

@export var data_points: int = 100:
	set(value):
		if value != data_points:
			data_points = value
			_update_generated_points()

@export var turns: float = 3.0:
	set(value):
		if value != turns:
			turns = value
			_update_generated_points()

@export var radius_growth: float = 0.1:
	set(value):
		if value != radius_growth:
			radius_growth = value
			_update_generated_points()

@export var noise: float = 0.5:
	set(value):
		if value != noise:
			noise = value
			_update_generated_points()

@export var class_a_color: Color = Color(0.2, 0.6, 1.0):
	set(value):
		class_a_color = value
		_update_generated_points()

@export var class_b_color: Color = Color(1.0, 0.4, 0.2):
	set(value):
		class_b_color = value
		_update_generated_points()

@export_range(0.1, 1.0, 0.1) var boundary_rendering_scale: float = 1.0
@export var start_training: bool = false: set = _start_training

@export var boundary_plot: Control
@export var loss_plot: Control
@export var lr_plot: Control
@export var accuracy_plot: Control

# === Network Parameters ===
@export_category("Network Parameters")
@export var layer_sizes: Array[int] = [2, 10, 10, 10, 1]
@export var hidden_activation: Activations.Type = Activations.Type.TANH
@export var output_activation: Activations.Type = Activations.Type.SIGMOID
@export var weight_init: NetworkLayer.WeightInitialization = NetworkLayer.WeightInitialization.XAVIER

# === Training Parameters ===
@export_category("Training Parameters")
@export var loss_function: Loss.Type = Loss.Type.BCE
@export var learning_rate: float = 0.07
@export var lambda_l2: float = 0.0
@export var epochs: int = 300
@export var batch_size: int = 10

@export_subgroup("Learning Rate Scheduler")
@export var learning_rate_schedular: LRSchedular.Type = LRSchedular.Type.COSINE
@export var schedular_params: Dictionary[String, Variant] = {
	ConfigKeys.LR_SCHEDULAR.STARTING_LR : 0.0,
	ConfigKeys.LR_SCHEDULAR.MIN_LR      : 0.0,
	ConfigKeys.LR_SCHEDULAR.EPOCHS      : 0,
	ConfigKeys.LR_SCHEDULAR.WARMPUP_EPOCHS : 0,
	ConfigKeys.LR_SCHEDULAR.DECAY_RATE  : 0.0,
	ConfigKeys.LR_SCHEDULAR.STEP_SIZE   : 0,
}

# === Internal State ===
var _runtime_ready: bool = false
var training_thread: Thread

# Graph panels
var plot_panel: PointPlotPanel
var loss_panel: EpochMetricGraphPanel
var lr_panel: EpochMetricGraphPanel
var accuracy_panel: EpochMetricGraphPanel

# Dataset & plotting buffers
var graph_size: Vector2
var points: Array[PackedFloat32Array] = []
var targets: Array[PackedFloat32Array] = []
var plot_inputs: Array[PackedFloat32Array]

# ML Components
var network: NeuralNetwork
var trainer: Trainer
var color_map_runner: ColorMapRunner


# === Node Lifecycle ===
func _ready() -> void:
	_runtime_ready = true
	_init_panels()
	_generate_spiral_points()

	await get_tree().process_frame
	plot_inputs = plot_panel.boundary.generate_full_plot_input_array(
		plot_panel.graph_node,
		plot_panel.graph_node.size,
		boundary_rendering_scale
	)
	graph_size = plot_panel.graph_node.size
	_start_training(start_training)


# === UI Panel Initialization ===
func _init_panels() -> void:
	plot_panel = PointPlotPanel.new_panel("Two‑Class Scatter", data_points, 4.0)
	boundary_plot.add_child(plot_panel)

	loss_panel = EpochMetricGraphPanel.new_panel()
	loss_plot.add_child(loss_panel)

	lr_panel = EpochMetricGraphPanel.new_panel("Learning rate over epochs", "Learning rate", 200, Color("98C379"))
	lr_plot.add_child(lr_panel)

	accuracy_panel = EpochMetricGraphPanel.new_panel("Accuracy over epochs", "Accuracy", 200, Color("E5C07B"))
	accuracy_plot.add_child(accuracy_panel)


# === Setters ===
func _update_generated_points() -> void:
	if _runtime_ready:
		_generate_spiral_points()

func _start_training(value: bool) -> void:
	start_training = value
	if not _runtime_ready:
		return
	if start_training and (training_thread == null or not training_thread.is_started()):
		training_thread = Thread.new()
		training_thread.start(_run_training)


# === Data Generation ===
func _generate_spiral_points() -> void:
	points.clear()
	targets.clear()

	for i: int in range(data_points):
		var angle: float = (float(i) / data_points) * turns * TAU
		var radius: float = (radius_growth * angle) + randf_range(-noise, noise)

		if i % 2 == 0:
			_add_point(angle, radius, class_a_color, 0.0)
		else:
			_add_point(angle + PI, radius, class_b_color, 1.0)

func _add_point(angle: float, radius: float, color: Color, target_value: float) -> void:
	var x: float = cos(angle) * radius
	var y: float = sin(angle) * radius
	points.append(PackedFloat32Array([x, y]))
	targets.append(PackedFloat32Array([target_value]))
	plot_panel.add_point(Vector2(x, y), color)


# === Training ===
func _run_training() -> void:
	color_map_runner = ColorMapRunner.new()

	network = NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES : layer_sizes,
		ConfigKeys.NETWORK.HIDDEN_ACT  : hidden_activation,
		ConfigKeys.NETWORK.OUTPUT_ACT  : output_activation,
		ConfigKeys.NETWORK.WEIGHT_INIT : weight_init
	})

	trainer = Trainer.new({
		ConfigKeys.TRAINER.NETWORK       : network,
		ConfigKeys.TRAINER.LOSS          : loss_function,
		ConfigKeys.TRAINER.LEARNING_RATE : learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2     : lambda_l2,
		ConfigKeys.TRAINER.EPOCHS        : epochs,
		ConfigKeys.TRAINER.BATCH_SIZE    : batch_size
	})
	trainer.lr_schedular = LRSchedular.new(learning_rate_schedular, schedular_params)
	trainer.epoch_finished.connect(_on_epoch_finished)

	print("Training time: %d ms" % _train_and_measure(points, targets))
	call_deferred("_on_training_complete")

func _train_and_measure(
	inputs: Array[PackedFloat32Array],
	targets_in: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(inputs, targets_in)
	return Time.get_ticks_msec() - start_time

func _on_training_complete() -> void:
	training_thread.wait_to_finish()


# === Epoch Callback ===
func _on_epoch_finished(loss: float, epoch: int) -> void:
	plot_panel.call_deferred("update_epoch_info", loss, epoch)
	loss_panel.call_deferred("add_metric", loss, epoch)
	lr_panel.call_deferred("add_metric", trainer.learning_rate, epoch)

	var accuracy: float = ModelEvaluator.evaluate_model(network, points, targets)
	accuracy_panel.call_deferred("add_metric", accuracy * 100.0, epoch)

	_render_boundary()


# === Decision Boundary Rendering ===
func _render_boundary() -> void:
	var start_time: int = Time.get_ticks_msec()
	var predictions: PackedFloat32Array = network.forward_pass(plot_inputs)
	var duration_ms: int = Time.get_ticks_msec() - start_time

	print_rich("[color=yellow]Forward pass time: %d ms" % duration_ms)
	plot_panel.call_deferred("update_decision_boundary_label", duration_ms)

	var predictions_buffer_rid: RID = color_map_runner.create_buffer(predictions)
	var output_texture_rid: RID = color_map_runner.dispatch_color_map(
		predictions_buffer_rid,
		int(graph_size.x * boundary_rendering_scale),
		int(graph_size.y * boundary_rendering_scale),
		[
			Color(class_a_color.r, class_a_color.g, class_a_color.b, class_a_color.a * 0.1),
			Color(class_b_color.r, class_b_color.g, class_b_color.b, class_b_color.a * 0.1)
		],
		8, 8
	)

	var raw_bytes: PackedByteArray = color_map_runner.rd.texture_get_data(output_texture_rid, 0)
	plot_panel.boundary.call_deferred(
		"set_texture_bytes",
		raw_bytes,
		int(graph_size.x * boundary_rendering_scale),
		int(graph_size.y * boundary_rendering_scale)
	)
