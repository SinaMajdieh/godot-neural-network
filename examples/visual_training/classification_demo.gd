extends Control

# === Test Bench Settings ===
@export_category("Test Bench Settings")
@export var animation_duration: float = 1.5
@export var data_points: int = 100
@export var turns: float = 3.0
@export var radius_growth: float = 0.1
@export var noise: float = 0.5
@export var class_a_color: Color = Color(0.2, 0.6, 1.0)	# Blue
@export var class_b_color: Color = Color(1.0, 0.4, 0.2)	# Orange

# === Network Parameters ===
@export_category("Network Parameters")
@export var layer_sizes: Array[int] = [2, 10, 10, 10, 1]	# Nodes per layer
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

# === Internal State ===
var training_thread: Thread
var plot_panel: PointPlotPanel
var graph_size: Vector2
var points: Array[PackedFloat32Array] = []
var targets: Array[PackedFloat32Array] = []
var plot_inputs: Array[PackedFloat32Array]

# === ML Components ===
var network: NeuralNetwork
var trainer: Trainer
var color_map_runner: ColorMapRunner

func _ready() -> void:
	# === Create visualization panel ===
	plot_panel = PointPlotPanel.new_panel("Two-Class Scatter", data_points, 4.0)
	add_child(plot_panel)
	
	# === Generate training points ===
	_generate_spiral_points()

	# === Prepare prediction input grid ===
	await get_tree().process_frame
	plot_inputs = plot_panel.boundary.generate_full_plot_input_array(
		plot_panel.graph_node,
		plot_panel.graph_node.size,
		1
	)
	graph_size = plot_panel.graph_node.size
	
	# === Launch training async ===
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
			# Class A spiral (clockwise)
			var x: float = cos(angle) * radius
			var y: float = sin(angle) * radius
			points.append(PackedFloat32Array([x, y]))
			targets.append(PackedFloat32Array([0.0]))
			plot_panel.add_point(Vector2(x, y), class_a_color)
		else:
			# Class B spiral (counter-clockwise)
			var angle_b: float = angle + PI
			var x_b: float = cos(angle_b) * radius
			var y_b: float = sin(angle_b) * radius
			points.append(PackedFloat32Array([x_b, y_b]))
			targets.append(PackedFloat32Array([1.0]))
			plot_panel.add_point(Vector2(x_b, y_b), class_b_color)


# === Training Thread Entry Point ===
func _run_training() -> void:
	color_map_runner = ColorMapRunner.new()

	# === Create and configure network ===
	network = NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES  : layer_sizes,
		ConfigKeys.NETWORK.HIDDEN_ACT   : hidden_activation,
		ConfigKeys.NETWORK.OUTPUT_ACT   : output_activation,
		ConfigKeys.NETWORK.WEIGHT_INIT  : weight_init
	})

	# === Create and configure trainer ===
	trainer = Trainer.new({
		ConfigKeys.TRAINER.NETWORK       : network,
		ConfigKeys.TRAINER.LOSS          : loss_function,
		ConfigKeys.TRAINER.LEARNING_RATE : learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2     : lambda_l2,
		ConfigKeys.TRAINER.EPOCHS        : epochs,
		ConfigKeys.TRAINER.BATCH_SIZE    : batch_size
	})
	trainer.epoch_finished.connect(_on_epoch_finished)

	var training_time: int = _train_and_measure(points, targets)
	print("Training time: %d ms" % training_time)

	call_deferred("_on_training_complete")


# === Helper to run training and measure ===
func _train_and_measure(
	train_inputs: Array[PackedFloat32Array],
	train_targets: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(train_inputs, train_targets)
	return Time.get_ticks_msec() - start_time


# === Post-training cleanup ===
func _on_training_complete() -> void:
	training_thread.wait_to_finish()


# === Epoch callback ===
func _on_epoch_finished(_loss: float, _epoch: int) -> void:
	if _epoch == epochs - 1:
		_render_boundary()


# === GPU Boundary Rendering ===
func _render_boundary() -> void:
	var start_time: int = Time.get_ticks_msec()
	var predictions: PackedFloat32Array = network.forward_pass(plot_inputs)
	print("Forward pass time: %d ms" % [Time.get_ticks_msec() - start_time])

	# === Create GPU prediction buffer ===
	var predictions_buffer_rid: RID = color_map_runner.create_buffer(predictions)

	# === GPU shader dispatch ===
	var output_texture_rid: RID = color_map_runner.dispatch_color_map(
		predictions_buffer_rid,
		int(graph_size.x),
		int(graph_size.y),
		[
			Color(class_a_color.r, class_a_color.g, class_a_color.b, class_a_color.a * 0.1),
			Color(class_b_color.r, class_b_color.g, class_b_color.b, class_b_color.a * 0.1)
		],
		8, 8
	)

	# === Read and update UI texture ===
	var raw_bytes: PackedByteArray = color_map_runner.rd.texture_get_data(output_texture_rid, 0)
	plot_panel.boundary.call_deferred(
		"set_texture_bytes",
		raw_bytes,
		int(graph_size.x),
		int(graph_size.y)
	)
