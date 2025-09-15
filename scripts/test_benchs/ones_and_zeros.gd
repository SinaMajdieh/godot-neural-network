## TrainingBench.gd
## Node-based test bench for training and evaluating a GPU-accelerated neural network.
## Loads image data, initializes network, runs training loop, and prints evaluation metrics.

extends Node

## Thread for asynchronous training.
var training_thread: Thread

## === Exported Parameters ===

@export_category("Test Bench Attributes")
@export_global_dir var zeros_dir: String
@export_global_dir var ones_dir: String
@export_range(0.1, 1.0) var image_scale: float = 0.25

@export_category("Input")
@export_range(0.0, 1.0) var input_size_ratio: float = 1.0

@export_category("Network Properties")
@export var layers: Array[int] = [32 * 32, 32, 16, 1]

@export_category("Training Properties")
@export_range(0.000001, 10) var learning_rate: float = 0.6
@export_range(1, 1000) var epochs: int = 400
@export_range(1, 1_000_000) var batch_size: int = 1024
@export_range(0.0, 1.0) var test_size: float = 0.2
@export var hidden_layers_activation: Activations.Type = Activations.Type.TANH
@export var output_layer_activation: Activations.Type = Activations.Type.SIGMOID

## === Internal Data Class ===

class Data:
	var inputs: Array[PackedFloat32Array]
	var targets: Array[PackedFloat32Array]

	func _init(inputs_: Array[PackedFloat32Array], targets_: Array[PackedFloat32Array]) -> void:
		inputs = inputs_
		targets = targets_

## Reads and combines ones/zeros image data, applies slicing and splitting.
func read_data() -> DataSplit:
	var result: DataSplit = DataSplit.new()
	var ones: Data = read_ones()
	var zeros: Data = read_zeros()

	var usable_size: int = int((zeros.inputs.size() + ones.inputs.size()) * input_size_ratio / 2)
	ones.inputs = ones.inputs.slice(0, usable_size)
	ones.targets = ones.targets.slice(0, usable_size)
	zeros.inputs = zeros.inputs.slice(0, usable_size)
	zeros.targets = zeros.targets.slice(0, usable_size)

	var split_0: DataSplit = DataSetUtils.train_test_split(zeros.inputs, zeros.targets, test_size)
	var split_1: DataSplit = DataSetUtils.train_test_split(ones.inputs, ones.targets, test_size)

	result.train_inputs.append_array(split_0.train_inputs)
	result.train_inputs.append_array(split_1.train_inputs)
	result.train_targets.append_array(split_0.train_targets)
	result.train_targets.append_array(split_1.train_targets)
	result.test_inputs.append_array(split_0.test_inputs)
	result.test_inputs.append_array(split_1.test_inputs)
	result.test_targets.append_array(split_0.test_targets)
	result.test_targets.append_array(split_1.test_targets)

	return result

## Reads images labeled "1" and returns input/target pairs.
func read_ones() -> Data:
	var ones: Array[PackedFloat32Array] = []
	var targets: Array[PackedFloat32Array] = []
	var ones_paths: PackedStringArray = list_files(ones_dir)
	for path in ones_paths:
		ones.append(read_image(path))
		targets.append(PackedFloat32Array([1.0]))
	return Data.new(ones, targets)

## Reads images labeled "0" and returns input/target pairs.
func read_zeros() -> Data:
	var zeros: Array[PackedFloat32Array] = []
	var targets: Array[PackedFloat32Array] = []
	var zeros_paths: PackedStringArray = list_files(zeros_dir)
	for path in zeros_paths:
		zeros.append(read_image(path))
		targets.append(PackedFloat32Array([0.0]))
	return Data.new(zeros, targets)

## Loads and preprocesses an image into a grayscale input vector.
func read_image(path: String) -> PackedFloat32Array:
	var result: PackedFloat32Array = PackedFloat32Array()
	var image: Image = Image.new()
	var error: int = image.load(path)

	if error != OK:
		push_error("Failed to load image: %s" % path)
		return result

	var width: int = int(image.get_width() * image_scale)
	var height: int = int(image.get_height() * image_scale)
	image.resize(width, height, Image.INTERPOLATE_LANCZOS)

	for y in range(height):
		for x in range(width):
			var color: Color = image.get_pixel(x, y)
			var value: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			if value > 0.95:
				value = 0.5
			result.append(clamp((value - 0.5) * 2.0, 0.0, 1.0))
	return result

## Lists all non-directory files in the given path.
func list_files(path: String) -> PackedStringArray:
	var list: PackedStringArray = PackedStringArray()
	var dir: DirAccess = DirAccess.open(path)
	if dir == null:
		push_error("Failed to open directory: %s" % path)
		return list

	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if not dir.current_is_dir():
			list.append(path + "/" + file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	return list

## Entry point: starts training in a separate thread.
func _ready() -> void:
	training_thread = Thread.new()
	training_thread.start(_run_training)

## Main training routine.
func _run_training() -> void:
	var runner: ShaderRunner = _create_shader_runner()
	var network: NeuralNetwork = _create_network(runner)
	var split: DataSplit = _load_and_split_data()

	var trainer: Trainer = Trainer.new(network, runner, Loss.Type.BCE)
	var training_time: int = _run_training_loop(trainer, split.train_inputs, split.train_targets)

	print("Training time: %d ms" % training_time)
	_evaluate_model(network, split.test_inputs, split.test_targets)
	call_deferred("_on_training_complete")

## Creates and returns a ShaderRunner instance.
func _create_shader_runner() -> ShaderRunner:
	return ShaderRunner.new(
		"res://scripts/neural_network/gpu/shaders/forward_pass.spv",
		"res://scripts/neural_network/gpu/shaders/backward_pass.spv"
	)

## Initializes the neural network with configured layer sizes and activations.
func _create_network(shader_runner: ShaderRunner) -> NeuralNetwork:
	return NeuralNetwork.new(layers, shader_runner, hidden_layers_activation, output_layer_activation)

## Loads and splits image data into training and testing sets.
func _load_and_split_data() -> DataSplit:
	return read_data()

## Runs the training loop and returns elapsed time in milliseconds.
func _run_training_loop(
	trainer: Trainer,
	train_inputs: Array[PackedFloat32Array],
	train_targets: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(train_inputs, train_targets, learning_rate, epochs, batch_size)
	var end_time: int = Time.get_ticks_msec()
	return end_time - start_time

## Called after training completes to clean up the training thread.
func _on_training_complete() -> void:
	training_thread.wait_to_finish()

## Evaluates model accuracy on test data and prints results.
func _evaluate_model(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> void:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(predictions_flat, 1)

	var correct: int = 0
	for i: int in range(predictions.size()):
		var pred: float = predictions[i][0]
		var target: float = targets[i][0]
		var binary_pred: int = int(pred >= 0.5)
		var color: String = "[color=red]"
		if binary_pred == int(target):
			color = "[color=green]"
			correct += 1
		print_rich("%sTest case %3d : prediction-class [%5.3f]-[%1d] target [%1d]" % [color, i, pred, binary_pred, target])

	var accuracy: float = float(correct) / float(predictions.size())
	print_rich("[color=cyan]Test Accuracy: %4.2f" % accuracy)
