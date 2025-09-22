# TrainNetwork.gd
# Handles end-to-end training workflow for digit recognition.
# Why: Encapsulates dataset prep, network setup, training, and export.

extends Control

# Thread for asynchronous training.
var training_thread: Thread

@export_category("Data")
@export_global_dir var training_data_dir: String
@export_range(0.0, 1.0) var image_scale: float = 0.25
@export_range(0.0, 1.0) var input_size_ratio: float = 1.0
@export var category_size: int = 900

@export_category("Network Properties")
@export var layer_sizes: Array[int] = [32 * 32, 32, 16, 1]
@export_file("*.tres") var export_path: String = "res://scripts/test_benchs/trained_neural_networks/digit_recognition.tres"
@export var export_network: bool = false

@export_category("Training Properties")
@export_range(0.000001, 10) var learning_rate: float = 0.6
@export_range(0.000001, 1) var lambda_l2: float = 0.0001
@export var loss: Loss.Type = Loss.Type.BCE
@export_range(1, 1000) var epochs: int = 400
@export_range(1, 1_000_000) var batch_size: int = 1024
@export_range(0.0, 1.0) var test_size_ratio: float = 0.2
@export var weight_initialization: NetworkLayer.WeightInitialization = \
	NetworkLayer.WeightInitialization.XAVIER
@export var hidden_layers_activation: Activations.Type = \
	Activations.Type.TANH
@export var output_layer_activation: Activations.Type = \
	Activations.Type.SIGMOID

# Holds per-category inputs and targets before concatenation.
var training_inputs_dic: Dictionary[int, Array] = {}
var training_targets_dic: Dictionary[int, Array] = {}

# Final concatenated training sets.
var training_inputs: Array[PackedFloat32Array] = []
var training_targets: Array[PackedFloat32Array] = []

# Called on scene load to prepare data and start training.
func _ready() -> void:
	_init_empty_datasets()
	_process_inputs_targets()
	show_input_as_image(training_inputs[0])
	training_thread = Thread.new()
	training_thread.start(_run_training)

# Initializes dictionaries for data storage.
func _init_empty_datasets() -> void:
	for i: int in range(10):
		training_inputs_dic[i] = []
		training_targets_dic[i] = []


# Prepares input and target sets from folder data.
func _process_inputs_targets() -> void:
	training_inputs_dic = training_data_to_dic(training_data_dir)
	training_targets_dic = generate_targets(training_inputs_dic)

	training_inputs = concatenate_data(
		training_inputs_dic,
		int(category_size * input_size_ratio)
	)
	training_targets = concatenate_data(
		training_targets_dic,
		int(category_size * input_size_ratio)
	)


# Renders a given input vector as a 32×32 image in the UI.
func show_input_as_image(data: PackedFloat32Array) -> void:
	var image: Image = ImageUtils.image_from_f32_array(data, 32, 32)
	var texture: Texture = ImageTexture.create_from_image(image)
	$TextureRect.texture = texture


# Main training routine, executed in a background thread.
func _run_training() -> void:
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(ConfigKeys.SHADERS_PATHS.FORWARD_PASS)
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new(ConfigKeys.SHADERS_PATHS.BACKWARD_PASS)
	var network: NeuralNetwork = _create_network(forward_runner)
	var split: DataSplit = DataSetUtils.train_test_split(
		training_inputs,
		training_targets,
		test_size_ratio
	)

	var trainer: Trainer = Trainer.new({
		ConfigKeys.TRAINER.NETWORK: network,
		ConfigKeys.TRAINER.RUNNER: backward_runner,
		ConfigKeys.TRAINER.LOSS: loss,
		ConfigKeys.TRAINER.LEARNING_RATE: learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2: lambda_l2,
		ConfigKeys.TRAINER.EPOCHS: epochs,
		ConfigKeys.TRAINER.BATCH_SIZE: batch_size,
		ConfigKeys.TRAINER.GRADIENT_CLIP_TRESHOLD: 1
	})

	var satisfied: bool = false
	while not satisfied:
		satisfied = true
		var training_time: int = _run_training_loop(
			trainer,
			split.train_inputs,
			split.train_targets
		)

		print("Training time: %d ms" % training_time)

		var test_acc: float = ModelEvaluator.evaluate_model_soft_max(
			network,
			split.test_inputs,
			split.test_targets,
			false
		)
		print_rich("[color=cyan]Test Accuracy: %4.2f" % test_acc)

		var training_acc: float = ModelEvaluator.evaluate_model_soft_max(
			network,
			split.train_inputs,
			split.train_targets
		)
		print_rich("[color=cyan]Training Accuracy: %4.2f" % training_acc)

		if test_acc >= 0.9:
			satisfied = true

	if export_network:
		NeuralNetworkSerializer.export(network, export_path)

	call_deferred("_on_training_complete")

# Initializes the neural network with set layers and activations.
func _create_network(shader_runner: ForwardPassRunner) -> NeuralNetwork:
	return NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: shader_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: hidden_layers_activation,
		ConfigKeys.NETWORK.OUTPUT_ACT: output_layer_activation,
		ConfigKeys.NETWORK.WEIGHT_INIT: weight_initialization
	})


# Runs training and returns elapsed time in milliseconds.
func _run_training_loop(
	trainer: Trainer,
	train_inputs: Array[PackedFloat32Array],
	train_targets: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(train_inputs, train_targets)
	var end_time: int = Time.get_ticks_msec()
	return end_time - start_time


# Cleans up training thread after completion.
func _on_training_complete() -> void:
	training_thread.wait_to_finish()


# Merges per-category arrays into a single dataset.
func concatenate_data(
	data: Dictionary[int, Array],
	data_size: int
) -> Array[PackedFloat32Array]:
	var result: Array[PackedFloat32Array] = []
	for i: int in range(data_size):
		for key: int in data:
			result.append(data[key][i])
	return result


# Reads training data from folder structure into dictionary form.
func training_data_to_dic(path: String) -> Dictionary[int, Array]:
	var results: Dictionary[int, Array] = {}
	var dirs: PackedStringArray = FileUtils.list_dirs(path)

	for i: int in range(dirs.size()):
		results[i] = ImageUtils.read_images(dirs[i], image_scale)
		var usable_size: int = int(results[i].size() * input_size_ratio)
		results[i].resize(usable_size)

	return results


# Creates a one-hot encoded vector for a given index.
func generate_one_hot(hot_idx: int, length: int) -> PackedFloat32Array:
	var one_hot: PackedFloat32Array = []
	one_hot.resize(length)
	one_hot.fill(0.0)
	one_hot[hot_idx] = 1.0
	return one_hot


# Generates repeated one-hot encoded vectors.
func generate_one_hots(
	hot_idx: int,
	length: int,
	num: int
) -> Array[PackedFloat32Array]:
	var results: Array[PackedFloat32Array] = []
	for i: int in range(num):
		results.append(generate_one_hot(hot_idx, length))
	return results


# Creates per-category one-hot targets for given inputs.
func generate_targets(inputs: Dictionary[int, Array]) -> Dictionary[int, Array]:
	var results: Dictionary[int, Array] = {}
	for key: int in inputs:
		results[key] = generate_one_hots(
			key, inputs.size(), inputs[key].size()
		)
	return results
