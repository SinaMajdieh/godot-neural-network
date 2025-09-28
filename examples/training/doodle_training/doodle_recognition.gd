extends Control
class_name TrainDoodle

## Handles end-to-end training workflow for digit recognition.
## WHY: Encapsulates dataset preparation, network setup, training,
##      evaluation, and optional export.

# -------------------------------------------------------------------
# Threads
# -------------------------------------------------------------------
var training_thread: Thread

# -------------------------------------------------------------------
# Data Properties
# -------------------------------------------------------------------
@export_category("Data")
@export_global_dir var training_data_dir: String
@export_range(0.0, 1.0) var image_scale: float = 64.0 / 254.0
@export_range(0.0, 1.0) var input_size_ratio: float = 1.0
@export var reference: Dictionary[int, String] = {0: "cat", 1: "guitar"}
@export var category_size: int = 900

# -------------------------------------------------------------------
# Network Properties
# -------------------------------------------------------------------
@export_category("Network Properties")
@export var layer_sizes: Array[int] = [64 * 64, 32, 16, 1]
@export_file("*.tres") var export_path: String = "res://scripts/test_benchs/trained_neural_networks/digit_recognition.tres"
@export var export_network: bool = false

# -------------------------------------------------------------------
# Training Properties
# -------------------------------------------------------------------
@export_category("Training Properties")
@export_range(0.000001, 10) var learning_rate: float = 0.6
@export_range(0.000001, 1) var lambda_l2: float = 0.0001
@export var loss: Loss.Type = Loss.Type.BCE
@export_range(1, 1000) var epochs: int = 400
@export_range(1, 1_000_000) var batch_size: int = 1024
@export_range(0.0, 1.0) var test_size_ratio: float = 0.2
@export var weight_initialization: NetworkLayer.WeightInitialization = NetworkLayer.WeightInitialization.XAVIER
@export var hidden_layers_activation: Activations.Type = Activations.Type.TANH
@export var output_layer_activation: Activations.Type = Activations.Type.SIGMOID

# -------------------------------------------------------------------
# Optimizer Properties
# -------------------------------------------------------------------
@export_category("Optimizers")
@export_subgroup("Gradient Clipping")
@export var gradient_clip_enable: bool = false
@export var gradient_clip_threshold: float = 1.0

# -------------------------------------------------------------------
# Training data storage
# -------------------------------------------------------------------
var training_inputs_dic: Dictionary[int, Array] = {}
var training_targets_dic: Dictionary[int, Array] = {}
var training_inputs: Array[PackedFloat32Array] = []
var training_targets: Array[PackedFloat32Array] = []

var loss_panel: EpochMetricGraphPanel

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------

func _ready() -> void:
	_init_empty_datasets()
	_process_inputs_targets()

	loss_panel = EpochMetricGraphPanel.new_panel()
	add_child(loss_panel)
	print("Number of images: %d" % training_inputs.size())

	training_thread = Thread.new()
	training_thread.start(_run_training)

# -------------------------------------------------------------------
# Dataset Preparation
# -------------------------------------------------------------------

## Initialize empty dataset dictionaries
func _init_empty_datasets() -> void:
	for i: int in range(reference.size()):
		training_inputs_dic[i] = []
		training_targets_dic[i] = []

## Load and process dataset into final training arrays
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

# -------------------------------------------------------------------
# Utility Visualization
# -------------------------------------------------------------------

## Shows an input vector as a 32×32 image in the UI  
func show_input_as_image(data: PackedFloat32Array) -> void:
	var image: Image = ImageUtils.image_from_f32_array(data, 32, 32)
	var texture: Texture = ImageTexture.create_from_image(image)
	$TextureRect.texture = texture

# -------------------------------------------------------------------
# Training Execution
# -------------------------------------------------------------------

## Main training routine (background thread)
func _run_training() -> void:
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(
		ConfigKeys.SHADERS_PATHS.FORWARD_PASS
	)
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new(
		ConfigKeys.SHADERS_PATHS.BACKWARD_PASS
	)

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
		ConfigKeys.TRAINER.GRADIENT_CLIP_OPTIMIZER:
			GradientClipOptimizer.new(
				gradient_clip_enable,
				gradient_clip_threshold
			)
	})
	trainer.lr_schedular = LRSchedular.new(
		LRSchedular.Type.COSINE,
		{
			ConfigKeys.LR_SCHEDULAR.STARTING_LR : learning_rate,
			ConfigKeys.LR_SCHEDULAR.MIN_LR : learning_rate * 0.25,
			ConfigKeys.LR_SCHEDULAR.EPOCHS : epochs
		}
	)
	trainer.epoch_finished.connect(on_epoch_finished)
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
			true
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

# Called upon finshing each epoch
func on_epoch_finished(loss_value: float, epoch: int) -> void:
	loss_panel.call_deferred("add_loss", loss_value, epoch)

## Construct network with configured layers and activations
func _create_network(shader_runner: ForwardPassRunner) -> NeuralNetwork:
	return NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: shader_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: hidden_layers_activation,
		ConfigKeys.NETWORK.OUTPUT_ACT: output_layer_activation,
		ConfigKeys.NETWORK.WEIGHT_INIT: weight_initialization
	})

## Executes the training loop and returns elapsed ms
func _run_training_loop(
	trainer: Trainer,
	train_inputs: Array[PackedFloat32Array],
	train_targets: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(train_inputs, train_targets)
	var end_time: int = Time.get_ticks_msec()
	return end_time - start_time

## Cleanup after training finishes
func _on_training_complete() -> void:
	training_thread.wait_to_finish()

# -------------------------------------------------------------------
# Data Helpers
# -------------------------------------------------------------------

## Merge per-category arrays into a single dataset
func concatenate_data(
	data: Dictionary[int, Array],
	data_size: int
) -> Array[PackedFloat32Array]:
	var result: Array[PackedFloat32Array] = []
	for i: int in range(data_size):
		for key: int in data:
			result.append(data[key][i])
	return result

## Read dataset from directories into category dictionary
func training_data_to_dic(path: String) -> Dictionary[int, Array]:
	var results: Dictionary[int, Array] = {}
	var dirs: PackedStringArray = FileUtils.list_dirs(path)

	for i: int in range(dirs.size()):
		results[i] = ImageUtils.read_images(dirs[i], image_scale, true)
		var usable_size: int = int(results[i].size() * input_size_ratio)
		results[i].resize(usable_size)

	return results

## Creates a one-hot encoded vector
func generate_one_hot(
	hot_idx: int,
	length: int
) -> PackedFloat32Array:
	var one_hot: PackedFloat32Array = []
	one_hot.resize(length)
	one_hot.fill(0.0)
	one_hot[hot_idx] = 1.0
	return one_hot

## Creates repeated one-hot vectors
func generate_one_hots(
	hot_idx: int,
	length: int,
	num: int
) -> Array[PackedFloat32Array]:
	var results: Array[PackedFloat32Array] = []
	for i: int in range(num):
		results.append(generate_one_hot(hot_idx, length))
	return results

## Create target dictionary with one-hots per category
func generate_targets(inputs: Dictionary[int, Array]) -> Dictionary[int, Array]:
	var results: Dictionary[int, Array] = {}
	for key: int in inputs:
		results[key] = generate_one_hots(
			key, inputs.size(), inputs[key].size()
		)
	return results
