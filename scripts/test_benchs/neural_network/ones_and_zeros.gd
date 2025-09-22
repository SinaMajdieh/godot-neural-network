# TrainingBench.gd
# Node-based test bench for GPU-accelerated neural network training.
# Why: Quick iteration and evaluation without modular overhead.

extends Node

# Thread for asynchronous training.
var training_thread: Thread

@export_category("Test Bench Attributes")
@export_global_dir var zeros_dir: String
@export_global_dir var ones_dir: String
@export_range(0.1, 1.0) var image_scale: float = 0.25

@export_category("Input")
@export_range(0.0, 1.0) var input_size_ratio: float = 1.0

@export_category("Network Properties")
@export var layer_sizes: Array[int] = [32 * 32, 32, 16, 1]

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

func _ready() -> void:
	training_thread = Thread.new()
	training_thread.start(_run_training)

# Holds paired input/target arrays for each label.
class Data:
	var inputs: Array[PackedFloat32Array]
	var targets: Array[PackedFloat32Array]

	func _init(
		inputs_: Array[PackedFloat32Array],
		targets_: Array[PackedFloat32Array]
	) -> void:
		inputs = inputs_
		targets = targets_

# Loads and merges zero/one datasets into a single DataSplit.
func read_data() -> DataSplit:
	var result: DataSplit = DataSplit.new()
	var ones: Data = read_label(
		ones_dir, PackedFloat32Array([0, 1])
	)
	var zeros: Data = read_label(
		zeros_dir, PackedFloat32Array([1, 0])
	)

	var usable_size: int = int(
		(zeros.inputs.size() + ones.inputs.size()) *
		input_size_ratio / 2
	)
	ones.inputs = ones.inputs.slice(0, usable_size)
	ones.targets = ones.targets.slice(0, usable_size)
	zeros.inputs = zeros.inputs.slice(0, usable_size)
	zeros.targets = zeros.targets.slice(0, usable_size)

	var split0: DataSplit = DataSetUtils.train_test_split(
		zeros.inputs, zeros.targets, test_size_ratio
	)
	var split1: DataSplit = DataSetUtils.train_test_split(
		ones.inputs, ones.targets, test_size_ratio
	)

	_append_split(result.train_inputs, split0.train_inputs, split1.train_inputs)
	_append_split(result.train_targets, split0.train_targets, split1.train_targets)
	_append_split(result.test_inputs, split0.test_inputs, split1.test_inputs)
	_append_split(result.test_targets, split0.test_targets, split1.test_targets)

	return result

# Reads images for a given label and applies a fixed target vector.
func read_label(
	dir: String,
	target_vec: PackedFloat32Array
) -> Data:
	var imgs: Array[PackedFloat32Array] = []
	var targets: Array[PackedFloat32Array] = []
	for path: String in FileUtils.list_files(dir):
		imgs.append(ImageUtils.read_image(path, image_scale))
		targets.append(target_vec)
	return Data.new(imgs, targets)

# Helper: merges category data into a single set.
func _append_split(
	dest: Array,
	s0: Array,
	s1: Array
) -> void:
	for i: int in range(s0.size()):
		dest.append(s0[i])
		dest.append(s1[i])

# Main training entry point.
func _run_training() -> void:
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(ConfigKeys.SHADERS_PATHS.FORWARD_PASS)
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new(ConfigKeys.SHADERS_PATHS.BACKWARD_PASS)
	var network: NeuralNetwork = _create_network(forward_runner)
	var split: DataSplit = read_data()

	var trainer: Trainer = Trainer.new({
		ConfigKeys.TRAINER.NETWORK: network,
		ConfigKeys.TRAINER.RUNNER: backward_runner,
		ConfigKeys.TRAINER.LOSS: loss,
		ConfigKeys.TRAINER.LEARNING_RATE: learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2: lambda_l2,
		ConfigKeys.TRAINER.EPOCHS: epochs,
		ConfigKeys.TRAINER.BATCH_SIZE: batch_size
	})
	trainer.lr_schedular = LRSchedular.new(
		LRSchedular.Type.COSINE, epochs, 0.005
	)

	var satisfied: bool = false
	while not satisfied:
		var time_ms: int = _train_once(
			trainer, split.train_inputs, split.train_targets
		)
		print("Training time: %d ms" % time_ms)

		var test_acc: float = ModelEvaluator.evaluate_model_soft_max(
			network, split.test_inputs, split.test_targets	
		)
		print_rich("[color=cyan]Test Accuracy: %4.2f" % test_acc)

		if test_acc >= 0.9:
			satisfied = true

		var train_acc: float = ModelEvaluator.evaluate_model_soft_max(
			network, split.train_inputs, split.train_targets
		)
		print_rich("[color=cyan]Training Accuracy: %4.2f" % train_acc)

	NeuralNetworkSerializer.export(
		network, "0_and_1_digit_recognition.tres"
	)
	call_deferred("_on_training_complete")

# Runs one training cycle and returns elapsed time in ms.
func _train_once(
	trainer: Trainer,
	tr_inputs: Array[PackedFloat32Array],
	tr_targets: Array[PackedFloat32Array]
) -> int:
	var start_ms: int = Time.get_ticks_msec()
	trainer.train(tr_inputs, tr_targets)
	return Time.get_ticks_msec() - start_ms

# Creates the neural network based on current parameters.
func _create_network(runner: ForwardPassRunner) -> NeuralNetwork:
	return NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: hidden_layers_activation,
		ConfigKeys.NETWORK.OUTPUT_ACT: output_layer_activation,
		ConfigKeys.NETWORK.WEIGHT_INIT: weight_initialization
	})

# Cleans up the thread after training is done.
func _on_training_complete() -> void:
	training_thread.wait_to_finish()
