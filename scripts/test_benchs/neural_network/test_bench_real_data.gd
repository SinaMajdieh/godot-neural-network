
## Trains and evaluates a GPU-accelerated neural network in a background thread.
## Why:
## Separating training to a dedicated thread prevents blocking the main
## application loop and allows the Godot editor/game to remain responsive.

extends Node

# ==========================
# ───── EXPORTED PROPERTIES ─────
# ==========================

@export_category("Input")
@export_range(0.1, 1.0)
var input_size_ratio: float = 1.0

@export_category("Network Properties")
@export var layer_sizes: Array[int] = [10, 16, 8, 1]

@export_category("Training Properties")
@export_range(0.000001, 10)
var learning_rate: float = 0.001
@export_range(0.000001, 0.01)
var lambda_l2: float = 0.0001
@export_range(1, 500)
var epochs: int = 100
@export_range(1, 10000)
var batch_size: int = 20
@export_range(0.1, 1.0)
var test_size_ratio: float = 0.2


# ==========================
# ───── INTERNAL STATE ─────
# ==========================

var training_thread: Thread


# ==========================
# ───── LIFECYCLE ─────
# ==========================

func _ready() -> void:
	training_thread = Thread.new()
	training_thread.start(_run_training)


# ==========================
# ───── TRAINING WORKFLOW ─────
# ==========================

func _run_training() -> void:
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(ConfigKeys.SHADERS_PATHS.FORWARD_PASS)
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new(ConfigKeys.SHADERS_PATHS.BACKWARD_PASS)
	var network: NeuralNetwork = _create_network(forward_runner)
	var split: DataSplit= _load_and_split_data()

	print(split.train_targets.size())
	print(split.train_targets.count(PackedFloat32Array([1.0])))
	print(split.train_targets)

	var trainer: Trainer= _create_trainer(backward_runner, network)
	var training_time: int = _run_training_loop(
		trainer,
		split.train_inputs,
		split.train_targets
	)

	print("Training time: %d ms" % training_time)
	var accuracy: float = ModelEvaluator.evaluate_model(
		network,
		split.test_inputs,
		split.test_targets
	)
	print("Average absolute error: %f" % accuracy)

	call_deferred("_on_training_complete")


func _create_network(shader_runner: ForwardPassRunner) -> NeuralNetwork:
	return NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: shader_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.SIGMOID,
		ConfigKeys.NETWORK.WEIGHT_INIT:
			NetworkLayer.WeightInitialization.XAVIER
	})

##
## Creates the trainer object with optimizer, loss, and other settings.
##
func _create_trainer(
	shader_runner: BackwardPassRunner,
	network: NeuralNetwork
) -> Trainer:
	return Trainer.new({
		ConfigKeys.TRAINER.NETWORK: network,
		ConfigKeys.TRAINER.RUNNER: shader_runner,
		ConfigKeys.TRAINER.LOSS: Loss.Type.BCE,
		ConfigKeys.TRAINER.LEARNING_RATE: learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2: lambda_l2,
		ConfigKeys.TRAINER.EPOCHS: epochs,
		ConfigKeys.TRAINER.BATCH_SIZE: batch_size
	})

##
## Loads CSV input/target data, slices per ratio, and returns train/test split.
##
func _load_and_split_data() -> DataSplit:
	var input_vectors: Array[PackedFloat32Array] = DataSetUtils.load_csv_as_batches(
		"res://data/processed_inputs.csv",
		10
	)
	var targets: Array[PackedFloat32Array] = DataSetUtils.load_csv_as_batches(
		"res://data/processed_targets.csv",
		1
	)

	assert(input_vectors.size() == targets.size())

	var usable_size: int = int(input_vectors.size() * input_size_ratio)
	input_vectors = input_vectors.slice(0, usable_size)
	targets = targets.slice(0, usable_size)

	return DataSetUtils.train_test_split(
		input_vectors,
		targets,
		test_size_ratio
	)


##
## Executes training and returns elapsed time in milliseconds.
##
func _run_training_loop(
	trainer: Trainer,
	train_inputs: Array[PackedFloat32Array],
	train_targets: Array[PackedFloat32Array]
) -> int:
	var start_time: int = Time.get_ticks_msec()
	trainer.train(train_inputs, train_targets)
	var end_time: int = Time.get_ticks_msec()
	return end_time - start_time



##
## Joins the training thread once training is done.
##
func _on_training_complete() -> void:

	training_thread.wait_to_finish()
