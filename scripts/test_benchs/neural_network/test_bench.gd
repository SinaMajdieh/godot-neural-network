## Entry point for training and evaluating a GPU-accelerated neural network.
## This script:
## - Generates synthetic nonlinear training data.
## - Trains a neural network using GPU shaders.
## - Reports performance metrics.
## Why:
## Centralizing setup, training, and evaluation here keeps the high-level
## workflow in one place while training logic stays inside other modules.


extends Node

# ==========================
# ───── EXPORTED PROPERTIES ─────
# ==========================

@export_category("Input")
@export var input_vectors_size: int = 100

@export_category("Network Properties")
@export var layer_sizes: Array[int] = [2, 8, 8, 1]

@export_category("Training Properties")
@export var learning_rate: float = 0.1
@export_range(0.000001, 0.01)
var lambda_l2: float = 0.0001
@export var epochs: int = 100
@export var batch_size: int = 20
@export_range(0.1, 1.0)
var test_size_ratio: float = 0.2


# ==========================
# ───── LIFECYCLE ─────
# ==========================

func _ready() -> void:

	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(ConfigKeys.SHADERS_PATHS.FORWARD_PASS)
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new(ConfigKeys.SHADERS_PATHS.BACKWARD_PASS)
	var network: NeuralNetwork = _create_network(forward_runner)

	var input_vectors: Array[PackedFloat32Array] = []
	var targets: Array[PackedFloat32Array] = []

	_generate_training_data(input_vectors, targets)

	var trainer: Trainer = _create_trainer(backward_runner, network)

	var start_time: int = Time.get_ticks_msec()
	trainer.train(input_vectors, targets)
	var end_time: int = Time.get_ticks_msec()

	print("Training time: %d ms" % (end_time - start_time))
	_evaluate_model(network, input_vectors, targets)


# ==========================
# ───── INITIALIZATION HELPERS ─────
# ==========================

func _create_network(shader_runner: ForwardPassRunner) -> NeuralNetwork:
	return NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: shader_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.WEIGHT_INIT:
			NetworkLayer.WeightInitialization.XAVIER
	})


func _create_trainer(
	shader_runner: BackwardPassRunner,
	network: NeuralNetwork
) -> Trainer:
	return Trainer.new({
		ConfigKeys.TRAINER.NETWORK: network,
		ConfigKeys.TRAINER.RUNNER: shader_runner,
		ConfigKeys.TRAINER.LOSS: Loss.Type.MSE,
		ConfigKeys.TRAINER.LEARNING_RATE: learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2: lambda_l2,
		ConfigKeys.TRAINER.EPOCHS: epochs,
		ConfigKeys.TRAINER.BATCH_SIZE: batch_size
	})


# ==========================
# ───── DATA GENERATION ─────
# ==========================

func _generate_training_data(
	input_vectors: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> void:
	print("Generating training data of size %d" % input_vectors_size)

	for i: int in range(input_vectors_size):
		var x: float = randf_range(-1.0, 1.0)
		var y: float = randf_range(-1.0, 1.0)
		input_vectors.append(PackedFloat32Array([x, y]))

		var z: float = sin(3.0 * x) * cos(2.0 * y) + 0.5 * x * y
		targets.append(PackedFloat32Array([z]))

	print("Data generation complete.")


# ==========================
# ───── EVALUATION ─────
# ==========================

func _evaluate_model(
	network: NeuralNetwork,
	input_vectors: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> void:
	var total_error: float = 0.0
	var test_count: int = int(input_vectors_size * test_size_ratio)

	var predictions: Array[PackedFloat32Array] = (
		TensorUtils.unflatten_batch(
			network.forward_pass(input_vectors.slice(0, test_count)),
			1
		)
	)

	for i: int in range(predictions.size()):
		var prediction: float = predictions[i][0]
		var target: float = targets[i][0]
		total_error += abs(prediction - target)

	print("Average absolute error: %f"
		% (total_error / float(predictions.size())))
