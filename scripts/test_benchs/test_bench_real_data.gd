extends Node

@export_category("Input")
@export_range(0.1, 1.0)
var input_size_ratio: float = 1.0

@export_category("Network Properties")
@export var layers: Array[int] = [10, 16, 8, 1]

@export_category("Training Properties")
@export_range(0.000001, 10) var learning_rate: float = 0.001
@export_range(0.000001, 0.01) var lambda_l2: float = 0.0001
@export_range(1, 500) var epochs: int = 100
@export_range(1, 10000) var batch_size: int = 20
@export_range(0.1, 1.0) var test_size: float = 0.2

var training_thread: Thread

##
## Called when the node is added to the scene.
## Starts the training thread.
##
func _ready() -> void:
    training_thread = Thread.new()
    training_thread.start(_run_training)

##
## Threaded entry point for training and evaluation.
##
func _run_training() -> void:
    var shader_runner: ShaderRunner = _create_shader_runner()
    var network: NeuralNetwork = _create_network(shader_runner)
    var split: DataSplit = _load_and_split_data()
    print(split.train_targets.size())
    print(split.train_targets.count(PackedFloat32Array([1.0])))
    print(split.train_targets)

    var trainer: Trainer = Trainer.new({
        ConfigKeys.TRAINER.NETWORK: network, 
        ConfigKeys.TRAINER.RUNNER: shader_runner, 
        ConfigKeys.TRAINER.LOSS: Loss.Type.BCE, 
        ConfigKeys.TRAINER.LEARNING_RATE: learning_rate, 
        ConfigKeys.TRAINER.LAMBDA_L2: lambda_l2, 
        ConfigKeys.TRAINER.EPOCHS: epochs, 
        ConfigKeys.TRAINER.BATCH_SIZE: batch_size
    })
    var training_time: int = _run_training_loop(trainer, split.train_inputs, split.train_targets)

    print("Training time: %d ms" % training_time)
    _evaluate_model(network, split.test_inputs, split.test_targets)
    call_deferred("_on_training_complete")

##
## Creates and returns a ShaderRunner instance.
##
func _create_shader_runner() -> ShaderRunner:
    return ShaderRunner.new(
        "res://scripts/neural_network/gpu/shaders/forward_pass.spv",
        "res://scripts/neural_network/gpu/shaders/backward_pass.spv"
    )

##
## Initializes the neural network with the configured layer sizes.
##
func _create_network(shader_runner: ShaderRunner) -> NeuralNetwork:
    return NeuralNetwork.new({
        ConfigKeys.NETWORK.LAYER_SIZES: layers, 
        ConfigKeys.NETWORK.RUNNER: shader_runner, 
        ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH, 
        ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.SIGMOID, 
        ConfigKeys.NETWORK.WEIGHT_INIT: NetworkLayer.WeightInitialization.XAVIER
    })

##
## Loads input and target data, applies slicing, and returns a train/test split.
##
func _load_and_split_data() -> DataSplit:
    var input_vectors: Array[PackedFloat32Array] = DataSetUtils.load_csv_as_batches(
        "res://data/processed_inputs.csv", 10
    )
    var targets: Array[PackedFloat32Array] = DataSetUtils.load_csv_as_batches(
        "res://data/processed_targets.csv", 1
    )

    assert(input_vectors.size() == targets.size())

    var usable_size: int = int(input_vectors.size() * input_size_ratio)
    input_vectors = input_vectors.slice(0, usable_size)
    targets = targets.slice(0, usable_size)

    return DataSetUtils.train_test_split(input_vectors, targets, test_size)

##
## Runs the training loop and returns elapsed time in milliseconds.
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
## Called after training completes to clean up the training thread.
##
func _on_training_complete() -> void:
    training_thread.wait_to_finish()

##
## Evaluates the trained model on a test subset and prints classification accuracy.
##
## @param network Trained neural network
## @param inputs Test input dataset
## @param targets Test target dataset
##
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
        if binary_pred == int(target):
            correct += 1

    var accuracy: float = float(correct) / float(predictions.size())
    print("Test Accuracy: %f" % accuracy)