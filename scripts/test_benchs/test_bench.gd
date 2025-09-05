extends Node

##
## Entry point for training and evaluating a GPU-accelerated neural network.
## Handles data generation, training execution, and error reporting.
##

@export_category("Input")
@export var input_vectors_size: int = 100

@export_category("Network Properties")
@export var layers: Array[int] = [2, 8, 8, 1]

@export_category("Training Properties")
@export var learning_rate: float = 0.1
@export var epochs: int = 100
@export var batch_size: int = 20
@export_range(0.1, 1.0) var test_size: float = 0.2

##
## Called when the node is added to the scene.
## Initializes components, generates data, trains the model, and evaluates performance.
##
func _ready() -> void:
    var shader_runner: ShaderRunner = ShaderRunner.new(
        "res://shaders/forward_pass.spv",
        "res://shaders/backward_pass.spv"
    )

    var network: NeuralNetwork = NeuralNetwork.new(layers, shader_runner)

    var input_vectors: Array[PackedFloat32Array] = []
    var targets: Array[PackedFloat32Array] = []

    _generate_training_data(input_vectors, targets)

    var trainer: Trainer = Trainer.new(network, shader_runner)
    var start_time: int = Time.get_ticks_msec()
    trainer.train(input_vectors, targets, learning_rate, epochs, batch_size)
    var end_time: int = Time.get_ticks_msec()

    print("Training time: %d ms" % (end_time - start_time))

    _evaluate_model(network, input_vectors, targets)

##
## Generates synthetic training data using a custom nonlinear function.
##
## @param input_vectors Array to populate with input samples
## @param targets Array to populate with target outputs
##
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

##
## Evaluates the trained model on a test subset and prints average absolute error.
##
## @param network Trained neural network
## @param input_vectors Full input dataset
## @param targets Full target dataset
##
func _evaluate_model(
    network: NeuralNetwork,
    input_vectors: Array[PackedFloat32Array],
    targets: Array[PackedFloat32Array]
) -> void:
    var total_error: float = 0.0
    var test_size_index: int = int(input_vectors_size * test_size)

    var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
        network.forward_pass(input_vectors.slice(0, test_size_index)),
        1
    )

    for i: int in range(predictions.size()):
        var prediction: float = predictions[i][0]
        var target: float = targets[i][0]
        total_error += abs(prediction - target)

    print("Average absolute error: %f" % (total_error / float(predictions.size())))
