## TestBench.gd
## Simple test bench to verify GPU-based neural network execution.
## Initializes a small network, runs a forward pass, and prints the output.

extends Node

## Entry point for the test bench.
func _ready() -> void:
    # Step 1: Create the shader runner
    var runner: ShaderRunner = ShaderRunner.new(
        "res://scripts/neural_network/gpu/shaders/forward_pass.spv",
        "res://scripts/neural_network/gpu/shaders/backward_pass.spv"
    )

    # Step 2: Define a small network architecture
    # Example: 2 inputs → 3 hidden → 1 output
    var layer_sizes: Array[int] = [2, 3, 1]
    var network: NeuralNetwork = NeuralNetwork.new(
        layer_sizes,
        runner,
        Activations.Type.TANH,
        Activations.Type.TANH
    )

    # Step 3: Create a small batch of input vectors
    var input_batch: Array[PackedFloat32Array] = [
        PackedFloat32Array([1.0, -1.0]),
        PackedFloat32Array([0.5, 0.5]),
        PackedFloat32Array([0.0, 1.0])
    ]

    # Step 4: Run the forward pass
    var output: PackedFloat32Array = network.forward_pass(input_batch)

    # Step 5: Print the results
    print("✅ Forward pass complete.")
    for i: int in range(input_batch.size()):
        print("Input %d: %s → Output: %f" % [i, input_batch[i], output[i]])

## Generates a randomized input batch for testing purposes.
## @param batch_size Number of input vectors
## @return Array of PackedFloat32Array inputs
func generate_input_batch(batch_size: int) -> Array[PackedFloat32Array]:
    var data: Array[PackedFloat32Array] = []
    for _i: int in range(batch_size):
        var input: PackedFloat32Array = PackedFloat32Array([
            randf_range(-1.0, 1.0),
            randf_range(-1.0, 1.0)
        ])
        data.append(input)
    return data
