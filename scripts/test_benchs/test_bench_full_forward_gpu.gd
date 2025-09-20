# TestBench.gd
# Simple test bench to verify GPU-based neural network execution.
# Why: Validates shader pipeline, network forward pass, 
# and output integrity in a minimal setup.

extends Node

# Called when the node enters the scene tree.
# Purpose: Initialize network, run a forward pass, display output.
func _ready() -> void:
	# Step 1: Create the shader runner for forward and backward passes.
	var runner: ShaderRunner = ShaderRunner.new(
		"res://scripts/neural_network/gpu/shaders/forward_pass.spv",
		"res://scripts/neural_network/gpu/shaders/backward_pass.spv"
	)

	# Step 2: Define a small network architecture (2→3→1).
	var layer_sizes: Array[int] = [2, 3, 1]
	var network: NeuralNetwork = NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.WEIGHT_INIT: 
			NetworkLayer.WeightInitialization.XAVIER
	})

	# Step 3: Create a batch of predefined input vectors.
	var input_batch: Array[PackedFloat32Array] = [
		PackedFloat32Array([1.0, -1.0]),
		PackedFloat32Array([0.5, 0.5]),
		PackedFloat32Array([0.0, 1.0])
	]

	# Step 4: Run the forward pass.
	var output: PackedFloat32Array = network.forward_pass(input_batch)

	# Step 5: Print results for verification.
	print("Forward pass complete.")
	for i: int in range(input_batch.size()):
		print("Input %d: %s → Output: %f" % 
			[i, input_batch[i], output[i]])


# Generates a randomized input batch for testing.
# Params:
#   batch_size (int): Number of input vectors to generate.
# Returns:
#   Array[PackedFloat32Array]: Batch of random input data.
func generate_input_batch(batch_size: int) -> Array[PackedFloat32Array]:
	var data: Array[PackedFloat32Array] = []
	for _i: int in range(batch_size):
		var input: PackedFloat32Array = PackedFloat32Array([
			randf_range(-1.0, 1.0),
			randf_range(-1.0, 1.0)
		])
		data.append(input)
	return data
