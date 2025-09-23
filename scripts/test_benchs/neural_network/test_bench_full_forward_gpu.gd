extends Node
class_name TestBench

## Simple test bench for GPU-based neural network execution.
## WHY: Validates shader pipeline, forward pass, and output integrity
##      for a minimal network setup.

var runner: BackwardPassRunner
var gradient_clip_runner: GradClipRunner

## Threshold for gradient clipping during tests
var gradient_clip_threshold: int = 1

## Loss function type used in the test
var loss_type: Loss.Type = Loss.Type.CCE

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------

func _ready() -> void:
	runner = BackwardPassRunner.new()
	gradient_clip_runner = GradClipRunner.new()

	# Step 1: Runner for forward and backward passes
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(
		ConfigKeys.SHADERS_PATHS.FORWARD_PASS
	)

	# Step 2: Minimal network architecture (2 → 3 → 2)
	var layer_sizes: Array[int] = [2, 3, 2]
	var network: NeuralNetwork = NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: forward_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.SOFTMAX,
		ConfigKeys.NETWORK.WEIGHT_INIT:
			NetworkLayer.WeightInitialization.XAVIER
	})

	# Step 3: Predefined input batch
	var input_batch: Array[PackedFloat32Array] = [
		PackedFloat32Array([0.5, 0.5])
	]
	var target_batch: Array[PackedFloat32Array] = [
		PackedFloat32Array([0.0, 1.0])
	]

	# Step 4: First forward pass
	var output: PackedFloat32Array = network.forward_pass(input_batch, false)
	print("Forward pass complete.")
	for i: int in range(input_batch.size()):
		print("Input %d: %s → Output: %s" % [
			i,
			input_batch[i],
			output.slice(
				i * layer_sizes[-1],
				(i + 1) * layer_sizes[-1]
			)
		])

	# Step 5: Core test sequence
	var learning_rate: float = 0.1
	var lambda_l2: float = 0.0

	var predictions: PackedFloat32Array = network.forward_pass(input_batch)
	print("output layer activated:", predictions)

	var logits: PackedFloat32Array = network.cached_pre_act_layer_outputs[-1]
	print("output layer logits:", logits)

	var batch_loss: float = compute_loss(
		predictions, logits, target_batch
	)
	print("loss: %f" % batch_loss)

	var propagated_errors: Array[PackedFloat32Array] = compute_output_errors(
		predictions, logits, target_batch,
		network.layers_activation[-1]
	)
	print("output errors:", propagated_errors)

	# Reverse loop to compute gradients
	var gradients_per_layer: Array[Dictionary] = []
	for layer_index: int in range(network.layers.size() - 1, -1, -1):
		var layer: NetworkLayer = network.layers[layer_index]
		print("layer: %d" % layer_index)

		var activation: PackedFloat32Array = network.cached_post_act_layer_outputs[layer_index]
		logits = network.cached_pre_act_layer_outputs[layer_index]
		print("activated:", activation)
		print("logits:", logits)

		var previous_output: PackedFloat32Array = (
			TensorUtils.flatten_batch(input_batch)
			if layer_index == 0
			else network.cached_post_act_layer_outputs[layer_index - 1]
		)
		print("previous output:", previous_output)

		var act_type: PackedInt32Array = PackedInt32Array([
			network.layers_activation[layer_index]
		])
		var err_flat: PackedFloat32Array = TensorUtils.flatten_batch(
			propagated_errors
		)

		var layer_grads: Dictionary = compute_gradients_for_layer(
			activation, act_type, err_flat, previous_output,
			layer, input_batch.size()
		)
		print("weight grads:", layer_grads["weight"])
		print("bias grads:", layer_grads["bias"])

		gradients_per_layer.insert(0, layer_grads)

		if layer_index > 0:
			propagated_errors = _backpropagate_errors(
				propagated_errors,
				layer,
				network.cached_post_act_layer_outputs[layer_index - 1],
				network.layers_activation[layer_index - 1],
				input_batch.size()
			)
		print("propagated errors:", propagated_errors)

	# Update parameters after all gradient calculations
	for i: int in range(network.layers.size()):
		var grads: Dictionary = gradients_per_layer[i]
		network.layers[i].update_parameters_with_gradients(
			grads["weight"], grads["bias"],
			learning_rate, lambda_l2, input_batch.size()
		)
		print_rich("[color=yellow]layer %d weights: %s" % [
			i, network.layers[i].get_flat_weights()
		])
		print_rich("[color=yellow]layer %d biases: %s" % [
			i, network.layers[i].get_bias_vector()
		])

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

static func generate_input_batch(
	batch_size: int
) -> Array[PackedFloat32Array]:
	var data: Array[PackedFloat32Array] = []
	for _i: int in range(batch_size):
		var input: PackedFloat32Array = PackedFloat32Array([
			randf_range(-1.0, 1.0),
			randf_range(-1.0, 1.0)
		])
		data.append(input)
	return data


func compute_gradients_for_layer(
	activation: PackedFloat32Array,
	activation_type: PackedInt32Array,
	errors: PackedFloat32Array,
	prev_output: PackedFloat32Array,
	layer: NetworkLayer,
	batch_size_: int
) -> Dictionary:
	# First compute gradients via GPU for performance
	var grad_bufs: Array[RID] = runner.dispatch_backward(
		activation, errors, prev_output, activation_type,
		layer.input_size, layer.output_size, batch_size_
	)
	var w_grad: PackedFloat32Array = TensorUtils.uint_bytes_to_floats(
		runner.get_buffer_data(grad_bufs[0])
	)
	var b_grad: PackedFloat32Array = TensorUtils.uint_bytes_to_floats(
		runner.get_buffer_data(grad_bufs[1])
	)

	# Free GPU buffers promptly to release VRAM
	runner.rd.free_rid(grad_bufs[0])
	runner.rd.free_rid(grad_bufs[1])

	return { "weight": w_grad, "bias": b_grad }


func compute_loss(
	acts_flat: PackedFloat32Array,
	logits_flat: PackedFloat32Array,
	targets: Array[PackedFloat32Array]
) -> float:
	var acts: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		acts_flat, targets[0].size()
	)
	var logits: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		logits_flat, targets[0].size()
	)
	var mean_loss: float = Loss.compute_batch_loss(
		loss_type, acts, logits, targets
	)
	if TensorUtils.is_nan_or_exploding(mean_loss, 1e6):
		push_error("Mean loss overflow or NaN after normalization")
	return mean_loss


func compute_output_errors(
	acts_flat: PackedFloat32Array,
	logits_flat: PackedFloat32Array,
	targets: Array[PackedFloat32Array],
	activation_type: Activations.Type
) -> Array[PackedFloat32Array]:
	var acts: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		acts_flat, targets[0].size()
	)
	var logits: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		logits_flat, targets[0].size()
	)
	return Loss.compute_batch_errors(
		loss_type, acts, logits, targets, activation_type
	)


func _backpropagate_errors(
	errors: Array[PackedFloat32Array],
	layer: NetworkLayer,
	prev_activations: PackedFloat32Array,
	activation_type: Activations.Type,
	batch_size_: int
) -> Array[PackedFloat32Array]:
	var weights: Array[Array] = layer.get_weight_matrix()
	var new_errors: Array[PackedFloat32Array] = []

	for i: int in range(batch_size_):
		var err_vec: PackedFloat32Array = PackedFloat32Array()
		for j: int in range(layer.input_size):
			var sum: float = 0.0
			for k: int in range(layer.output_size):
				sum += errors[i][k] * weights[k][j]

			var deriv: float = Activations.activation_derivative(
				prev_activations[i * layer.input_size + j],
				activation_type
			)
			err_vec.append(sum * deriv)
		new_errors.append(err_vec)

	return new_errors
