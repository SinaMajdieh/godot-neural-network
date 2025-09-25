extends RefCounted
class_name Trainer

## Configuration keys for the Trainer
const KEYS: Dictionary = {
	NETWORK = "network",
	RUNNER = "runner",
	LOSS = "loss",
	LEARNING_RATE = "learning_rate",
	LAMBDA_L2 = "lambda_l2",
	EPOCHS = "epochs",
	BATCH_SIZE = "batch_size",
	GRADIENT_CLIP_OPTIMIZER = "gradient_clip_optimizer"
}

signal epoch_finished(loss: float, epoch: int)

var network: NeuralNetwork
var runner: BackwardPassRunner

var loss_type: Loss.Type
var learning_rate: float
var lambda_l2: float
var epochs: int
var batch_size: int

var lr_schedular: LRSchedular
var gradients_clip_optimizer: GradientClipOptimizer

func _init(config: Dictionary) -> void:
	# Merge user config with defaults
	var defaults: Dictionary = {
		KEYS.NETWORK: null,
		KEYS.RUNNER: null,
		KEYS.LOSS: Loss.Type.BCE,
		KEYS.LEARNING_RATE: 0.1,
		KEYS.LAMBDA_L2: 1e-4,
		KEYS.EPOCHS: 50,
		KEYS.BATCH_SIZE: 64,
		KEYS.GRADIENT_CLIP_OPTIMIZER: GradientClipOptimizer.new()
	}
	config = defaults.merged(config, true)

	network = config[KEYS.NETWORK]
	runner = config[KEYS.RUNNER]
	gradients_clip_optimizer = config[KEYS.GRADIENT_CLIP_OPTIMIZER]

	set_training_attributes(
		config[KEYS.LOSS],
		config[KEYS.LEARNING_RATE],
		config[KEYS.LAMBDA_L2],
		config[KEYS.EPOCHS],
		config[KEYS.BATCH_SIZE]
	)

	lr_schedular = LRSchedular.new(LRSchedular.Type.NONE)

func set_training_attributes(
	loss_type_: Loss.Type,
	learning_rate_: float,
	lambda_l2_: float,
	epochs_: int,
	batch_size_: int
) -> void:
	loss_type = loss_type_
	learning_rate = learning_rate_
	lambda_l2 = lambda_l2_
	epochs = epochs_
	batch_size = batch_size_

func train(
	full_input: Array[PackedFloat32Array],
	full_targets: Array[PackedFloat32Array]
) -> void:
	for epoch: int in range(epochs):
		TensorUtils.shuffle_data(full_input, full_targets)

		var batches: Array[Dictionary] = TensorUtils.create_batches(
			full_input, full_targets, batch_size
		)

		learning_rate = lr_schedular.get_lr(epoch, learning_rate)
		_train_epoch(epoch, batches)

func _train_epoch(
	epoch: int,
	batches: Array[Dictionary]
) -> void:
	var epoch_loss: float = 0.0

	for batch: Dictionary in batches:
		var input_batch: Array[PackedFloat32Array] = batch["inputs"]
		var target_batch: Array[PackedFloat32Array] = batch["targets"]

		var batch_loss: float = _train_batch(input_batch, target_batch)
		epoch_loss += batch_loss

	var avg_loss: float = epoch_loss / float(batches.size())
	epoch_finished.emit(avg_loss, epoch)

	print("Epoch %d - Avg Loss: %f" % [epoch, avg_loss])

func _train_batch(
	input_batch: Array[PackedFloat32Array],
	target_batch: Array[PackedFloat32Array]
) -> float:
	var predictions: PackedFloat32Array = network.forward_pass(input_batch)
	var logits: PackedFloat32Array = network.cached_pre_act_layer_outputs[-1]

	# Compute loss from final layer predictions
	var batch_loss: float = compute_loss(
		predictions, logits, target_batch
	)

	# Initial backpropagation errors for the output layer
	var propagated_errors: Array[PackedFloat32Array] = compute_output_errors(
		predictions, logits, target_batch, network.layers_activation[-1]
	)

	var gradients_per_layer: Array[Dictionary] = []

	# Reverse layer traversal to compute gradients
	for layer_index: int in range(network.layers.size() - 1, -1, -1):
		var layer: NetworkLayer = network.layers[layer_index]
		var activation: PackedFloat32Array = (
			network.cached_post_act_layer_outputs[layer_index]
		)
		logits = network.cached_pre_act_layer_outputs[layer_index]

		var previous_output: PackedFloat32Array = (
			TensorUtils.flatten_batch(input_batch)
			if layer_index == 0
			else network.cached_post_act_layer_outputs[layer_index - 1]
		)

		var act_type: PackedInt32Array = PackedInt32Array([
			network.layers_activation[layer_index]
		])

		var err_flat: PackedFloat32Array = TensorUtils.flatten_batch(
			propagated_errors
		)

		var layer_grads: Dictionary = _compute_gradients_for_layer(
			activation, act_type, err_flat, previous_output,
			layer, input_batch.size()
		)

		gradients_per_layer.insert(0, layer_grads)

		if layer_index > 0:
			propagated_errors = _backpropagate_errors(
				propagated_errors,
				layer,
				network.cached_post_act_layer_outputs[layer_index - 1],
				network.layers_activation[layer_index - 1],
				input_batch.size()
			)

	_update_all_parameters(gradients_per_layer, input_batch.size())

	return batch_loss

func _update_all_parameters(
	gradients_per_layer: Array[Dictionary],
	input_batch_size: int
) -> void:
	for i: int in range(network.layers.size()):
		var grads: Dictionary = gradients_per_layer[i]
		network.layers[i].update_parameters_with_gradients(
			grads["weight"], grads["bias"],
			learning_rate, lambda_l2, input_batch_size
		)

func _compute_gradients_for_layer(
	activation: PackedFloat32Array,
	activation_type: PackedInt32Array,
	errors: PackedFloat32Array,
	prev_output: PackedFloat32Array,
	layer: NetworkLayer,
	batch_size_: int
) -> Dictionary:
	# GPU-based gradient computation for performance
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

	# Free GPU buffers immediately to conserve VRAM
	runner.rd.free_rid(grad_bufs[0])
	runner.rd.free_rid(grad_bufs[1])

	return gradients_clip_optimizer.run(layer, w_grad, b_grad)

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

	# Guard against silent NaN or overflows
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
