extends RefCounted
class_name Trainer

const KEYS: Dictionary = {
	NETWORK       = "network",
	RUNNER        = "runner",
	LOSS          = "loss",
	LEARNING_RATE = "learning_rate",
	LAMBDA_L2     = "lambda_l2",
	EPOCHS        = "epochs",
	BATCH_SIZE    = "batch_size",
	GRADIENT_CLIP_TRESHOLD = "gradient_clip_treshold"
}

signal epoch_finished(loss: float)

var network: NeuralNetwork
var runner: BackwardPassRunner
var gradient_clip_runner: GradClipRunner
var loss_type: Loss.Type
var learning_rate: float
var lambda_l2: float
var epochs: int
var batch_size: int
var gradient_clip_treshold: float
var lr_schedular: LRSchedular

func _init(config: Dictionary) -> void:
	var defaults: Dictionary = {
		KEYS.NETWORK: null,
		KEYS.RUNNER: null,
		KEYS.LOSS: Loss.Type.BCE,
		KEYS.LEARNING_RATE: 0.1,
		KEYS.LAMBDA_L2: 1e-4,
		KEYS.EPOCHS: 50,
		KEYS.BATCH_SIZE: 64,
		KEYS.GRADIENT_CLIP_TRESHOLD: 1.0
	}
	config = defaults.merged(config, true)
	network = config[KEYS.NETWORK]
	runner = config[KEYS.RUNNER]
	gradient_clip_runner = GradClipRunner.new()
	set_training_attributes(
		config[KEYS.LOSS], config[KEYS.LEARNING_RATE],
		config[KEYS.LAMBDA_L2], config[KEYS.EPOCHS],
		config[KEYS.BATCH_SIZE],
		config[KEYS.GRADIENT_CLIP_TRESHOLD]
	)
	lr_schedular = LRSchedular.new(LRSchedular.Type.NONE)

func set_training_attributes(
	loss_type_: Loss.Type,
	learning_rate_: float,
	lambda_l2_: float,
	epochs_: int,
	batch_size_: int,
	gradient_clip_treshold_: float
) -> void:
	loss_type = loss_type_
	learning_rate = learning_rate_
	lambda_l2 = lambda_l2_
	epochs = epochs_
	batch_size = batch_size_
	gradient_clip_treshold = gradient_clip_treshold_

func train(
	full_input: Array[PackedFloat32Array],
	full_targets: Array[PackedFloat32Array]
) -> void:
	for epoch: int in range(epochs):
		TensorUtils.shuffle_data(full_input, full_targets)
		var batches: Array[Dictionary] = (
			TensorUtils.create_batches(
				full_input, full_targets, batch_size
			)
		)
		learning_rate = lr_schedular.get_lr(
			epoch, learning_rate
		)
		_train_epoch(epoch, batches)

func _train_epoch(
	epoch: int, batches: Array[Dictionary]
) -> void:
	var epoch_loss: float = 0.0
	for batch: Dictionary in batches:
		var in_batch: Array[PackedFloat32Array] = batch["inputs"]
		var tgt_batch: Array[PackedFloat32Array] = batch["targets"]
		var b_loss: float = _train_batch(in_batch, tgt_batch)
		epoch_loss += b_loss
	var avg_loss: float = epoch_loss / float(batches.size())
	epoch_finished.emit(avg_loss)
	print("Epoch %d - Avg Loss: %f" % [epoch, avg_loss])

func _train_batch(
	input_batch: Array[PackedFloat32Array],
	target_batch: Array[PackedFloat32Array]
) -> float:
	var activations: PackedFloat32Array = network.forward_pass(
		input_batch, false
	)
	var logits: PackedFloat32Array = (
		network.cached_pre_act_layer_outputs[-1]
	)
	var loss: float = compute_loss(
		activations, logits, target_batch
	)

	var output_errors: Array[PackedFloat32Array] = compute_output_errors(
		activations, logits, target_batch,
		network.layers_activation[-1]
	)
	var propagated_errors: Array[PackedFloat32Array] = output_errors

	for i: int in range(network.layers.size() - 1, -1, -1):
		var layer: NetworkLayer = network.layers[i]
		var act: PackedFloat32Array = (
			network.cached_post_act_layer_outputs[i]
		)
		var act_type: PackedInt32Array = PackedInt32Array([
			network.layers_activation[i]
		])
		var prev_out: PackedFloat32Array = (
			TensorUtils.flatten_batch(input_batch) if i == 0
			else network.cached_post_act_layer_outputs[i - 1]
		)
		var err_flat: PackedFloat32Array = TensorUtils.flatten_batch(
			propagated_errors
		)

		var grads: Dictionary = _compute_gradients_for_layer(
			act, act_type, err_flat, prev_out, layer,
			input_batch.size()
		)
		layer.update_parameters_with_gradients(
			grads["weight"], grads["bias"],
			learning_rate, lambda_l2,
			input_batch.size()
		)
		if i > 0:
			propagated_errors = _backpropagate_errors(
				propagated_errors, layer,
				network.cached_post_act_layer_outputs[i - 1],
				network.layers_activation[i - 1],
				input_batch.size()
			)
	return loss

func _compute_gradients_for_layer(
	activation: PackedFloat32Array,
	activation_type: PackedInt32Array,
	errors: PackedFloat32Array,
	prev_output: PackedFloat32Array,
	layer: NetworkLayer,
	batch_size_: int
) -> Dictionary:
	# Gradients first computed on GPU for perf
	var grad_bufs: Array[RID] = runner.dispatch_backward(
		activation, errors, prev_output, activation_type,
		layer.input_size, layer.output_size, batch_size_
	)
	var w_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(
		runner.get_buffer_data(grad_bufs[0])
	)
	var b_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(
		runner.get_buffer_data(grad_bufs[1])
	)
	# Free GPU buffers asap to save VRAM
	runner.rd.free_rid(grad_bufs[0])
	runner.rd.free_rid(grad_bufs[1])

	# Clip gradients in one pass
	var combined: PackedFloat32Array = PackedFloat32Array(w_grad)
	combined.append_array(b_grad)
	var clip_buf: RID = gradient_clip_runner.create_buffer(
		combined
	)
	gradient_clip_runner.clip_gradients(
		clip_buf, combined.size(), gradient_clip_treshold
	)
	var clipped: PackedFloat32Array = TensorUtils.bytes_to_floats(
		gradient_clip_runner.get_buffer_data(clip_buf)
	)
	var w_size: int = layer.input_size * layer.output_size
	return {
		"weight": clipped.slice(0, w_size),
		"bias": clipped.slice(w_size, combined.size())
	}

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
	# Prevent silent NaN/overflow propagation
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
