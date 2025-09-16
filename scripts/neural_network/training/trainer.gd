## Handles training of a NeuralNetwork using GPU-accelerated backpropagation.
## Manages batching, loss computation, gradient updates, and error propagation.

extends RefCounted
class_name Trainer

## Signal emitted after each epoch with average loss.
signal epoch_finished(loss: float)

## Training context.
var network: NeuralNetwork
var runner: ShaderRunner

## Loss and error function dispatchers.
var error_function: Callable
var loss_function: Callable

# Training attributes
var	learning_rate: float
var	lambda_l2: float
var	epochs: int
var	batch_size: int

var lr_schedular: LRSchedular

## Constructs the trainer with a neural network and shader runner.
## @param network_ NeuralNetwork instance
## @param runner_ ShaderRunner instance
## @param loss Loss.Type enum specifying loss function
func _init(network_: NeuralNetwork, runner_: ShaderRunner, loss: Loss.Type, 
	learning_rate_: float,
	lambda_l2_: float,
	epochs_: int,
	batch_size_: int,
) -> void:
	network = network_
	runner = runner_
	set_training_attributes(learning_rate_, lambda_l2_, epochs_, batch_size_)
	set_loss_function(loss)
	lr_schedular = LRSchedular.new(LRSchedular.Type.COSINE, epochs, 0.2)

func set_training_attributes(
	learning_rate_: float,
	lambda_l2_: float,
	epochs_: int,
	batch_size_: int,
) -> void:
	learning_rate = learning_rate_
	lambda_l2 = lambda_l2_
	epochs = epochs_
	batch_size = batch_size_

## Sets the loss and error functions based on selected loss type.
func set_loss_function(loss: Loss.Type) -> void:
	loss_function = Loss.loss_dispatch[loss]
	error_function = Loss.error_dispatch[loss]

## Trains the network using mini-batch backpropagation.
## @param full_input Full dataset inputs
## @param full_targets Full dataset targets
## @param learning_rate Learning rate for gradient descent
## @param epochs Number of training epochs
## @param batch_size Number of samples per batch
func train(
	full_input: Array[PackedFloat32Array],
	full_targets: Array[PackedFloat32Array],
) -> void:
	for epoch: int in range(epochs):
		TensorUtils.shuffle_data(full_input, full_targets)
		var batches: Array[Dictionary] = TensorUtils.create_batches(full_input, full_targets, batch_size)
		_train_epoch(epoch, batches)

## Trains the network for a single epoch.
## @param epoch Current epoch index
## @param batches Array of input/target batches
## @param learning_rate Learning rate for gradient descent
func _train_epoch(epoch: int, batches: Array[Dictionary]) -> void:
	learning_rate = lr_schedular.get_lr(epoch, learning_rate)
	var epoch_loss: float = 0.0
	for batch: Dictionary in batches:
		var input_batch: Array[PackedFloat32Array] = batch["inputs"]
		var target_batch: Array[PackedFloat32Array] = batch["targets"]
		var batch_loss: float = _train_batch(input_batch, target_batch)
		epoch_loss += batch_loss
	var avg_loss: float = epoch_loss / float(batches.size())
	epoch_finished.emit(avg_loss)
	print("Epoch %d - Avg Loss: %f" % [epoch, avg_loss])

## Trains the network on a single mini-batch.
## @param input_batch Batch of input samples
## @param target_batch Batch of target samples
## @param learning_rate Learning rate for gradient descent
## @return Loss value for the batch
func _train_batch(
	input_batch: Array[PackedFloat32Array],
	target_batch: Array[PackedFloat32Array],
) -> float:
	var predictions: PackedFloat32Array = network.forward_pass(input_batch)
	var loss: float = compute_loss(predictions, target_batch)

	var output_errors: Array[PackedFloat32Array] = compute_output_errors(predictions, target_batch)
	var propagated_errors: Array[PackedFloat32Array] = output_errors

	for layer_index: int in range(network.layers.size() - 1, -1, -1):
		var layer: NetworkLayer = network.layers[layer_index]
		var activation: PackedFloat32Array = network.cached_layer_outputs[layer_index]
		var layer_activation: PackedInt32Array = PackedInt32Array([network.layers_activation[layer_index]])
		var previous_output: PackedFloat32Array = (
			TensorUtils.flatten_batch(input_batch) if layer_index == 0
			else network.cached_layer_outputs[layer_index - 1]
		)
		var errors_flat: PackedFloat32Array = TensorUtils.flatten_batch(propagated_errors)

		var gradients: Dictionary = _compute_gradients_for_layer(
			activation,
			layer_activation,
			errors_flat,
			previous_output,
			layer,
			input_batch.size()
		)

		layer.update_parameters_with_gradients(
			gradients["weight"] as PackedFloat32Array,
			gradients["bias"] as PackedFloat32Array,
			learning_rate,
			lambda_l2,
			input_batch.size()
		)

		if layer_index > 0:
			propagated_errors = _backpropagate_errors(propagated_errors, layer, input_batch.size())

	return loss

## Computes gradients for weights and biases of a layer using compute shaders.
## @return Dictionary containing weight and bias gradients
func _compute_gradients_for_layer(
	activation: PackedFloat32Array,
	activation_type: PackedInt32Array,
	errors: PackedFloat32Array,
	prev_output: PackedFloat32Array,
	layer: NetworkLayer,
	batch_size_: int
) -> Dictionary:
	var grad_buffers: Array[RID] = runner.dispatch_backward(
		activation, errors, prev_output, activation_type,
		layer.input_size, layer.output_size,
		batch_size_
	)

	var weight_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(grad_buffers[0]))
	var bias_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(grad_buffers[1]))

	runner.rd.free_rid(grad_buffers[0])
	runner.rd.free_rid(grad_buffers[1])

	return {
		"weight": weight_grad,
		"bias": bias_grad
	}

## Computes average loss over predictions and targets using selected loss function.
func compute_loss(predictions: PackedFloat32Array, targets: Array[PackedFloat32Array]) -> float:
	var loss: float = 0.0
	var output_size: int = targets[0].size()

	for i: int in range(targets.size()):
		for j: int in range(output_size):
			var pred: float = predictions[i * output_size + j]
			var target: float = targets[i][j]
			loss += loss_function.call(pred, target)
			if TensorUtils.is_nan_or_exploding(loss, 1e6):
				push_error("Loss overflow detected at sample %d, output %d" % [i, j])

	return loss / float(targets.size() * output_size)

## Computes output layer errors (derivative of loss function).
func compute_output_errors(
	predictions: PackedFloat32Array,
	targets: Array[PackedFloat32Array]
) -> Array[PackedFloat32Array]:
	var output_size: int = targets[0].size()
	var errors: Array[PackedFloat32Array] = []

	for i: int in range(targets.size()):
		var error_vector: PackedFloat32Array = PackedFloat32Array()
		for j: int in range(output_size):
			var pred: float = predictions[i * output_size + j]
			var target: float = targets[i][j]
			var error: float = error_function.call(pred, target)
			if TensorUtils.is_nan_or_exploding(error, 100.0):
				push_error("Error overflow detected at sample %d, output %d" % [i, j])
				error = clamp(error, -10.0, 10.0)
			error_vector.append(error)
		errors.append(error_vector)

	return errors

## Backpropagates errors to the previous layer using weight transposition.
func _backpropagate_errors(
	errors: Array[PackedFloat32Array],
	layer: NetworkLayer,
	batch_size_: int
) -> Array[PackedFloat32Array]:
	var weights: Array[Array] = layer.get_weight_matrix()
	var new_errors: Array[PackedFloat32Array] = []

	for i: int in range(batch_size_):
		var error_vector: PackedFloat32Array = PackedFloat32Array()
		for j: int in range(layer.input_size):
			var sum: float = 0.0
			for k: int in range(layer.output_size):
				sum += errors[i][k] * weights[k][j]
			error_vector.append(sum)
		new_errors.append(error_vector)

	return new_errors
