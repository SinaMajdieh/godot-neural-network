extends RefCounted
class_name Trainer

##
## Handles training of a NeuralNetwork using GPU-accelerated backpropagation.
## Manages batching, loss computation, gradient updates, and error propagation.
##

var network: NeuralNetwork
var runner: ShaderRunner

##
## Constructor: initializes the trainer with a network and shader runner.
##
func _init(network_: NeuralNetwork, runner_: ShaderRunner) -> void:
    network = network_
    runner = runner_

##
## Trains the network using mini-batch backpropagation and compute shaders.
##
## @param full_input Array of input vectors
## @param full_targets Array of target vectors
## @param learning_rate Learning rate for gradient updates
## @param epochs Number of training epochs
## @param batch_size Number of samples per batch
##
func train(
    full_input: Array[PackedFloat32Array],
    full_targets: Array[PackedFloat32Array],
    learning_rate: float,
    epochs: int,
    batch_size: int
) -> void:
    for epoch: int in range(epochs):
        TensorUtils.shuffle_data(full_input, full_targets)
        var batches: Array[Dictionary] = TensorUtils.create_batches(full_input, full_targets, batch_size)
        _train_epoch(epoch, batches, learning_rate)

##
## Trains the network for a single epoch.
##
## @param epoch Current epoch index
## @param batches Array of input/target batches
## @param learning_rate Learning rate for gradient updates
##
func _train_epoch(epoch: int, batches: Array[Dictionary], learning_rate: float) -> void:
    var epoch_loss: float = 0.0

    for batch: Dictionary in batches:
        var input_batch: Array[PackedFloat32Array] = batch["inputs"]
        var target_batch: Array[PackedFloat32Array] = batch["targets"]
        var batch_loss: float = _train_batch(input_batch, target_batch, learning_rate)
        epoch_loss += batch_loss

    print("Epoch %d - Avg Loss: %f" % [epoch, epoch_loss / float(batches.size())])

##
## Trains the network on a single mini-batch.
##
## @param input_batch Array of input vectors
## @param target_batch Array of target vectors
## @param learning_rate Learning rate for gradient updates
## @return Batch loss value
##
func _train_batch(
    input_batch: Array[PackedFloat32Array],
    target_batch: Array[PackedFloat32Array],
    learning_rate: float
) -> float:
    var batch_loss: float = 0.0
    var predictions: PackedFloat32Array = network.forward_pass(input_batch)
    var loss: float = _compute_loss(predictions, target_batch)
    batch_loss += loss

    var output_errors: Array[PackedFloat32Array] = _compute_output_errors(predictions, target_batch)
    var propagated_errors: Array[PackedFloat32Array] = output_errors

    for layer_index: int in range(network.layers.size() - 1, -1, -1):
        var layer: NetworkLayer = network.layers[layer_index]
        var activation: PackedFloat32Array = network.cached_layer_outputs[layer_index]
        var previous_output: PackedFloat32Array = (
            TensorUtils.flatten_batch(input_batch) if layer_index == 0
            else network.cached_layer_outputs[layer_index - 1]
        )

        var gradients: Dictionary = _compute_gradients_for_layer(
            activation, propagated_errors, previous_output, layer, input_batch.size()
        )

        layer.apply_gradients(gradients["weight"], gradients["bias"], learning_rate, input_batch.size())

        if layer_index > 0:
            propagated_errors = _backpropagate_errors(propagated_errors, layer, input_batch.size())

    return batch_loss

##
## Computes mean squared error loss between predictions and targets.
##
## @param predictions Flattened output predictions
## @param targets Array of target vectors
## @return Average loss value
##
func _compute_loss(predictions: PackedFloat32Array, targets: Array[PackedFloat32Array]) -> float:
    var loss: float = 0.0
    var output_size: int = targets[0].size()

    for i: int in range(targets.size()):
        for j: int in range(output_size):
            var pred: float = predictions[i * output_size + j]
            var target: float = targets[i][j]
            var diff: float = pred - target
            loss += diff * diff
            if TensorUtils.is_nan_or_exploding(loss, 1e6):
                push_error("Loss overflow detected at sample %d, output %d" % [i, j])

    return loss / float(targets.size())

##
## Computes output layer errors (derivative of loss function).
##
## @param predictions Flattened output predictions
## @param targets Array of target vectors
## @return Array of error vectors
##
func _compute_output_errors(
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
            var error: float = 2.0 * (pred - target)
            if TensorUtils.is_nan_or_exploding(error, 100.0):
                push_error("Error overflow detected at sample %d, output %d" % [i, j])
                error = clamp(error, -10.0, 10.0)
            error_vector.append(error)
        errors.append(error_vector)

    return errors

##
## Computes gradients for weights and biases of a layer using compute shaders.
##
## @param activation Activation values from current layer
## @param errors Error vectors from current layer
## @param prev_output Flattened output from previous layer
## @param layer Layer to compute gradients for
## @param batch_size Number of samples in batch
## @return Dictionary containing weight and bias gradients
##
func _compute_gradients_for_layer(
    activation: PackedFloat32Array,
    errors: Array[PackedFloat32Array],
    prev_output: PackedFloat32Array,
    layer: NetworkLayer,
    batch_size: int
) -> Dictionary:
    var act_buf: RID = runner.create_buffer(activation)
    var err_buf: RID = runner.create_buffer(TensorUtils.flatten_batch(errors))
    var in_buf: RID = runner.create_buffer(prev_output)

    var grad_buffers: Array[RID] = runner.dispatch_backward(
        act_buf, err_buf, in_buf,
        layer.input_size, layer.output_size,
        batch_size
    )

    var weight_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(grad_buffers[0]))
    var bias_grad: PackedFloat32Array = TensorUtils.bytes_to_floats(runner.get_buffer_data(grad_buffers[1]))

    runner.rd.free_rid(grad_buffers[0])
    runner.rd.free_rid(grad_buffers[1])

    return {
        "weight": weight_grad,
        "bias": bias_grad
    }

##
## Backpropagates errors to the previous layer using weight transposition.
##
## @param errors Error vectors from current layer
## @param layer Layer whose weights are used for propagation
## @param batch_size Number of samples in batch
## @return Array of propagated error vectors
##
func _backpropagate_errors(
    errors: Array[PackedFloat32Array],
    layer: NetworkLayer,
    batch_size: int
) -> Array[PackedFloat32Array]:
    var weights: Array[Array] = layer.get_weight_matrix()
    var new_errors: Array[PackedFloat32Array] = []

    for i: int in range(batch_size):
        var error_vector: PackedFloat32Array = PackedFloat32Array()
        for j: int in range(layer.input_size):
            var sum: float = 0.0
            for k: int in range(layer.output_size):
                sum += errors[i][k] * weights[k][j]
            error_vector.append(sum)
        new_errors.append(error_vector)

    return new_errors