extends RefCounted
class_name ModelEvaluator

## Provides model evaluation utilities for classification accuracy.
## WHY: Keeps testing logic modular and separate from training routines.

# -------------------------------------------------------------------
# Multi-class classification evaluation
# -------------------------------------------------------------------

## Evaluates classification accuracy for multi-class models using softmax.
## WHY: Uses argmax comparison against one-hot encoded targets to measure
##      prediction correctness.
##
## Params:
##   network - model under evaluation
##   inputs - batch of input vectors
##   targets - one-hot encoded expected outputs
##   debug - optional flag to print misclassified samples
##
## Returns:
##   Accuracy score between 0.0 and 1.0
static func evaluate_model_soft_max(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array],
	debug: bool = false
) -> float:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat,
		targets[0].size()
	)
	var correct: int = 0

	for i: int in range(predictions.size()):
		var pred: PackedFloat32Array = predictions[i]
		var target: PackedFloat32Array = targets[i]
		var idx: int = find_max_value_index(pred)

		if target[idx] == 1:
			correct += 1
		elif debug:
			print_rich(
				"Test case %3d: prediction [color=red]%d[/color] target " +
				"[color=green]%d[/color]" % [i, idx, find_max_value_index(target)]
			)

	return float(correct) / float(predictions.size())

# -------------------------------------------------------------------
# Binary classification evaluation
# -------------------------------------------------------------------

## Evaluates binary classification accuracy using a fixed threshold of 0.5.
## WHY: Common for sigmoid output, translates continuous predictions to
##      discrete class labels.
##
## Params:
##   network - model under evaluation
##   inputs - batch of input vectors
##   targets - expected binary outputs
##
## Returns:
##   Accuracy score between 0.0 and 1.0
static func evaluate_model(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> float:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat,
		1
	)
	var correct: int = 0

	for i: int in range(predictions.size()):
		var pred: float = predictions[i][0]
		var target: float = targets[i][0]
		if int(pred >= 0.5) == int(target):
			correct += 1

	return float(correct) / float(predictions.size())

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

## Returns the index of the largest value in an array.
## WHY: Argmax is the standard for selecting predicted class in classification.
##      Returns -1 if input array is empty (avoids undefined index).
static func find_max_value_index(arr: PackedFloat32Array) -> int:
	var max_value: float = -INF
	var max_idx: int = -1
	for i: int in range(arr.size()):
		if arr[i] > max_value:
			max_value = arr[i]
			max_idx = i
	return max_idx
