class_name ModelEvaluator
extends RefCounted

# Provides model evaluation utilities for classification accuracy.
# Why: Keeps testing logic clean and independent from training code.

# Evaluates classification accuracy for multi-class models using softmax.
# Params:
#   network (NeuralNetwork): Model to evaluate.
#   inputs (Array[PackedFloat32Array]): Input batch.
#   targets (Array[PackedFloat32Array]): Expected class one-hot vectors.
#   debug (bool): If true, prints misclassifications.
# Returns:
#   float: Accuracy value between 0.0 and 1.0.
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
				"Test case %3d : prediction [color=red]%d[/color] "
				+ "target [color=green]%d[/color]"
				% [i, idx + 1, find_max_value_index(target) + 1]
			)
	return float(correct) / float(predictions.size())


# Evaluates binary classification model accuracy.
# Params:
#   network (NeuralNetwork): Model to evaluate.
#   inputs (Array[PackedFloat32Array]): Input batch.
#   targets (Array[PackedFloat32Array]): Expected binary values.
# Returns:
#   float: Accuracy value between 0.0 and 1.0.
static func evaluate_model(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> float:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat, 1
	)
	var correct: int = 0

	for i: int in range(predictions.size()):
		var pred: float = predictions[i][0]
		var target: float = targets[i][0]
		if int(pred >= 0.5) == int(target):
			correct += 1
	return float(correct) / float(predictions.size())


# Finds the index of the maximum value in a float array.
# Params:
#   arr (PackedFloat32Array): Input values.
# Returns:
#   int: Index of the max value, or -1 if empty.
static func find_max_value_index(arr: PackedFloat32Array) -> int:
	var max_value: float = -INF
	var max_idx: int = -1
	for i: int in range(arr.size()):
		if arr[i] > max_value:
			max_value = arr[i]
			max_idx = i
	return max_idx
