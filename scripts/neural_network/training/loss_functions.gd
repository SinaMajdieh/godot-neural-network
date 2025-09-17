## Loss.gd
## Provides loss function computation and error derivatives for training neural networks.
## Supports Mean Squared Error (MSE) and Binary Cross Entropy (BCE).

class_name Loss
extends RefCounted

## Enum representing supported loss function types.
enum Type{
	MSE,  ## Mean Squared Error
	BCE,  ## Binary Cross Entropy
	CCE,  ## Categorial Cross Entropy
}

## Dispatch tables for loss and error functions.
static var loss_dispatch: Dictionary[Type, Callable] = {
	Type.MSE: compute_mse,
	Type.BCE: compute_bce,
	Type.CCE: compute_cce
}

static var error_dispatch: Dictionary[Type, Callable] = {
	Type.MSE: compute_mse_error,
	Type.BCE: compute_bce_error,
	Type.CCE: compute_cce_error
}

## Computes Mean Squared Error between prediction and target.
static func compute_mse(prediction: float, target: float) -> float:
	var diff: float = prediction - target
	return diff * diff

## Computes derivative of MSE with respect to prediction.
static func compute_mse_error(prediction: float, target: float) -> float:
	return 2.0 * (prediction - target)

## Computes Binary Cross Entropy loss between prediction and target.
## Clamps prediction to avoid log(0) instability.
static func compute_bce(prediction: float, target: float) -> float:
	prediction = clamp(prediction, 1e-7, 1 - 1e-7)
	return -target * log(prediction) - (1.0 - target) * log(1.0 - prediction)

## Computes derivative of BCE with respect to prediction.
static func compute_bce_error(prediction: float, target: float) -> float:
	return prediction - target

# === CCE ===
## Computes categorical cross entropy loss given a probability prediction and one-hot target.
## For performance, sum over classes outside this function for batch processing.
static func compute_cce(prediction: float, target: float) -> float:
	prediction = clamp(prediction, 1e-7, 1.0) # avoid log(0)
	return -target * log(prediction)

## For Softmax + CCE, dL/dz = prediction - target
static func compute_cce_error(prediction: float, target: float) -> float:
	return prediction - target