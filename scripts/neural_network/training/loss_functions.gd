class_name Loss
extends RefCounted

enum Type {
	MSE,  # Mean Squared Error
	BCE,  # Binary Cross Entropy
	CCE,  # Categorical Cross Entropy
}

# === Batch Loss Functions ===

static func compute_batch_loss(
	loss_type: Type,
	activations: Array[PackedFloat32Array],
	logits: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> float:
	# Defensive check: batches must align and not be empty
	assert(targets.size() == activations.size()
		and activations.size() == logits.size()
		and targets.size() > 0)

	var total_loss: float = 0.0

	for sample_index: int in range(targets.size()):
		match loss_type:
			Type.MSE:
				total_loss += compute_loss_mse_from_activations(
					activations[sample_index],
					targets[sample_index]
				)
			Type.BCE:
				total_loss += compute_loss_bce_from_logits(
					logits[sample_index],
					targets[sample_index]
				)
			Type.CCE:
				total_loss += compute_loss_cce_from_logits(
					logits[sample_index],
					targets[sample_index]
				)
			_:
				push_error("Unknown Loss Type: %s" % str(loss_type))

	# Mean over all batch elements (matches PyTorch/TensorFlow)
	var output_size: int = targets[0].size()
	return total_loss / (targets.size() * output_size)


static func compute_batch_errors(
	loss_type: Type,
	activations: Array[PackedFloat32Array],
	logits: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array],
	activation_type: Activations.Type
) -> Array[PackedFloat32Array]:
	var errors: Array[PackedFloat32Array] = []

	for sample_index: int in range(targets.size()):
		match loss_type:
			Type.MSE:
				errors.append(compute_error_mse_from_activations(
					activations[sample_index],
					targets[sample_index],
					activation_type
				))
			Type.BCE:
				errors.append(compute_error_bce_from_logits(
					logits[sample_index],
					targets[sample_index]
				))
			Type.CCE:
				errors.append(compute_error_cce_from_logits(
					logits[sample_index],
					targets[sample_index]
				))
			_:
				push_error("Unknown Loss Type: %s" % str(loss_type))
				errors.append(PackedFloat32Array())
	return errors

# === Single-sample Loss Functions ===

static func compute_loss_mse_from_activations(
	activations: PackedFloat32Array,
	targets: PackedFloat32Array
) -> float:
	var loss: float = 0.0
	for value_index: int in range(activations.size()):
		var diff: float = activations[value_index] - targets[value_index]
		loss += diff * diff
	return loss


static func compute_loss_bce_from_logits(
	logits: PackedFloat32Array,
	targets: PackedFloat32Array
) -> float:
	var loss: float = 0.0
	for value_index: int in range(logits.size()):
		var z: float = logits[value_index]
		# Branch by z’s sign to prevent exp overflow
		if z >= 0.0:
			loss += (1.0 - targets[value_index]) * z \
			        + log(1.0 + exp(-z))
		else:
			loss += -targets[value_index] * z \
			        + log(1.0 + exp(z))
	return loss


static func compute_loss_cce_from_logits(
	logits: PackedFloat32Array,
	targets: PackedFloat32Array
) -> float:
	var loss: float = 0.0

	# Shift logits by max for stable exp
	var max_logit: float = -INF
	for logit: float in logits:
		if logit > max_logit:
			max_logit = logit

	var sum_exp: float = 0.0
	for logit: float in logits:
		sum_exp += exp(logit - max_logit)

	var log_sum_exp: float = log(sum_exp) + max_logit
	# Only target index contributes to CCE
	for value_index: int in range(logits.size()):
		if targets[value_index] > 0.0:
			loss -= targets[value_index] * (
				logits[value_index] - log_sum_exp
			)
	return loss

# === Single-sample Error Functions ===

static func compute_error_mse_from_activations(
	activations: PackedFloat32Array,
	targets: PackedFloat32Array,
	activation_type: Activations.Type
) -> PackedFloat32Array:
	var errors: PackedFloat32Array = PackedFloat32Array()
	for value_index: int in range(activations.size()):
		# Derivative of MSE w.r.t. z = 2(a - t) * f'(a)
		var diff: float = 2.0 * (
			activations[value_index] - targets[value_index]
		)
		var derivative: float = Activations.activation_derivative(
			activations[value_index], activation_type
		)
		var error: float = diff * derivative
		if TensorUtils.is_nan_or_exploding(error, 100.0):
			push_error("Error overflow detected : %3.2f" % error)
		errors.append(error)
	return errors


static func compute_error_bce_from_logits(
	logits: PackedFloat32Array,
	targets: PackedFloat32Array
) -> PackedFloat32Array:
	var errors: PackedFloat32Array = PackedFloat32Array()
	for value_index: int in range(logits.size()):
		# From BCE+sigmoid simplification: σ(z) - y
		var s: float = 1.0 / (1.0 + exp(-logits[value_index]))
		var error: float = s - targets[value_index]
		if TensorUtils.is_nan_or_exploding(error, 100.0):
			push_error("Error overflow detected : %3.2f" % error)
		errors.append(error)
	return errors


static func compute_error_cce_from_logits(
	logits: PackedFloat32Array,
	targets: PackedFloat32Array
) -> PackedFloat32Array:
	var errors: PackedFloat32Array = PackedFloat32Array()

	# Softmax shift for stability
	var max_logit: float = -INF
	for logit: float in logits:
		if logit > max_logit:
			max_logit = logit

	var sum_exp: float = 0.0
	for logit: float in logits:
		sum_exp += exp(logit - max_logit)

	# Gradient: p - y
	for value_index: int in range(logits.size()):
		var p: float = exp(
			logits[value_index] - max_logit
		) / sum_exp
		var error: float = p - targets[value_index]
		if TensorUtils.is_nan_or_exploding(error, 100.0):
			push_error("Error overflow detected : %3.2f" % error)
		errors.append(error)
	return errors
