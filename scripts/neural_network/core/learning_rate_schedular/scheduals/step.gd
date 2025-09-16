extends LRScheduleBase
class_name StepSchedular

var decay_factor: float
var step_size: int

func _init(decay_factor_: float, step_size_: int) -> void:
    decay_factor = decay_factor_
    step_size = step_size_

func get_lr(epoch: int, lr: float) -> float:
    return lr * pow(decay_factor, floor(epoch / float(step_size)))
