extends LRScheduleBase
class_name ExpSchedular

var decay_rate: float

func _init(decay_rate_: float) -> void:
    decay_rate = decay_rate_

func get_lr(epoch: int, lr: float) -> float:
    return lr * exp(-decay_rate * epoch)
