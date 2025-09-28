extends LRScheduleBase
class_name ExpSchedular

var decay_rate: float

func _init(config: Dictionary) -> void:
    super(config[KEYS.STARTING_LR])
    decay_rate = config[KEYS.DECAY_RATE]

func get_lr(epoch: int) -> float:
    return starting_lr * exp(-decay_rate * epoch)
