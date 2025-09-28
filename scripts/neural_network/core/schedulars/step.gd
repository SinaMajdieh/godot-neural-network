extends LRScheduleBase
class_name StepSchedular

var decay_rate: float
var step_size: int

func _init(config: Dictionary) -> void:
    super(config[KEYS.STARTING_LR])
    decay_rate = config[KEYS.DECAY_RATE]
    step_size = config[KEYS.STEP_SIZE]

func get_lr(epoch: int) -> float:
    return starting_lr * pow(decay_rate, floor(epoch / float(step_size)))
