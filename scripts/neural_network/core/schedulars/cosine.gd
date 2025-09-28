extends LRScheduleBase
class_name CosineSchedular

var total_epochs: int
var min_lr : float

func _init(config: Dictionary) -> void:
    super(config[KEYS.STARTING_LR])
    total_epochs = config[KEYS.EPOCHS]
    min_lr = config[KEYS.MIN_LR]


func get_lr(epoch: int) -> float:
    return min_lr + 0.5 * (starting_lr - min_lr) * (1 + cos(PI * epoch / total_epochs))
