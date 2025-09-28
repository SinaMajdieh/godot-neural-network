extends LRScheduleBase
class_name WarmupDecaySchedular

var warmup_epochs: int
var total_epochs: int
var min_lr : float

func _init(config: Dictionary) -> void:
    super(config[KEYS.STARTING_LR])
    warmup_epochs = config[KEYS.WARMPUP_EPOCHS]
    total_epochs = config[KEYS.EPOCHS]
    min_lr = config[KEYS.MIN_LR]

func get_lr(epoch: int) -> float:
    if epoch < warmup_epochs:
        return starting_lr * (epoch + 1) / warmup_epochs
    var t: int = epoch - total_epochs
    var t_max: int = total_epochs - warmup_epochs
    return min_lr + 0.5 * (starting_lr - min_lr) * (1 + cos(PI * t / t_max))
