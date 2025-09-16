extends LRScheduleBase
class_name WarmupDecaySchedular

var warmup_epochs: int
var total_epochs: int
var min_lr : float

func _init(warmup_epochs_: int, total_epochs_:int, min_lr_: float) -> void:
    warmup_epochs = warmup_epochs_
    total_epochs = total_epochs_
    min_lr = min_lr_

func get_lr(epoch: int, lr: float) -> float:
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    var t: int = epoch - total_epochs
    var t_max: int = total_epochs - warmup_epochs
    return min_lr + 0.5 * (lr - min_lr) * (1 + cos(PI * t / t_max))
