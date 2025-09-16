extends LRScheduleBase
class_name CosineSchedular

var total_epochs: int
var min_lr : float

func _init(total_epochs_: int, min_lr_: float) -> void:
    total_epochs = total_epochs_
    min_lr = min_lr_

func get_lr(epoch: int, lr: float) -> float:
    return min_lr + 0.5 * (lr - min_lr) * (1 + cos(PI * epoch / total_epochs))
