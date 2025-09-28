class_name LRScheduleBase

const KEYS: Dictionary = {
	STARTING_LR = "starting_lr",
	EPOCHS = "epochs",
	MIN_LR = "min_lr",
	DECAY_RATE = "decay_rate",
	STEP_SIZE = "step_size",
    WARMPUP_EPOCHS = "warmup_epochs"
}

var starting_lr: float

func _init(starting_lr_: float) -> void:
    starting_lr = starting_lr_

func get_lr(_epoch: int) -> float:
    return starting_lr