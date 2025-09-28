class_name LRSchedular

var schedular: LRScheduleBase

enum Type {
	NONE,
	STEP,           # Drop the learning rate in discrete chunks
	EXPONENTIAL,    # Smooth instead of stepping
	COSINE,         # reduces LR to near zero by the end of training
	WARMUP_DECAY    # Warmup ramps up before scheduling
}

func _init(type: Type, param: Dictionary) -> void:
	match type:
		Type.STEP:
			schedular = StepSchedular.new(param)
		Type.EXPONENTIAL:
			schedular = ExpSchedular.new(param)
		Type.COSINE:
			schedular = CosineSchedular.new(param)
		Type.WARMUP_DECAY:
			schedular = WarmupDecaySchedular.new(param)
		_:
			schedular = LRScheduleBase.new(param[LRScheduleBase.KEYS.STARTING_LR])


func get_lr(epoch: int) -> float:
	return schedular.get_lr(epoch)
