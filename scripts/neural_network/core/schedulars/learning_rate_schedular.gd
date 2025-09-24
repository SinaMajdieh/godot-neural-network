class_name LRSchedular

var schedular: LRScheduleBase

enum Type {
    NONE,
    STEP,           # Drop the learning rate in discrete chunks
    EXPONENTIAL,    # Smooth instead of stepping
    COSINE,         # reduces LR to near zero by the end of training
    WARMUP_DECAY    # Warmup ramps up before scheduling
}

func _init(type: Type, ...param) -> void:
    match type:
        Type.STEP:
            if param.size() < 2:
                return
            schedular = StepSchedular.new(param[0], param[1])
        Type.EXPONENTIAL:
            if param.size() < 1:
                return
            schedular = ExpSchedular.new(param[0])
        Type.COSINE:
            if param.size() < 2:
                return
            schedular = CosineSchedular.new(param[0], param[1])
        Type.WARMUP_DECAY:
            if param.size() < 3:
                return
            schedular = WarmupDecaySchedular.new(param[0], param[1], param[2])
        _:
            schedular = LRScheduleBase.new()


func get_lr(epoch: int, lr: float) -> float:
    return schedular.get_lr(epoch, lr)
