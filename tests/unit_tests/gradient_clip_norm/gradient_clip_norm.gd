extends Node

##
## Test bench for verifying grad_clip_norm.comp shader correctness.
## Compares GPU-computed squared L2 norm with CPU-calculated reference.
##

@export_file var SHADER_PATH: String = "res://scripts/neural_network/gpu/shaders/grad_clip_norm.spv"

func _ready() -> void:
    # Step 1: Input gradient vector (example)
    var gradients: PackedFloat32Array = PackedFloat32Array([1.0, -2.0, 3.0, -4.0])

    # Step 2: CPU expected value
    var cpu_sum_sq: float = 0.0
    for gradient: float in gradients:
        cpu_sum_sq += gradient * gradient
    print("CPU sum of squares: ", cpu_sum_sq)

    # Step 3: GPU calculation
    var runner: GradClipRunner = GradClipRunner.new(SHADER_PATH)
    var grad_buff: RID = runner.create_buffer(gradients)
    var norm_buf: RID = runner.dispatch_calc_norm(grad_buff, gradients.size())

    # Step 4: Read GPU result
    var norm_bytes: PackedByteArray = runner.get_buffer_data(norm_buf)
    var gpu_sum_sq: float = TensorUtils.bytes_to_floats(norm_bytes)[0]

    print("GPU sum of squares: ", gpu_sum_sq)

    # Step 5: Comparison
    if is_equal_approx(cpu_sum_sq, gpu_sum_sq):
        print_rich("[color=green][PASS] GPU matches CPU")
    else:
        print_rich("[color=red][FAIL] Difference detected")

    # Cleanup
    runner.rd.free_rid(grad_buff)
    runner.rd.free_rid(norm_buf)