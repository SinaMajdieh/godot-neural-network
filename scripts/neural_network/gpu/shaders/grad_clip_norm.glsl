#version 450
#extension GL_EXT_shader_atomic_float : enable

// 256 threads per workgroup
layout(local_size_x = 256) in;

// Gradients input
layout(std430, binding = 0) readonly buffer GradBuf {
    float grads[];
};

// Output norm² sum
layout(std430, binding = 1) buffer NormBuf {
    float norm2[];
};

// Push constants
layout(push_constant) uniform ClipParams {
    uint grad_count;
};

// Shared memory for reduction
shared float local_sum[256];

void main() {
    uint tid   = gl_GlobalInvocationID.x;  // global thread id
    uint ltid  = gl_LocalInvocationID.x;   // local thread id
    float val_sq = 0.0;

    if (tid < grad_count) {
        float g = grads[tid];
        val_sq = g * g;
    }

    local_sum[ltid] = val_sq;
    barrier();

    // Reduce within workgroup
    for (uint stride = 256u / 2u; stride > 0u; stride >>= 1u) {
        if (ltid < stride) {
            local_sum[ltid] += local_sum[ltid + stride];
        }
        barrier();
    }

    // First thread in each workgroup accumulates into global sum
    if (ltid == 0u) {
        atomicAdd(norm2[0], local_sum[0]);
    }
}
