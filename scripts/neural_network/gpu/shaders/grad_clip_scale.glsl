#version 450

// Match grad_clip_norm's workgroup size for symmetry
layout(local_size_x = 256) in;

// Gradients buffer (to scale in-place)
layout(std430, binding = 0) buffer GradBuf {
    float grads[];
};

// Norm² buffer (output from grad_clip_norm.comp)
layout(std430, binding = 1) readonly buffer NormBuf {
    float norm2[];
};

// Push constants
// Vulkan/Godot will pad this to 16 bytes for alignment.
layout(push_constant) uniform ClipParams {
    uint grad_count;   // number of gradient floats
    float clip_norm;   // clip threshold
};

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= grad_count) {
        return;
    }

    // Read the previously computed squared norm
    float norm_sq = norm2[0];

    // Compare squared values to avoid an expensive sqrt unless needed
    float th_sq = clip_norm * clip_norm;
    if (norm_sq <= th_sq || norm_sq == 0.0) {
        return;
    }

    // Compute scaling factor
    float norm_val = sqrt(norm_sq);
    float scale = clip_norm / norm_val;

    grads[tid] *= scale;
}
