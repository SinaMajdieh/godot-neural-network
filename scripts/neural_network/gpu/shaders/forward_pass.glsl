#version 450
#define ACT_LINEAR       0
#define ACT_SIGMOID      1
#define ACT_TANH         2
#define ACT_RELU         3
#define ACT_LEAKY_RELU   4
#define ACT_SOFTMAX      5

layout(local_size_x = 64) in;

//
// === Buffer Bindings ===
//
layout(std430, binding = 0) buffer InputBuffer {
    float inputs[];
};

layout(std430, binding = 1) buffer WeightBuffer {
    float weights[];
};

layout(std430, binding = 2) buffer BiasBuffer {
    float biases[];
};

layout(std430, binding = 3) buffer ActivationTypeBuffer {
    uint activation_types[]; // One per layer
};

layout(std430, binding = 4) buffer IntermediatesBuffer {
    float intermediates[];
};

layout(std430, binding = 5) buffer MetaBuffer {
    uint layer_count;
    uint batch_size;
    uint input_sizes[32];
    uint output_sizes[32];
    uint weight_offsets[32];
    uint bias_offsets[32];
    uint interm_offsets[32];
};


//
// === Activation Functions ===
//
float activate_sigmoid(float x) {
    x = clamp(x, -20.0, 20.0);
    x = 1.0 / (1.0 + exp(-x));
    return x;
}

float activate_tanh(float x) {
    x = clamp(x, -20.0, 20.0);
    float e_pos = exp(x);
    float e_neg = exp(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

float activate_relu(float x) {
    return max(0.0, x);
}

float activate_leaky_relu(float x) {
    return x > 0.0 ? x : 0.01 * x;
}

//
// === Softmax for a single sample in one layer ===
//
void softmax_layer(uint sample_idx, uint out_size, uint out_off) {
    // 1. Find max logit (to avoid overflow)
    float max_val = -3.402823e38; // -FLT_MAX
    for (uint i = 0u; i < out_size; ++i) {
        float v = intermediates[out_off + sample_idx * out_size + i];
        max_val = max(max_val, v);
    }

    // 2. Exponentiate shifted values & find sum
    float sum_exp = 0.0;
    for (uint i = 0u; i < out_size; ++i) {
        float e = exp(intermediates[out_off + sample_idx * out_size + i] - max_val);
        intermediates[out_off + sample_idx * out_size + i] = e;
        sum_exp += e;
    }

    // 3. Normalize
    float inv_sum = 1.0 / sum_exp;
    for (uint i = 0u; i < out_size; ++i) {
        intermediates[out_off + sample_idx * out_size + i] *= inv_sum;
    }
}

float apply_activation(float x, uint layer_idx) {
    uint act_type = activation_types[layer_idx];

    if (act_type == ACT_LINEAR)      return x;
    if (act_type == ACT_SIGMOID)     return activate_sigmoid(x);
    if (act_type == ACT_TANH)        return activate_tanh(x);
    if (act_type == ACT_RELU)        return activate_relu(x);
    if (act_type == ACT_LEAKY_RELU)  return activate_leaky_relu(x);
    // if (act_type == ACT_SOFTMAX)  // Typically applied across vector, not per neuron

    return x; // Fallback
}


//
// === Main Kernel ===
//
void main() {
    uint global_id = gl_GlobalInvocationID.x;

    uint batch = batch_size;
    uint neurons_first_layer = output_sizes[0];

    uint sample_idx = global_id / neurons_first_layer;
    uint neuron_idx = global_id % neurons_first_layer;

    if (sample_idx >= batch) {
        return;
    }

    for (uint layer_idx = 0u; layer_idx < layer_count; ++layer_idx) {
        uint in_size  = input_sizes[layer_idx];
        uint out_size = output_sizes[layer_idx];
        uint w_off = weight_offsets[layer_idx];
        uint b_off = bias_offsets[layer_idx];
        uint out_off_layer = interm_offsets[layer_idx];

        if (neuron_idx < out_size) {
            float sum = biases[b_off + neuron_idx];
            for (uint i = 0u; i < in_size; ++i){
                float in_val;
                if (layer_idx == 0u) {
                    in_val = inputs[sample_idx * in_size + i];
                } else {
                    uint prev_off = interm_offsets[layer_idx - 1u];
                    in_val = intermediates[prev_off + sample_idx * in_size + i];
                }
                float w_val = weights[w_off + neuron_idx * in_size + i];
                sum += in_val * w_val;
            }

            // for everything but softmax apply immediatly
            if (activation_types[layer_idx] != ACT_SOFTMAX){
                float activated = apply_activation(sum, layer_idx);
                intermediates[out_off_layer + sample_idx * out_size + neuron_idx] = activated;
            } else {
                // Store raw logits for softmax later
                intermediates[out_off_layer + sample_idx * out_size + neuron_idx] = sum;
            }
           
        }

        // === softmax stage ===
        if (activation_types[layer_idx] == ACT_SOFTMAX) {
            // let only one thread per sample handle normalization
            if (neuron_idx == 0u) {
                softmax_layer(sample_idx, out_size, out_off_layer);
            }
        }

        neurons_first_layer = out_size;
        neuron_idx = global_id % neurons_first_layer;
    }
}
