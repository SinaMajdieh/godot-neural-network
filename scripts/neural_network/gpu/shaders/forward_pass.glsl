#version 450
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

layout(std430, binding = 3) buffer IntermediatesBuffer {
    float intermediates[];
};

layout(std430, binding = 4) buffer MetaBuffer {
    uint layer_count;
    uint batch_size;
    uint input_sizes[32];
    uint output_sizes[32];
    uint weight_offsets[32];
    uint bias_offsets[32];
    uint interm_offsets[32];
};

//
// === Activation ===
//
float activate_tanh(float x) {
    float e_pos = exp(x);
    float e_neg = exp(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

float safe_activation(float x) {
    float act = activate_tanh(x);

    // Clamp if NaN or Inf
    if (isnan(act) || isinf(act)) {
        //return 0.0; // or use tanh(0.0) = 0.0
        // Optional: clamp extreme values
        return clamp(act, -10.0, 10.0);
    }

    // Optional: clamp extreme values
    return clamp(act, -10.0, 10.0);
}


//
// === Main Kernel ===
// Each thread -> 1 neuron for 1 sample in *first* layer
// Then loop over layers sequentially, reusing same thread ID logic
//
void main() {
    uint global_id = gl_GlobalInvocationID.x;

    uint batch = batch_size;
    uint neurons_first_layer = output_sizes[0];

    // Map thread to sample index and neuron index in first layer
    uint sample_idx = global_id / neurons_first_layer;
    uint neuron_idx = global_id % neurons_first_layer;

    if (sample_idx >= batch) {
        return;
    }

    // We'll process *each* layer serially per thread
    for (uint layer_idx = 0u; layer_idx < layer_count; ++layer_idx) {
        uint in_size  = input_sizes[layer_idx];
        uint out_size = output_sizes[layer_idx];

        // Only threads for valid neurons in this layer do work
        if (neuron_idx < out_size) {
            uint w_off = weight_offsets[layer_idx];
            uint b_off = bias_offsets[layer_idx];
            uint out_off_layer = interm_offsets[layer_idx];

            // Source data: for layer 0, from inputs[]
            // Otherwise, from intermediates[] of previous layer
            float sum = biases[b_off + neuron_idx];
            for (uint i = 0u; i < in_size; ++i) {
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

            intermediates[out_off_layer + sample_idx * out_size + neuron_idx] = safe_activation(sum);
        }

        // Update neuron index mapping for next layer
        // Every thread re-maps based on first layer indexing
        neurons_first_layer = out_size;
        neuron_idx = global_id % neurons_first_layer;
    }
}
