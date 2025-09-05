#version 450
layout(local_size_x = 64) in;

//
// Input and Output Buffers
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

layout(std430, binding = 3) buffer OutputBuffer {
    float outputs[];
};

//
// Uniforms for Layer Configuration
//
layout(std140, binding = 4) uniform LayerMeta {
    uint input_size;        // Number of inputs per vector
    uint output_size;       // Number of neurons in the layer
    uint num_vectors;       // Number of input vectors (batch size)
};

//
// Applies the tanh activation function.
// Used to introduce non-linearity in neuron output.
//
float activate_tanh(float x) {
    float e_pos = exp(x);
    float e_neg = exp(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

//
// Computes the weighted sum for a single neuron in a given input vector.
// Adds bias and multiplies each input by its corresponding weight.
//
float compute_neuron_output(uint vector_index, uint neuron_index) {
    float sum = biases[neuron_index];

    for (uint input_index = 0u; input_index < input_size; ++input_index) {
        uint input_offset = vector_index * input_size + input_index;
        uint weight_offset = neuron_index * input_size + input_index;

        float input_value = inputs[input_offset];
        float weight_value = weights[weight_offset];

        sum += input_value * weight_value;
    }

    return sum;
}

//
// Main compute function.
// Each thread computes the output of one neuron in one input vector.
//
void main() {
    uint global_id = gl_GlobalInvocationID.x;

    // Determine which input vector and neuron this thread is responsible for
    uint vector_index = global_id / output_size;
    uint neuron_index = global_id % output_size;

    // Skip out-of-range threads
    if (vector_index >= num_vectors) {
        return;
    }

    // Compute weighted sum and apply activation
    float raw_output = compute_neuron_output(vector_index, neuron_index);
    float activated_output = activate_tanh(raw_output);

    // Write result to output buffer
    uint output_offset = vector_index * output_size + neuron_index;
    outputs[output_offset] = activated_output;
}