#version 450
#extension GL_EXT_shader_atomic_float : enable
#define ACT_LINEAR       0
#define ACT_SIGMOID      1
#define ACT_TANH         2
#define ACT_RELU         3
#define ACT_LEAKY_RELU   4
#define ACT_SOFTMAX      5

layout(local_size_x = 64) in;

//
// Input Buffers
//
layout(std430, binding = 0) buffer PreActivationBuffer {
    float pre_acts[];
};

layout(std430, binding = 1) buffer ErrorBuffer {
    float errors[];
};

layout(std430, binding = 2) buffer InputBuffer {
    float inputs[];
};

//
// Activation Type Buffer (per layer)
//
layout(std430, binding = 3) buffer ActivationTypeBuffer {
    uint activation_types[];
};

//
// Output Buffers for Gradients
//
layout(std430, binding = 4) buffer WeightGradientBuffer {
    float weight_gradients[];
};

layout(std430, binding = 5) buffer BiasGradientBuffer {
    float bias_gradients[];
};

//
// Uniforms for Layer Configuration
//
layout(std140, binding = 6) uniform LayerMeta {
    uint input_size;        // Number of inputs per vector
    uint output_size;       // Number of neurons in the layer
    uint num_vectors;       // Number of input vectors (batch size)
};

// Activation functions (recomputed here for derivative)
// To avoid storing both a and z, we derive a from z locally when needed
float sigmoid_from_z(float z) {
    return 1.0 / (1.0 + exp(-z));
}
float tanh_from_z(float z) {
    float e_pos = exp(z);
    float e_neg = exp(-z);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

// Derivatives using z
float compute_sigmoid_derivative_from_z(float z) {
    float s = sigmoid_from_z(z);
    return s * (1.0 - s);
}

float compute_tanh_derivative_from_z(float z) {
    float t = tanh_from_z(z);
    return 1.0 - t * t;
}

float compute_relu_derivative_from_z(float z) {
    return z > 0.0 ? 1.0 : 0.0;
}

float compute_leaky_relu_derivative_from_z(float z) {
    return z > 0.0 ? 1.0 : 0.01;
}

float compute_derivative(float z, uint act_type) {
    if (act_type == ACT_SIGMOID)     return compute_sigmoid_derivative_from_z(z);
    if (act_type == ACT_TANH)        return compute_tanh_derivative_from_z(z);
    if (act_type == ACT_RELU)        return compute_relu_derivative_from_z(z);
    if (act_type == ACT_LEAKY_RELU)  return compute_leaky_relu_derivative_from_z(z);
    if (act_type == ACT_SOFTMAX)     return 1.0; // handled in loss for stability
    return 1.0;
}


//
// Computes the gradient (delta) for a single neuron in a given input vector.
// Combines error signal with activation derivative.
//
float compute_neuron_delta(uint vector_index, uint neuron_index, uint act_type) {
    uint offset = vector_index * output_size + neuron_index;
    float z = pre_acts[offset]; // now reading z
    float error = errors[offset];
    float derivative = compute_derivative(z, act_type);
    return error * derivative;
}

//
// Accumulates weight gradients for a single neuron across all inputs.
// Uses atomicAdd to safely update shared gradient buffer.
//
void accumulate_weight_gradients(uint vector_index, uint neuron_index, float delta) {
    for (uint input_index = 0u; input_index < input_size; ++input_index) {
        uint input_offset = vector_index * input_size + input_index;
        uint grad_offset = neuron_index * input_size + input_index;

        float input_value = inputs[input_offset];
        float gradient_contribution = input_value * delta;

        atomicAdd(weight_gradients[grad_offset], gradient_contribution);
    }
}

//
// Accumulates bias gradient for a single neuron.
// Uses atomicAdd to safely update shared bias buffer.
//
void accumulate_bias_gradient(uint neuron_index, float delta) {
    atomicAdd(bias_gradients[neuron_index], delta);
}

//
// Main compute function.
// Each thread processes one neuron in one input vector.
// Computes delta and accumulates gradients.
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

    uint act_type = activation_types[0]; // Assuming one layer at a time
    // Compute delta and accumulate gradients
    float delta = compute_neuron_delta(vector_index, neuron_index, act_type);
    // delta = clamp(delta, -1.0, 1.0);
    accumulate_weight_gradients(vector_index, neuron_index, delta);
    accumulate_bias_gradient(neuron_index, delta);
}