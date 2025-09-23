#version 450
// Activation codes
#define ACT_LINEAR       0
#define ACT_SIGMOID      1
#define ACT_TANH         2
#define ACT_RELU         3
#define ACT_LEAKY_RELU   4
#define ACT_SOFTMAX      5

layout(local_size_x = 256) in;

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
// Gradient Buffers — UINT backing
//
layout(std430, binding = 4) buffer WeightGradientBuffer {
    uint weight_gradients[];
};

layout(std430, binding = 5) buffer BiasGradientBuffer {
    uint bias_gradients[];
};

//
// Uniforms for layer metadata
//
layout(std140, binding = 6) uniform LayerMeta {
    uint input_size;        
    uint output_size;       
    uint num_vectors;       
    uint layer_index; // safer indexing for activation type
};

// --------------------------------------------------------
// Activation helpers
// --------------------------------------------------------
float sigmoid_from_z(float z) {
    return 1.0 / (1.0 + exp(-z));
}

float tanh_from_z(float z) {
    float e_pos = exp(z);
    float e_neg = exp(-z);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

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
    if (act_type == ACT_SOFTMAX)     return 1.0;
    return 1.0;
}

// --------------------------------------------------------
// CAS-loop atomic float on UINT buffer
// --------------------------------------------------------
void atomicAddWeight(uint index, float val) {
    if (index >= input_size * output_size) return;
    uint oldBits, newBits;
    float oldVal, newVal;
    do {
        oldBits = weight_gradients[index];
        oldVal  = uintBitsToFloat(oldBits);
        newVal  = oldVal + val;
        newBits = floatBitsToUint(newVal);
    } while (atomicCompSwap(weight_gradients[index], oldBits, newBits) != oldBits);
}

void atomicAddBias(uint index, float val) {
    if (index >= output_size) return;
    uint oldBits, newBits;
    float oldVal, newVal;
    do {
        oldBits = bias_gradients[index];
        oldVal  = uintBitsToFloat(oldBits);
        newVal  = oldVal + val;
        newBits = floatBitsToUint(newVal);
    } while (atomicCompSwap(bias_gradients[index], oldBits, newBits) != oldBits);
}

// --------------------------------------------------------
// Delta and gradient accumulation
// --------------------------------------------------------
float compute_neuron_delta(uint vector_index, uint neuron_index, uint act_type) {
    uint offset = vector_index * output_size + neuron_index;
    float z = pre_acts[offset];
    float error = errors[offset];
    float derivative = compute_derivative(z, act_type);
    return error * derivative;
}

void accumulate_weight_gradients(uint vector_index, uint neuron_index, float delta) {
    for (uint input_index = 0u; input_index < input_size; ++input_index) {
        uint input_offset = vector_index * input_size + input_index;
        uint grad_offset  = neuron_index * input_size + input_index;
        float input_value = inputs[input_offset];
        float gradient_contribution = input_value * delta;
        atomicAddWeight(grad_offset, gradient_contribution);
    }
}

void accumulate_bias_gradient(uint neuron_index, float delta) {
    atomicAddBias(neuron_index, delta);
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
void main() {
    uint global_id = gl_GlobalInvocationID.x;
    uint vector_index = global_id / output_size;
    uint neuron_index = global_id % output_size;

    if (vector_index >= num_vectors) return;

    uint act_type = activation_types[layer_index]; // safe activation fetch
    float delta = compute_neuron_delta(vector_index, neuron_index, act_type);

    accumulate_weight_gradients(vector_index, neuron_index, delta);
    accumulate_bias_gradient(neuron_index, delta);
}
