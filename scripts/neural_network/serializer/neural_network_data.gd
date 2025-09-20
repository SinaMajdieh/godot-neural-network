# NetworkData.gd
# Resource container for entire neural network configuration.
# Why: Keeps layer parameters and activation types together for
#      easier save/load of trained models.

extends Resource
class_name NetworkData

# Ordered list of layers that make up the network.
@export var layers: Array[LayerData]

# Activation function type for each layer's output.
@export var activations: Array[Activations.Type]
