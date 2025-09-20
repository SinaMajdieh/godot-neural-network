# NeuralNetworkSerializer.gd
# Handles save/load of trained neural networks.
# Why: Centralizes serialization logic for easy persistence/reuse.

extends RefCounted
class_name NeuralNetworkSerializer

# ----- Export -----

# Converts a single network layer to a LayerData resource.
static func export_layer(layer: NetworkLayer) -> LayerData:
	var data: LayerData = LayerData.new()
	data.weight_matrix = layer.get_weight_matrix()
	data.bias_vector = layer.get_bias_vector()
	data.input_size = layer.input_size
	data.output_size = layer.output_size
	return data

# Converts an entire network to a NetworkData resource.
static func export_network(network: NeuralNetwork) -> NetworkData:
	var res: NetworkData = NetworkData.new()
	res.activations = network.layers_activation
	for layer: NetworkLayer in network.layers:
		res.layers.append(export_layer(layer))
	return res

# Saves a network resource to disk.
static func export(network: NeuralNetwork, path: String) -> void:
	var res: NetworkData = export_network(network)
	var err: int = ResourceSaver.save(res, path)

	if err != OK:
		push_error(
			"Failed to save network to %s (Error: %s)" % [path, err]
		)
		return
	print("Network saved to %s" % path)

# ----- Import -----

# Loads a network resource from disk and assigns a shader runner.
static func import(
	runner: ShaderRunner,
	path: String
) -> NeuralNetwork:
	var res: Resource = load(path)
	if res == null:
		push_error("Failed to load network from: %s" % path)
		return null
	if not (res is NetworkData):
		push_error(
			"Loaded file is not NetworkData: %s" % path
		)
		return null

	var net: NeuralNetwork = import_network(res)
	net.runner = runner
	return net

# Converts a NetworkData resource back into a NeuralNetwork.
static func import_network(res: NetworkData) -> NeuralNetwork:
	var net: NeuralNetwork = NeuralNetwork.new({})
	net.layers_activation = res.activations
	for layer_data: LayerData in res.layers:
		var layer: NetworkLayer = import_layer(layer_data)
		net.layers.append(layer)
	return net

# Creates a NetworkLayer from stored LayerData with shape validation.
static func import_layer(layer_data: LayerData) -> NetworkLayer:
	# Validate weight matrix outer size == output_size
	if layer_data.weight_matrix.size() != layer_data.output_size:
		push_error(
			"Weight matrix row count mismatch: expected %d, got %d" %
			[layer_data.output_size, layer_data.weight_matrix.size()]
		)
		return null

	# Validate weight matrix inner size == input_size
	for row: Array in layer_data.weight_matrix:
		if row.size() != layer_data.input_size:
			push_error(
				"Weight matrix column count mismatch: expected %d, got %d" %
				[layer_data.input_size, row.size()]
			)
			return null

	# Validate bias vector length == output_size
	if layer_data.bias_vector.size() != layer_data.output_size:
		push_error(
			"Bias vector size mismatch: expected %d, got %d" %
			[layer_data.output_size, layer_data.bias_vector.size()]
		)
		return null

	var layer: NetworkLayer = NetworkLayer.new(
		layer_data.input_size,
		layer_data.output_size
	)
	layer.set_weight_matrix(layer_data.weight_matrix)
	layer.set_bias_vector(layer_data.bias_vector)
	return layer
