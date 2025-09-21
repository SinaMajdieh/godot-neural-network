# PredictImage.gd
# Control-based test node for running inference on a single image.
# Why: Quick visual and console validation of trained networks.

extends Control

@export_global_file var network_path: String
@export_global_file var test_image: String
@export var target: int

# Runs once when scene is ready to perform prediction.
func _ready() -> void:
	var forward_runner: ForwardPassRunner = ForwardPassRunner.new(ConfigKeys.SHADERS_PATHS.FORWARD_PASS)
	var network: NeuralNetwork = NeuralNetworkSerializer.import(
		forward_runner, network_path
	)

	var input: Array[PackedFloat32Array] = [
		ImageUtils.read_image(test_image, 0.25)
	]

	_show_input_image(input[0])

	var result: PackedFloat32Array = network.forward_pass(input)
	var prediction: int = ModelEvaluator.find_max_value_index(result)
	print("The image is a %d" % prediction)

	if target == prediction:
		print_rich("[color=green]Correct!")
	else:
		print_rich("[color=red]Wrong!")

# Displays the model input image in the TextureRect node.
func _show_input_image(data: PackedFloat32Array) -> void:
	var image: Image = ImageUtils.image_from_f32_array(data, 32, 32)
	var texture: Texture = ImageTexture.create_from_image(image)
	$TextureRect.texture = texture
