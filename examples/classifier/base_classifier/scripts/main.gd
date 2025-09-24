extends Control

## Buttons and selectors for running forward passes on chosen inputs.
## WHY: Provides a simple evaluation UI to inspect trained network predictions.

# -------------------------------------------------------------------
# Exported UI References
# -------------------------------------------------------------------
@export var network_button: Button
## WHY: Triggers loading of a serialized neural network.

@export var input_folder_button: Button
## WHY: Opens folder dialog to load dataset samples visually.

@export var item_select: InputSelector
## WHY: Displays loaded images as selectable items.

@export var input_button: Button
## WHY: Shows the icon and prediction result after running forward pass.

@export var confidence_list: ConfidenceList
## WHY: Shows the confidence and probabilities of each outcome after running forward pass.

@export var refrence_list: Array[String] = [
	"Zero", "One", "Two", "Three", "Four", "Five",
	"Six", "Seven", "Eight", "Nine"
]
## WHY: Maps each probability to a refrence text.

@export var image_scale: float = 0.25
## WHY: proper image scale to feed the network.

@export var invert_image: bool = false
## WHY: invert image or not

# -------------------------------------------------------------------
# Internal State
# -------------------------------------------------------------------
var forward_runner: ForwardPassRunner
var network: NeuralNetwork

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------
func _ready() -> void:
	## WHY: Connects UI events to handlers for network loading, input loading, and selection.
	network_button.file_chosen.connect(load_network)
	input_folder_button.directory_chosen.connect(load_inputs)
	item_select.input_selected.connect(pass_input)
	confidence_list.set_refrence_list(refrence_list)

# -------------------------------------------------------------------
# Network Loading
# -------------------------------------------------------------------
func load_network(path: String) -> void:
	## WHY: Instantiates a forward runner and imports a serialized network for inference.
	forward_runner = ForwardPassRunner.new()
	network = NeuralNetworkSerializer.import(forward_runner, path)

# -------------------------------------------------------------------
# Inputs Loading
# -------------------------------------------------------------------
func load_inputs(path: String) -> void:
	## WHY: Loads dataset images into the ItemList for selection.
	item_select.load_directory(path, image_scale, invert_image)

# -------------------------------------------------------------------
# Forward Pass & UI Update
# -------------------------------------------------------------------
func pass_input(input_meta: Dictionary) -> void:
	## WHY: Runs a forward pass on the selected image data and updates
	##      the button to display the prediction visually.
	var image: Image = Image.load_from_file(input_meta["PATH"])
	var icon: Texture2D = ImageTexture.create_from_image(image)

	var result: PackedFloat32Array = network.forward_pass([input_meta["DATA"]])
	var prediction: int = ModelEvaluator.find_max_value_index(result)

	input_button.icon = icon
	input_button.text = "This is a %s" % refrence_list[prediction]
	
	confidence_list.show_confidence_list(result)
