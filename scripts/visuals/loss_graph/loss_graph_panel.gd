extends Control
class_name LossGraphPanel
## UI panel wrapping the LossGraph class into a Control-compatible widget.
## Can be instantiated and added directly to any scene.

const PANEL_SCENE: String = "res://scripts/visuals/loss_graph/loss_graph_panel.tscn"

# === Exported properties ===
@export var graph_node: LossGraph    # Panel's Graph     
@export var title_label: Label
@export var loss_value_label: Label
@export var epoch_value_label: Label

var title: String :
	set(value):
		title = value
		title_label.text = title
# === Constructor ===
static func new_panel(
	title_: String = "Loss over epochs",
	max_points_: int = 200, 
	line_color_: Color = LossGraph.DEFAULT_LINE_COLOR
) -> LossGraphPanel:
	var packed: PackedScene = load(PANEL_SCENE)
	var panel_instance: LossGraphPanel = packed.instantiate() as LossGraphPanel
	panel_instance.graph_node.max_points = max_points_
	panel_instance.graph_node.line_color = line_color_
	panel_instance.title = title_
	return panel_instance

# === Public API ===
func add_loss(value: float, epoch: int) -> void:
	graph_node.add_loss(value)
	loss_value_label.text = "Loss: %3.3f" % value
	epoch_value_label.text = "Epoch: %6d" % epoch

func clear() -> void:
	graph_node.clear()
	loss_value_label.text = ""
