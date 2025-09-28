extends Control
class_name EpochMetricGraphPanel
## UI panel wrapping the EpochMetricGraph class into a Control-compatible widget.
## Can be instantiated and added directly to any scene.

const PANEL_SCENE: String = "res://scripts/visuals/loss_graph/epoch_metric_graph_panel.tscn"

# === Exported properties ===
@export var graph_node: EpochMetricGraph    # Panel's Graph     
@export var title_label: Label
@export var metric_value_label: Label
@export var epoch_value_label: Label

var metric_name: String

var title: String :
	set(value):
		title = value
		title_label.text = title
# === Constructor ===
static func new_panel(
	title_: String = "Loss over epochs",
	metric_name_: String = "Loss",
	max_points_: int = 200, 
	line_color_: Color = EpochMetricGraph.DEFAULT_LINE_COLOR
) -> EpochMetricGraphPanel:
	var packed: PackedScene = load(PANEL_SCENE)
	var panel_instance: EpochMetricGraphPanel = packed.instantiate() as EpochMetricGraphPanel
	panel_instance.graph_node.max_points = max_points_
	panel_instance.graph_node.line_color = line_color_
	panel_instance.title = title_
	panel_instance.metric_name = metric_name_
	return panel_instance

# === Public API ===
func add_metric(value: float, epoch: int) -> void:
	graph_node.add_metric(value)
	metric_value_label.text = "%s: %3.3f" % [metric_name, value]
	epoch_value_label.text = "Epoch: %6d" % epoch

func clear() -> void:
	graph_node.clear()
	metric_value_label.text = ""
