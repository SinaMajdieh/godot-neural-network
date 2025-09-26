extends Control
class_name PointPlotPanel
## UI panel wrapping the PointPlot class into a Control-compatible widget.
## Can be instantiated and added directly to any scene.

const PANEL_SCENE: String = "res://examples/visual_training/plot_panel.tscn"

# === Exported properties ===
@export var graph_node: PointPlot                # Panel's Plot node instance
@export var boundary: ClassificationBoundary
@export var title_label: Label
@export var x_value_label: Label
@export var y_value_label: Label

var title: String:
	set(value):
		title = value
		title_label.text = title

# === Constructor ===
static func new_panel(
	title_: String = "Scatter Plot",
	max_points_: int = 200,
	point_radius_: float = PointPlot.DEFAULT_POINT_RADIUS
) -> PointPlotPanel:
	var packed: PackedScene = load(PANEL_SCENE)
	var panel_instance: PointPlotPanel = packed.instantiate() as PointPlotPanel
	panel_instance.graph_node.max_points = max_points_
	panel_instance.graph_node.point_radius = point_radius_
	panel_instance.title = title_
	return panel_instance

# === Public API ===
## Add a point with optional color; updates the displayed X/Y labels.
func add_point(point_position: Vector2, color: Color = PointPlot.DEFAULT_POINT_COLOR) -> void:
	graph_node.add_point(point_position, color)
	x_value_label.text = "X: %6.2f" % point_position.x
	y_value_label.text = "Y: %6.2f" % point_position.y

## Clears the plot and resets value labels.
func clear() -> void:
	graph_node.clear()
	x_value_label.text = ""
	y_value_label.text = ""
