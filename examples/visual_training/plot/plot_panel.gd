extends Control
class_name PointPlotPanel

# === Constants ===
const PANEL_SCENE: String = "res://examples/visual_training/plot/plot_panel.tscn"

# === Exported UI Components ===
@export var graph_node: PointPlot					# Plotting surface
@export var boundary: ClassificationBoundary		# Decision boundary overlay
@export var title_label: Label						# Panel title text
@export var loss_label: Label						# Displays current loss
@export var epoch_label: Label						# Displays current epoch
@export var decision_boundary_label: Label			# Displays render timing

# === Internal State ===
var title: String:
	set(value):
		# Why: Synchronize internal title string with UI label
		title = value
		title_label.text = title


# === Constructor ===
static func new_panel(
	title_: String = "Scatter Plot",
	max_points_: int = 200,
	point_radius_: float = PointPlot.DEFAULT_POINT_RADIUS
) -> PointPlotPanel:
	# Why: Load panel scene so it can be instantiated programmatically
	var packed: PackedScene = load(PANEL_SCENE)
	var panel_instance: PointPlotPanel = packed.instantiate() as PointPlotPanel

	# Why: Apply initial configuration before returning instance
	panel_instance.graph_node.max_points = max_points_
	panel_instance.graph_node.point_radius = point_radius_
	panel_instance.title = title_
	return panel_instance


# === Public API ===
func add_point(point_position: Vector2, color: Color = PointPlot.DEFAULT_POINT_COLOR) -> void:
	# Why: Add a point to the plot with an optional color override
	graph_node.add_point(point_position, color)


func clear() -> void:
	# Why: Reset plot and clear informational labels
	graph_node.clear()
	loss_label.text = ""
	epoch_label.text = ""


func update_epoch_info(loss: float, epoch: int) -> void:
	# Why: Show latest loss and epoch during training
	loss_label.text = "Loss: %3.3f" % loss
	epoch_label.text = "Epoch: %6d" % epoch


func update_decision_boundary_label(value: int) -> void:
	# Why: Show render timing for decision boundary updates
	decision_boundary_label.text = "Decision Boundary Rendering time : %d ms" % value
