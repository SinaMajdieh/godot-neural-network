extends Control
class_name EpochMetricGraph

# === Constants ===
const DEFAULT_LINE_COLOR: Color = Color("61AFEF")	# Default line color for graph

# === Exported Settings ===
@export var max_points: int = 200					# Max number of points to store
@export var line_color: Color = DEFAULT_LINE_COLOR	# Curve color
@export var line_width: float = 3.0				# Curve thickness
@export var hover_threshold: float = 6.0			# Pixel distance for hover detection
@export var grid: GraphGrid

# === Internal State ===
var loss_values: Array[float] = []					# Stored loss values in order
var _hover_index: int = -1							# Currently hovered point index (-1 = none)

# === Public API ===
func add_metric(value: float) -> void:
	loss_values.append(value)
	if loss_values.size() > max_points:				# Keep fixed-size buffer
		loss_values.pop_front()
	queue_redraw()									# Forces _draw() on next frame

func clear() -> void:
	loss_values.clear()
	queue_redraw()

# === Drawing ===
func _draw() -> void:
	if loss_values.size() < 2:
		return										# Need at least 2 points to draw a segment

	var loss_range: float = _get_loss_range()		# Vertical scaling factor
	var step_x: float = size.x / max(loss_values.size() - 1, 1)	# Horizontal spacing per point
	var segments: Array[Vector2] = _build_segments(step_x, loss_range)

	draw_multiline(segments, line_color, line_width, true)	# Draw connected line segments

# === Event Handling ===
func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		_update_hover_value(event.position)			# Track mouse hover position

# === Hover Logic ===
func _update_hover_value(mouse_position: Vector2) -> void:
	if loss_values.is_empty():
		_hover_index = -1
		return

	# Horizontal position to fractional index
	var step_x: float = size.x / max(loss_values.size() - 1, 1)
	var fractional_index: float = clamp(mouse_position.x / step_x, 0.0, float(loss_values.size() - 1))

	# Nearest two indices for interpolation
	var left_index: int = int(floor(fractional_index))
	var right_index: int = min(left_index + 1, loss_values.size() - 1)
	var factor: float = fractional_index - float(left_index)

	# Vertical position of curve at mouse X (interpolated between two points)
	var y_at_cursor: float = _get_interpolated_y(left_index, right_index, factor)

	# Hover only if mouse is close to actual curve Y position
	_hover_index = left_index if _is_within_vertical_threshold(mouse_position.y, y_at_cursor) else -1
	get_tooltip()			# Trigger tooltip update

func _is_within_vertical_threshold(mouse_y: float, line_y: float) -> bool:
	return abs(mouse_y - line_y) <= hover_threshold

# === Data / Calculation Helpers ===
func _build_segments(step_x: float, loss_range: float) -> Array[Vector2]:
	var segments: Array[Vector2] = []
	var min_loss: float = loss_values.min()

	var prev_point: Vector2 = Vector2.ZERO
	for i: int in range(loss_values.size()):
		# Map loss value to normalized Y coordinate
		# Formula: bottom - ((loss - min_loss) / range) * height
		var x: float = step_x * i
		var y: float = size.y - ((loss_values[i] - min_loss) / loss_range) * size.y
		var current_point: Vector2 = Vector2(x, y)

		if i > 0:
			segments.append(prev_point)				# Previous point in segment
			segments.append(current_point)			# Current point in segment

		prev_point = current_point

	return segments

func _get_loss_range() -> float:
	var min_loss: float = loss_values.min()
	var max_loss: float = loss_values.max()
	var loss_range: float = max_loss - min_loss
	if loss_range == 0:
		loss_range = 1								# Prevent division by zero
	return loss_range

func _get_interpolated_y(left_index: int, right_index: int, t: float) -> float:
	var loss_range: float = _get_loss_range()
	var min_loss: float = loss_values.min()

	var y_left: float = size.y - ((loss_values[left_index] - min_loss) / loss_range) * size.y
	var y_right: float = size.y - ((loss_values[right_index] - min_loss) / loss_range) * size.y

	return lerp(y_left, y_right, t)					# Linear interpolate between two points

# === Tooltip ===
func _get_tooltip(_at_position: Vector2) -> String:
	if _hover_index >= 0:
		return "Loss at epoch %d: %2.4f" % [_hover_index, loss_values[_hover_index]]
	return ""
