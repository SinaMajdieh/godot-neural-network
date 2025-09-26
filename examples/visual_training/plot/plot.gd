extends Control
class_name PointPlot

# === Supporting Types ===
class AxisRange:
	var min_value: float
	var max_value: float
	var range_size: float
	
	func _init(min_value_: float, max_value_: float) -> void:
		self.min_value = min_value_
		self.max_value = max_value_
		self.range_size = max_value - min_value
		if self.range_size == 0.0:
			self.range_size = 1.0

class PointData:
	var position: Vector2
	var color: Color
	
	func _init(position_: Vector2, color_: Color) -> void:
		self.position = position_
		self.color = color_

# === Constants ===
const DEFAULT_POINT_COLOR: Color = Color("61AFEF")           # Default point color
const DEFAULT_POINT_RADIUS: float = 4.0                      # Default point radius in pixels

# === Exported Settings ===
@export var max_points: int = 200                            # Max number of stored points
@export var point_radius: float = DEFAULT_POINT_RADIUS       # Point size in pixels
@export var hover_threshold: float = 6.0                     # Hover detection radius in pixels
@export var grid: GraphGrid                                  # Optional grid to draw behind plot

# === Internal State ===
var points: Array[PointData] = []                            # Stored data points (with colors)
var _hover_index: int = -1                                   # Index of hovered point (-1 if none)

# === Public API ===
## Add a new point with optional custom color
func add_point(point_position: Vector2, color: Color = DEFAULT_POINT_COLOR) -> void:
	var new_point: PointData = PointData.new(point_position, color)
	points.append(new_point)
	if points.size() > max_points:
		points.pop_front()
	queue_redraw()

func clear() -> void:
	points.clear()
	queue_redraw()

# === Drawing ===
func _draw() -> void:
	if points.is_empty():
		return
	
	var x_axis: AxisRange = _calculate_x_range()
	var y_axis: AxisRange = _calculate_y_range()
	
	for point_data: PointData in points:
		var plot_x: float = ((point_data.position.x - x_axis.min_value) / x_axis.range_size) * size.x
		var plot_y: float = size.y - ((point_data.position.y - y_axis.min_value) / y_axis.range_size) * size.y
		draw_circle(Vector2(plot_x, plot_y), point_radius, point_data.color)

# === Event Handling ===
func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		_update_hover(event.position)

# === Hover Detection ===
func _update_hover(mouse_pos: Vector2) -> void:
	if points.is_empty():
		_hover_index = -1
		return
	
	var closest_index: int = -1
	var closest_distance: float = hover_threshold
	var x_axis: AxisRange = _calculate_x_range()
	var y_axis: AxisRange = _calculate_y_range()
	
	for i: int in range(points.size()):
		var plot_x: float = ((points[i].position.x - x_axis.min_value) / x_axis.range_size) * size.x
		var plot_y: float = size.y - ((points[i].position.y - y_axis.min_value) / y_axis.range_size) * size.y
		var distance: float = mouse_pos.distance_to(Vector2(plot_x, plot_y))
		if distance <= closest_distance:
			closest_distance = distance
			closest_index = i
	
	_hover_index = closest_index

# === Range Calculation Helpers ===
func _calculate_x_range() -> AxisRange:
	var min_val: float = points[0].position.x
	var max_val: float = points[0].position.x
	for p: PointData in points:
		if p.position.x < min_val:
			min_val = p.position.x
		if p.position.x > max_val:
			max_val = p.position.x
	return AxisRange.new(min_val, max_val)

func _calculate_y_range() -> AxisRange:
	var min_val: float = points[0].position.y
	var max_val: float = points[0].position.y
	for p: PointData in points:
		if p.position.y < min_val:
			min_val = p.position.y
		if p.position.y > max_val:
			max_val = p.position.y
	return AxisRange.new(min_val, max_val)

# === Tooltip ===
func _get_tooltip(_pos: Vector2) -> String:
	if _hover_index >= 0:
		var p: PointData = points[_hover_index]
		return "Point #%d: (%.2f, %.2f)" % [
			_hover_index,
			p.position.x,
			p.position.y
		]
	return ""
