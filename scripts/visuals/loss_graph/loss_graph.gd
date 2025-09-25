extends Control
class_name LossGraph

const DEFAULT_LINE_COLOR: Color = Color("61AFEF")

# === Exported settings ===
@export var max_points: int = 200
@export var line_color: Color = DEFAULT_LINE_COLOR
@export var line_width: float = 1.0

# === Internal state ===
var loss_values: Array[float] = []

# === Public API ===
func add_loss(value: float) -> void:
    loss_values.append(value)
    if loss_values.size() > max_points:
        loss_values.pop_front()
    queue_redraw()   # triggers _draw()

func clear() -> void:
    loss_values.clear()
    queue_redraw()

func _build_segments(step_x: float, loss_range: float) -> Array[Vector2]:
    var segments: Array[Vector2] = []
    var min_loss: float = loss_values.min()
    # Build segment list directly
    var prev_point: Vector2 = Vector2.ZERO
    for i: int in range(loss_values.size()):
        var x: float = step_x * i
        var y: float = size.y - ((loss_values[i] - min_loss) / loss_range) * size.y
        var current_point: Vector2 = Vector2(x, y)
        
        if i > 0:
            segments.append(prev_point)
            segments.append(current_point)
        
        prev_point = current_point
    return segments

func _get_loss_range() -> float:
    var min_loss: float = loss_values.min()
    var max_loss: float = loss_values.max()
    var loss_range: float = max_loss - min_loss
    if loss_range == 0:
        loss_range = 1
    return loss_range

# === Drawing ===
func _draw() -> void:
    if loss_values.size() < 2:
        return

    var loss_range: float = _get_loss_range()
    var step_x: float = size.x / max(loss_values.size() - 1, 1)
    var segments: Array[Vector2] = _build_segments(step_x, loss_range)

    draw_multiline(segments, line_color, line_width, true)  # antialiasing = true
