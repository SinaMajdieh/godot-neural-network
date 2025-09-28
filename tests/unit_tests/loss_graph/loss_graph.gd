extends Control

@export var animation_duration: float = 1.5
@export var data_points: int = 90
@export var inital_loss: float = 2.5

func _ready() -> void:
	var loss_panel: EpochMetricGraphPanel = EpochMetricGraphPanel.new_panel()
	add_child(loss_panel)

	var loss: float = inital_loss
	# Example: simulate adding loss values
	for i: int in range(data_points):
		await get_tree().create_timer(animation_duration / data_points).timeout
		loss_panel.add_metric(loss, i)
		loss = get_next_loss_non_linear(loss)
		

func get_next_loss_non_linear(loss: float) -> float:
	# Random gear factor for non-linear speed
	var decay_factor: float = 0.90 + randf() * 0.07 # between 0.90 and 0.97

	# Slow down as loss gets small (non-linear curve)
	var speed_modifier: float = clamp(loss, 0.05, 1.0) # higher loss = faster drop

	# Calculate next value
	loss = loss * (decay_factor * speed_modifier + (1.0 - speed_modifier)) 

	# Add small random noise
	var noise: float= (randf() - 0.5) * 0.02  # ±0.01 random
	loss = max(0.0, loss + noise)

	return loss