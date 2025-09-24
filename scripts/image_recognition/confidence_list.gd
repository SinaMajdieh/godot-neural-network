extends VBoxContainer
class_name ConfidenceList

## Displays confidence scores for each possible class label, sorted descending.
## WHY: Gives a clear visual readout of model certainty for each prediction.

# -------------------------------------------------------------------
# Exported References
# -------------------------------------------------------------------
@export var ref_label: Label
## WHY: Template label to duplicate for each class confidence entry.

# -------------------------------------------------------------------
# Internal State
# -------------------------------------------------------------------
var labels: Array[Label]
var confidence_list: Array[Dictionary]

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
var refrence: Array[String] = [
	"Zero", "One", "Two", "Three", "Four", "Five",
	"Six", "Seven", "Eight", "Nine"
]
## WHY: Fixed class names for display — indexes match model output order.

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Public Methods
# -------------------------------------------------------------------
## Accepts model confidence vector, sorts, and displays.
## WHY: Updates the UI to represent the latest inference results.
func show_confidence_list(confidence: PackedFloat32Array) -> void:
	confidence_list.clear()
	_create_confidence_list(confidence)
	_sort_confidence_list()
	_show()

# -------------------------------------------------------------------
# Private Methods
# -------------------------------------------------------------------
## Builds confidence_list from raw float scores with class references.
func _create_confidence_list(confidence: PackedFloat32Array) -> void:
	for i: int in range(confidence.size()):
		confidence_list.append({
			"value": confidence[i],
			"ref": refrence[i]
		})

## Sorts confidence_list in-place by descending score.
func _sort_confidence_list() -> void:
	confidence_list.sort_custom(_custom_sort)

## Comparator for sort_custom — higher scores come first.
func _custom_sort(a: Dictionary, b: Dictionary) -> bool:
	return a["value"] > b["value"]

## Displays sorted confidence entries in labels.
func _show() -> void:
	assert(confidence_list.size() == labels.size())
	for i: int in range(confidence_list.size()):
		var item: Dictionary = confidence_list[i]
		labels[i].text = "%-5s: %2.2f" % [item["ref"], item["value"] * 100]

## Creates label instances for each class reference.
func _initial_labels() -> void:
	for i: int in range(refrence.size()):
		var label: Label = ref_label.duplicate()
		add_child(label)
		label.show()
		labels.append(label)

# Sets the refrence list
func set_refrence_list(new_refrence: Array[String]) -> void:
	refrence = new_refrence
	_initial_labels()
