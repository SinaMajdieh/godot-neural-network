extends RefCounted
class_name DataSplit

##
## Container class for train/test split results
##

var train_inputs: Array[PackedFloat32Array] = []
var train_targets: Array[PackedFloat32Array] = []
var test_inputs: Array[PackedFloat32Array] = []
var test_targets: Array[PackedFloat32Array] = []

# Optional helper methods
func get_train_size() -> int:
    return train_inputs.size()

func get_test_size() -> int:
    return test_targets.size()

func summary() -> void:
    print("Train size: %d" % get_train_size())
    print("Test size: %d" % get_test_size())
