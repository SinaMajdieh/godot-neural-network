extends RefCounted
class_name DataSetUtils

##
## Utility class for dataset-level operations such as train/test splitting and CSV loading.
##

##
## Splits input and target data into training and testing sets.
##
## @param inputs Array of input vectors
## @param targets Array of target vectors
## @param test_ratio Fraction of data to allocate to test set (0.0–1.0)
## @return DataSplit containing train and test subsets
##
static func train_test_split(
    inputs: Array[PackedFloat32Array],
    targets: Array[PackedFloat32Array],
    test_ratio: float
) -> DataSplit:
    assert(inputs.size() == targets.size())
    assert(test_ratio >= 0.0 and test_ratio <= 1.0)

    var total_size: int = inputs.size()
    var test_size: int = int(total_size * test_ratio)

    var indices: Array[int] = []
    for i: int in range(total_size):
        indices.append(i)
    indices.shuffle()

    var split: DataSplit = DataSplit.new()

    for i: int in range(total_size):
        var idx: int = indices[i]
        if i < test_size:
            split.test_inputs.append(inputs[idx])
            split.test_targets.append(targets[idx])
        else:
            split.train_inputs.append(inputs[idx])
            split.train_targets.append(targets[idx])

    return split

##
## Loads a CSV file and returns an array of float vectors.
##
## @param path File path to CSV
## @param vector_size Number of values per row
## @param skip_head Whether to skip the first line (header)
## @return Array of PackedFloat32Array vectors
##
static func load_csv_as_batches(path: String, vector_size: int, skip_head: bool = true) -> Array[PackedFloat32Array]:
    var result: Array[PackedFloat32Array] = []
    var file: FileAccess = FileAccess.open(path, FileAccess.READ)
    if file == null:
        push_error("Failed to open file: %s" % path)
        return result

    while not file.eof_reached():
        var line: String = file.get_line().strip_edges()
        if skip_head:
            skip_head = false
            continue
        if line == "":
            continue

        var tokens: PackedStringArray = line.split(",")
        if tokens.size() != vector_size:
            push_error("Line has incorrect number of values: %s" % line)
            continue

        var vec: PackedFloat32Array = PackedFloat32Array()
        for token: String in tokens:
            vec.append(token.to_float())
        result.append(vec)

    file.close()
    return result

##
## Loads a CSV file and returns a single flattened float array.
##
## @param path File path to CSV
## @param skip_head Whether to skip the first line (header)
## @return Flattened PackedFloat32Array
##
static func load_csv_as_flat_array(path: String, skip_head: bool = true) -> PackedFloat32Array:
    var flat: PackedFloat32Array = PackedFloat32Array()
    var file: FileAccess = FileAccess.open(path, FileAccess.READ)
    if file == null:
        push_error("Failed to open file: %s" % path)
        return flat

    while not file.eof_reached():
        var line: String = file.get_line().strip_edges()
        if skip_head:
            skip_head = false
            continue
        if line == "":
            continue

        var tokens: PackedStringArray = line.split(",")
        for token: String in tokens:
            flat.append(token.to_float())

    file.close()
    return flat
