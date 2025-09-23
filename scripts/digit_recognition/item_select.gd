extends ItemList
class_name InputSelector

## Emitted when an item in the list is selected.
## WHY: Allows external UI/logic to respond to specific input selection.
signal input_selected(input: PackedFloat32Array)

# -------------------------------------------------------------------
# Directory Loading
# -------------------------------------------------------------------

## Loads all images from a given directory into the list.
## WHY: Provides quick visual selection of dataset samples for inspection/testing.
func load_directory(path: String) -> void:
	var images_path: PackedStringArray = FileUtils.list_files(path)
	var images: Array[PackedFloat32Array] = ImageUtils.read_images(path, 0.25)

	for i: int in range(images_path.size()):
		_add_image_item(
			"image-%d" % i,
			images_path[i],
			{
				"PATH": images_path[i],
				"DATA": images[i]
			}
		)

# -------------------------------------------------------------------
# Item Adding
# -------------------------------------------------------------------

## Adds an image as an item with metadata into the list.
## WHY: Stores both file path and raw image data for retrieval on selection.
func _add_image_item(
	label: String,
	image_path: String,
	meta: Dictionary
) -> void:
	var index: int = item_count
	var image: Image = Image.load_from_file(image_path)
	var icon: Texture2D = ImageTexture.create_from_image(image)

	add_item(label, icon)
	set_item_metadata(index, meta)

# -------------------------------------------------------------------
# Selection Handling
# -------------------------------------------------------------------

## Called when an item is selected from the list.
## WHY: Emits input_selected signal to hook into external logic.
func on_item_select(index: int) -> void:
	input_selected.emit(get_item_metadata(index))
