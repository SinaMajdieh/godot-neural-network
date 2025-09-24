extends Button
# DirectoryChooseButton.gd
# Custom button that opens a FileDialog for selecting a directory.


# Reference to the FileDialog that will handle folder selection
var dir_dialog: FileDialog

# Signal emitted when the user chooses a directory
signal directory_chosen(path: String)

func _ready() -> void:
	# Create and configure FileDialog dynamically (no need to add in scene)
	dir_dialog = FileDialog.new()
	dir_dialog.name = "DirDialog"

	# Allow access to full system paths, not just res://
	dir_dialog.access = FileDialog.ACCESS_FILESYSTEM

	# Set directory selection mode
	dir_dialog.file_mode = FileDialog.FILE_MODE_OPEN_DIR

	# Optional: start browsing at user's home directory
	dir_dialog.current_dir = FileUtils.get_home_directory()

	# Connect the selection signal to internal handler
	dir_dialog.dir_selected.connect(_on_directory_selected)

	# Add dialog as child of the button so it’s part of the same scene
	add_child(dir_dialog)

	# Connect the button click to opening the dialog
	pressed.connect(_on_button_pressed)


# Open the directory picker when the button is pressed.
func _on_button_pressed() -> void:
	dir_dialog.popup_centered()


# Handle user directory choice.
# Emit the directory_chosen signal for external use.
func _on_directory_selected(path: String) -> void:
	text = path
	directory_chosen.emit(path)
