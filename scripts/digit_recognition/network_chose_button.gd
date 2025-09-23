extends Button
# TresFileChooseButton.gd
# Custom button that opens a FileDialog for selecting a single `.tres` file.


# Reference to FileDialog used for file selection
var file_dialog: FileDialog

# Signal emitted when .tres file is chosen
signal file_chosen(path: String)


func _ready() -> void:
    # Create and configure FileDialog dynamically
    file_dialog = FileDialog.new()
    file_dialog.name = "FileDialog"

    # Allow access to the entire filesystem
    file_dialog.access = FileDialog.ACCESS_FILESYSTEM
    
    # Select single file mode
    file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE

    # Restrict to .tres files only
    file_dialog.filters = PackedStringArray(["*.tres ; Godot Resource Files"])
    
    # Optional: start from user's home directory
    file_dialog.current_dir = FileUtils.get_home_directory()

    # Connect the signal for file selection
    file_dialog.file_selected.connect(_on_file_selected)

    # Add dialog to this button’s node tree
    add_child(file_dialog)

    # Connect the button click to open the dialog
    pressed.connect(_on_button_pressed)


# Open the file picker when the button is pressed.
func _on_button_pressed() -> void:
    file_dialog.popup_centered()


# Called when the user chooses a file.
# Emits the chosen path through `file_chosen`.
func _on_file_selected(path: String) -> void:
    text = path
    file_chosen.emit(path)
