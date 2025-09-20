class_name FileUtils
extends RefCounted

# Provides reusable helper functions for listing files and folders.
# Why: Keeps file I/O logic separate from image processing or evaluation.

# Lists all non-directory file paths in the given directory.
# Params:
#   path (String): Path to directory to scan.
# Returns:
#   PackedStringArray: List of file paths.
static func list_files(path: String) -> PackedStringArray:
	var files: PackedStringArray = []
	var dir: DirAccess = DirAccess.open(path)
	if dir == null:
		push_error("Failed to open directory: %s" % path)
		return files

	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if not dir.current_is_dir():
			files.append(path + "/" + file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	return files


# Lists all subdirectory paths within the given directory.
# Params:
#   path (String): Path to directory to scan.
# Returns:
#   PackedStringArray: List of subdirectory paths.
static func list_dirs(path: String) -> PackedStringArray:
	var dirs: PackedStringArray = []
	var dir: DirAccess = DirAccess.open(path)
	if dir == null:
		push_error("Failed to open directory: %s" % path)
		return dirs

	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if dir.current_is_dir():
			dirs.append(path + "/" + file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	return dirs
