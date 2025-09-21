# ConfigKeys.gd
# Why: Central constants for configuring network, trainer, and shader
# paths. Keeps keys and defaults in one place to avoid hardcoding.

class_name ConfigKeys
extends RefCounted

const NETWORK: Dictionary = NeuralNetwork.KEYS
const TRAINER: Dictionary = Trainer.KEYS

const SHADERS_PATHS: Dictionary = {
	FORWARD_PASS = ForwardPassRunner.DEFAULT_SHADER_PATH,
	BACKWARD_PASS = BackwardPassRunner.DEFAULT_SHADER_PATH,
	GRADIENT_NORM = GradClipRunner.DEFAULT_SHADER_PATHS.NORM_SHADER_PATH,
	GRADIENT_SCALE = GradClipRunner.DEFAULT_SHADER_PATHS.SCALE_SHADER_PATH
}
