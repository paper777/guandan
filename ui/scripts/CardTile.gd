extends Button
class_name CardTile

signal selection_changed(card_id: String, selected: bool)

var card_id: String = ""


func _ready() -> void:
	toggle_mode = true
	custom_minimum_size = Vector2(70, 98)
	focus_mode = Control.FOCUS_NONE
	toggled.connect(_on_toggled)


func set_card_id(value: String) -> void:
	card_id = value
	text = _display_text(value)
	tooltip_text = value


func set_selected(value: bool) -> void:
	button_pressed = value


func _on_toggled(selected: bool) -> void:
	selection_changed.emit(card_id, selected)


func _display_text(value: String) -> String:
	var parts := value.split("-")
	if parts.size() == 2:
		return "%s\n%s" % [parts[1], parts[0]]
	if parts.size() == 3:
		return "%s\n%s\n%s" % [parts[2], parts[1], parts[0]]
	return value
