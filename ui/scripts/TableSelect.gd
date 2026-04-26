extends Control

signal joined_table

var api
var session

var server_url_edit: LineEdit
var table_id_edit: LineEdit
var table_list: OptionButton
var seat_picker: OptionButton
var display_name_edit: LineEdit
var status_label: Label
var join_button: Button


func setup(api_client, session_state) -> void:
	api = api_client
	session = session_state


func _ready() -> void:
	_build_ui()
	server_url_edit.text = session.base_url
	display_name_edit.text = session.display_name


func _build_ui() -> void:
	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 32)
	margin.add_theme_constant_override("margin_top", 32)
	margin.add_theme_constant_override("margin_right", 32)
	margin.add_theme_constant_override("margin_bottom", 32)
	add_child(margin)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 14)
	margin.add_child(root)

	var title := Label.new()
	title.text = "Guandan Table"
	title.add_theme_font_size_override("font_size", 30)
	root.add_child(title)

	server_url_edit = LineEdit.new()
	server_url_edit.placeholder_text = "Server URL"
	root.add_child(_field("Server", server_url_edit))

	var table_row := HBoxContainer.new()
	table_row.add_theme_constant_override("separation", 8)
	table_id_edit = LineEdit.new()
	table_id_edit.placeholder_text = "table-id"
	table_id_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	table_row.add_child(table_id_edit)

	var create_button := Button.new()
	create_button.text = "Create"
	create_button.pressed.connect(_on_create_pressed)
	table_row.add_child(create_button)

	var refresh_button := Button.new()
	refresh_button.text = "Refresh"
	refresh_button.pressed.connect(_on_refresh_pressed)
	table_row.add_child(refresh_button)
	root.add_child(_field("Table", table_row))

	table_list = OptionButton.new()
	table_list.item_selected.connect(_on_table_selected)
	root.add_child(_field("Available Tables", table_list))

	seat_picker = OptionButton.new()
	for seat in ["E", "S", "W", "N"]:
		seat_picker.add_item(seat)
	root.add_child(_field("Seat", seat_picker))

	display_name_edit = LineEdit.new()
	display_name_edit.placeholder_text = "Display name"
	root.add_child(_field("Name", display_name_edit))

	join_button = Button.new()
	join_button.text = "Join Human Seat"
	join_button.pressed.connect(_on_join_pressed)
	root.add_child(join_button)

	status_label = Label.new()
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(status_label)


func _field(label_text: String, control: Control) -> Control:
	var box := VBoxContainer.new()
	var label := Label.new()
	label.text = label_text
	box.add_child(label)
	box.add_child(control)
	return box


func _apply_server_url() -> void:
	session.base_url = server_url_edit.text.strip_edges()
	api.set_base_url(session.base_url)


func _on_create_pressed() -> void:
	_apply_server_url()
	status_label.text = "Creating table..."
	var result = await api.create_table()
	if not result.get("ok", false):
		status_label.text = result.get("error", "create failed")
		return
	var data: Dictionary = result["data"]
	table_id_edit.text = str(data.get("table_id", ""))
	status_label.text = "Created %s" % table_id_edit.text
	await _refresh_tables()


func _on_refresh_pressed() -> void:
	_apply_server_url()
	await _refresh_tables()


func _refresh_tables() -> void:
	status_label.text = "Loading tables..."
	var result = await api.list_tables()
	table_list.clear()
	if not result.get("ok", false):
		status_label.text = result.get("error", "refresh failed")
		return
	for table_id in result["data"].get("tables", []):
		table_list.add_item(str(table_id))
	status_label.text = "Loaded %s table(s)" % table_list.get_item_count()


func _on_table_selected(index: int) -> void:
	table_id_edit.text = table_list.get_item_text(index)


func _on_join_pressed() -> void:
	_apply_server_url()
	var table_id := table_id_edit.text.strip_edges()
	if table_id == "":
		status_label.text = "Choose or create a table first."
		return
	session.table_id = table_id
	session.seat = seat_picker.get_item_text(seat_picker.selected)
	session.display_name = display_name_edit.text.strip_edges()
	if session.display_name == "":
		session.display_name = "Player"

	join_button.disabled = true
	status_label.text = "Joining %s..." % table_id
	var result = await api.join_human(session.table_id, session.seat, session.display_name)
	join_button.disabled = false
	if not result.get("ok", false):
		status_label.text = result.get("error", "join failed")
		return
	session.update_from_command_response(result["data"])
	status_label.text = "Joined %s as %s" % [session.table_id, session.seat]
	joined_table.emit()
