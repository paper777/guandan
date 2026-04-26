extends Control

signal leave_table

const CardTileScene := preload("res://scenes/CardTile.tscn")

var api
var session

var status_label: Label
var table_label: Label
var countdown_label: Label
var seat_labels: Dictionary = {}
var hand_container: HBoxContainer
var event_log: RichTextLabel
var play_button: Button
var pass_button: Button
var start_next_button: Button
var poll_timer: Timer


func setup(api_client, session_state) -> void:
	api = api_client
	session = session_state


func _ready() -> void:
	_build_ui()
	_refresh_snapshot()
	poll_timer = Timer.new()
	poll_timer.wait_time = 1.0
	poll_timer.timeout.connect(_refresh_snapshot)
	add_child(poll_timer)
	poll_timer.start()


func _process(_delta: float) -> void:
	_update_countdown()


func _exit_tree() -> void:
	if poll_timer != null:
		poll_timer.stop()


func _build_ui() -> void:
	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 24)
	margin.add_theme_constant_override("margin_top", 20)
	margin.add_theme_constant_override("margin_right", 24)
	margin.add_theme_constant_override("margin_bottom", 20)
	add_child(margin)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 12)
	margin.add_child(root)

	var header := HBoxContainer.new()
	header.add_theme_constant_override("separation", 10)
	table_label = Label.new()
	table_label.text = "Game"
	table_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	table_label.add_theme_font_size_override("font_size", 24)
	header.add_child(table_label)
	countdown_label = Label.new()
	header.add_child(countdown_label)
	var leave_button := Button.new()
	leave_button.text = "Leave Local Session"
	leave_button.pressed.connect(func(): leave_table.emit())
	header.add_child(leave_button)
	root.add_child(header)

	var table_grid := GridContainer.new()
	table_grid.columns = 2
	table_grid.add_theme_constant_override("h_separation", 12)
	table_grid.add_theme_constant_override("v_separation", 12)
	root.add_child(table_grid)
	for seat in ["N", "W", "E", "S"]:
		var panel := PanelContainer.new()
		panel.custom_minimum_size = Vector2(280, 90)
		var label := Label.new()
		label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		panel.add_child(label)
		table_grid.add_child(panel)
		seat_labels[seat] = label

	var hand_title := Label.new()
	hand_title.text = "Hand"
	root.add_child(hand_title)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.custom_minimum_size = Vector2(0, 128)
	hand_container = HBoxContainer.new()
	hand_container.add_theme_constant_override("separation", 6)
	scroll.add_child(hand_container)
	root.add_child(scroll)

	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 8)
	play_button = Button.new()
	play_button.text = "Play Selected"
	play_button.pressed.connect(_on_play_pressed)
	actions.add_child(play_button)
	pass_button = Button.new()
	pass_button.text = "Pass"
	pass_button.pressed.connect(_on_pass_pressed)
	actions.add_child(pass_button)
	start_next_button = Button.new()
	start_next_button.text = "Start Next Deal"
	start_next_button.pressed.connect(_on_start_next_pressed)
	actions.add_child(start_next_button)
	var refresh_button := Button.new()
	refresh_button.text = "Refresh"
	refresh_button.pressed.connect(_refresh_snapshot)
	actions.add_child(refresh_button)
	root.add_child(actions)

	status_label = Label.new()
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(status_label)

	event_log = RichTextLabel.new()
	event_log.custom_minimum_size = Vector2(0, 130)
	event_log.fit_content = false
	root.add_child(event_log)


func _refresh_snapshot() -> void:
	if not session.joined():
		return
	var result = await api.seat_snapshot(session.table_id, session.seat, session.controller_id)
	if not result.get("ok", false):
		status_label.text = result.get("error", "snapshot failed")
		return
	session.seat_snapshot = result["data"]
	session.public_snapshot = session.seat_snapshot.get("public", {})
	session.last_event_seq = int(session.public_snapshot.get("event_seq", session.last_event_seq))
	_render_snapshot()


func _render_snapshot() -> void:
	var public_snapshot: Dictionary = session.public_snapshot
	var phase := str(public_snapshot.get("phase", "unknown"))
	table_label.text = "%s | seat %s | %s" % [session.table_id, session.seat, phase]
	_render_seats(public_snapshot)
	_render_hand(session.seat_snapshot.get("hand", []))
	_render_actions()
	_update_countdown()


func _render_seats(snapshot: Dictionary) -> void:
	var seats: Dictionary = snapshot.get("seats", {})
	var hand_counts: Dictionary = snapshot.get("hand_counts", {})
	var current_turn := str(snapshot.get("current_turn", ""))
	var finish_order: Array = snapshot.get("finish_order", [])
	for seat in ["N", "W", "E", "S"]:
		var label: Label = seat_labels[seat]
		var name := "empty"
		if seats.has(seat):
			var player: Dictionary = seats[seat]
			name = str(player.get("display_name", player.get("player_id", "player")))
		var markers: Array[String] = []
		if seat == session.seat:
			markers.append("you")
		if seat == current_turn:
			markers.append("turn")
		var finish_index := finish_order.find(seat)
		if finish_index >= 0:
			markers.append("out #%s" % (finish_index + 1))
		var marker_text := ""
		if not markers.is_empty():
			marker_text = " [%s]" % ", ".join(markers)
		label.text = "%s: %s%s\ncards: %s" % [seat, name, marker_text, hand_counts.get(seat, "-")]


func _render_hand(card_ids: Array) -> void:
	var existing_selected: Array = session.selected_card_ids.duplicate()
	for child in hand_container.get_children():
		child.queue_free()
	session.selected_card_ids.clear()
	for card_id_value in card_ids:
		var card_id := str(card_id_value)
		var tile = CardTileScene.instantiate()
		tile.set_card_id(card_id)
		tile.selection_changed.connect(_on_card_selection_changed)
		hand_container.add_child(tile)
		if existing_selected.has(card_id):
			tile.set_selected(true)


func _render_actions() -> void:
	var legal_action = session.seat_snapshot.get("legal_action", null)
	var phase := str(session.public_snapshot.get("phase", ""))
	var can_act := legal_action != null
	play_button.disabled = not can_act or session.selected_card_ids.is_empty()
	pass_button.disabled = legal_action != "play_or_pass"
	start_next_button.visible = phase == "DEAL_COMPLETE"


func _update_countdown() -> void:
	var deadline = session.public_snapshot.get("action_deadline_epoch_ms", null)
	if deadline == null:
		countdown_label.text = ""
		return
	var remaining_ms := int(deadline) - Time.get_unix_time_from_system() * 1000.0
	var seconds = max(0, int(ceil(remaining_ms / 1000.0)))
	var acting := str(session.public_snapshot.get("acting_seat", session.public_snapshot.get("current_turn", "")))
	countdown_label.text = "%s: %ss" % [acting, seconds]


func _on_card_selection_changed(card_id: String, selected: bool) -> void:
	if selected and not session.selected_card_ids.has(card_id):
		session.selected_card_ids.append(card_id)
	elif not selected:
		session.selected_card_ids.erase(card_id)
	_render_actions()


func _on_play_pressed() -> void:
	if session.selected_card_ids.is_empty():
		status_label.text = "Select at least one card."
		return
	status_label.text = "Playing cards..."
	var cards: Array[String] = session.selected_card_ids.duplicate()
	var result = await api.play_cards(session.table_id, session.seat, session.controller_id, cards)
	_handle_command_result(result, "Play accepted.")


func _on_pass_pressed() -> void:
	status_label.text = "Passing..."
	var result = await api.pass_turn(session.table_id, session.seat, session.controller_id)
	_handle_command_result(result, "Pass accepted.")


func _on_start_next_pressed() -> void:
	status_label.text = "Starting next deal..."
	var result = await api.start(session.table_id)
	_handle_command_result(result, "Next deal started.")


func _handle_command_result(result: Dictionary, success_message: String) -> void:
	if not result.get("ok", false):
		status_label.text = result.get("error", "command failed")
		return
	session.update_from_command_response(result["data"])
	_append_events(result["data"].get("events", []))
	session.selected_card_ids.clear()
	status_label.text = success_message
	await _refresh_snapshot()


func _append_events(events: Array) -> void:
	for event in events:
		if event is Dictionary:
			event_log.append_text("#%s %s %s\n" % [
				event.get("seq", "?"),
				event.get("type", "Event"),
				JSON.stringify(event.get("payload", {}))
			])
	event_log.scroll_to_line(max(0, event_log.get_line_count() - 1))
