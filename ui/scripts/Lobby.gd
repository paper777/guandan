extends Control

signal open_game
signal leave_table

var api
var session

var status_label: Label
var table_label: Label
var seat_labels: Dictionary = {}
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


func _exit_tree() -> void:
	if poll_timer != null:
		poll_timer.stop()


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

	table_label = Label.new()
	table_label.text = "Lobby"
	table_label.add_theme_font_size_override("font_size", 28)
	root.add_child(table_label)

	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 12)
	root.add_child(grid)
	for seat in ["E", "S", "W", "N"]:
		var panel := PanelContainer.new()
		panel.custom_minimum_size = Vector2(260, 88)
		var label := Label.new()
		label.text = "%s: empty" % seat
		label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		panel.add_child(label)
		grid.add_child(panel)
		seat_labels[seat] = label

	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 8)
	var ready_button := Button.new()
	ready_button.text = "Ready"
	ready_button.pressed.connect(_on_ready_pressed)
	actions.add_child(ready_button)

	var start_button := Button.new()
	start_button.text = "Start Match"
	start_button.pressed.connect(_on_start_pressed)
	actions.add_child(start_button)

	var refresh_button := Button.new()
	refresh_button.text = "Refresh"
	refresh_button.pressed.connect(_refresh_snapshot)
	actions.add_child(refresh_button)

	var leave_button := Button.new()
	leave_button.text = "Leave Local Session"
	leave_button.pressed.connect(func(): leave_table.emit())
	actions.add_child(leave_button)
	root.add_child(actions)

	status_label = Label.new()
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(status_label)


func _refresh_snapshot() -> void:
	if session.table_id == "":
		return
	var result = await api.table_snapshot(session.table_id)
	if not result.get("ok", false):
		status_label.text = result.get("error", "snapshot failed")
		return
	session.public_snapshot = result["data"]
	session.last_event_seq = int(session.public_snapshot.get("event_seq", session.last_event_seq))
	_render_snapshot(session.public_snapshot)
	var phase := str(session.public_snapshot.get("phase", ""))
	if phase in ["PLAYING", "TRIBUTE", "DEAL_COMPLETE", "MATCH_COMPLETE"]:
		open_game.emit()


func _render_snapshot(snapshot: Dictionary) -> void:
	table_label.text = "Lobby: %s (%s)" % [session.table_id, snapshot.get("phase", "unknown")]
	var seats: Dictionary = snapshot.get("seats", {})
	for seat in ["E", "S", "W", "N"]:
		var label: Label = seat_labels[seat]
		if seats.has(seat):
			var player: Dictionary = seats[seat]
			var marker := " <- you" if seat == session.seat else ""
			label.text = "%s: %s\n%s%s" % [
				seat,
				player.get("display_name", player.get("player_id", "player")),
				player.get("kind", "human"),
				marker
			]
		else:
			label.text = "%s: empty" % seat


func _on_ready_pressed() -> void:
	status_label.text = "Marking ready..."
	var result = await api.ready(session.table_id, session.seat, session.controller_id)
	if not result.get("ok", false):
		status_label.text = result.get("error", "ready failed")
		return
	session.update_from_command_response(result["data"])
	status_label.text = "Ready accepted."
	await _refresh_snapshot()


func _on_start_pressed() -> void:
	status_label.text = "Starting match..."
	var result = await api.start(session.table_id)
	if not result.get("ok", false):
		status_label.text = result.get("error", "start failed")
		return
	session.update_from_command_response(result["data"])
	status_label.text = "Match started."
	open_game.emit()
