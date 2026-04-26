extends Control

const ApiClientScene := preload("res://scripts/ApiClient.gd")
const SessionStateScript := preload("res://scripts/SessionState.gd")
const TableSelectScene := preload("res://scenes/TableSelect.tscn")
const LobbyScene := preload("res://scenes/Lobby.tscn")
const GameTableScene := preload("res://scenes/GameTable.tscn")

var api
var session
var current_screen: Control


func _ready() -> void:
	api = ApiClientScene.new()
	add_child(api)
	session = SessionStateScript.new()
	_show_table_select()


func _show_table_select() -> void:
	var screen = TableSelectScene.instantiate()
	screen.setup(api, session)
	screen.joined_table.connect(_show_lobby)
	_set_screen(screen)


func _show_lobby() -> void:
	var screen = LobbyScene.instantiate()
	screen.setup(api, session)
	screen.open_game.connect(_show_game)
	screen.leave_table.connect(_leave_table)
	_set_screen(screen)


func _show_game() -> void:
	var screen = GameTableScene.instantiate()
	screen.setup(api, session)
	screen.leave_table.connect(_leave_table)
	_set_screen(screen)


func _leave_table() -> void:
	session.reset_table()
	_show_table_select()


func _set_screen(screen: Control) -> void:
	if current_screen != null:
		current_screen.queue_free()
	current_screen = screen
	add_child(screen)
	screen.set_anchors_preset(Control.PRESET_FULL_RECT)
