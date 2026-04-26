extends RefCounted
class_name SessionState

var base_url: String = "http://127.0.0.1:8000"
var table_id: String = ""
var seat: String = "E"
var player_id: String = ""
var controller_id: String = ""
var display_name: String = "Player"
var selected_card_ids: Array[String] = []
var public_snapshot: Dictionary = {}
var seat_snapshot: Dictionary = {}
var last_event_seq: int = 0


func reset_table() -> void:
	table_id = ""
	seat = "E"
	player_id = ""
	controller_id = ""
	selected_card_ids.clear()
	public_snapshot = {}
	seat_snapshot = {}
	last_event_seq = 0


func joined() -> bool:
	return table_id != "" and seat != "" and controller_id != ""


func update_from_command_response(payload: Dictionary) -> void:
	if payload.has("player_id"):
		player_id = str(payload["player_id"])
	if payload.has("controller_id"):
		controller_id = str(payload["controller_id"])
	if payload.has("event_seq"):
		last_event_seq = int(payload["event_seq"])
	if payload.has("snapshot") and payload["snapshot"] is Dictionary:
		public_snapshot = payload["snapshot"]
