extends Node
class_name ApiClient

var base_url: String = "http://127.0.0.1:8000"
var timeout_seconds: float = 10.0


func set_base_url(value: String) -> void:
	base_url = value.strip_edges().trim_suffix("/")


func create_table() -> Dictionary:
	return await request_json("POST", "/tables", {})


func list_tables() -> Dictionary:
	return await request_json("GET", "/tables")


func table_snapshot(table_id: String) -> Dictionary:
	return await request_json("GET", "/tables/%s" % table_id.uri_encode())


func seat_snapshot(table_id: String, seat: String, controller_id: String) -> Dictionary:
	return await request_json(
		"GET",
		"/tables/%s/seats/%s/snapshot" % [table_id.uri_encode(), seat.uri_encode()],
		null,
		{"controller_id": controller_id}
	)


func join_human(table_id: String, seat: String, display_name: String) -> Dictionary:
	return await request_json(
		"POST",
		"/tables/%s/join-human" % table_id.uri_encode(),
		{"seat": seat, "display_name": display_name}
	)


func join_local_bot(table_id: String, seat: String, display_name: String) -> Dictionary:
	return await request_json(
		"POST",
		"/tables/%s/join-local-bot" % table_id.uri_encode(),
		{"seat": seat, "display_name": display_name}
	)


func ready(table_id: String, seat: String, controller_id: String) -> Dictionary:
	return await request_json(
		"POST",
		"/tables/%s/ready" % table_id.uri_encode(),
		{"seat": seat, "controller_id": controller_id}
	)


func start(table_id: String) -> Dictionary:
	return await request_json("POST", "/tables/%s/start" % table_id.uri_encode(), {})


func play_cards(table_id: String, seat: String, controller_id: String, card_ids: Array[String]) -> Dictionary:
	return await request_json(
		"POST",
		"/tables/%s/play" % table_id.uri_encode(),
		{"seat": seat, "controller_id": controller_id, "card_ids": card_ids}
	)


func pass_turn(table_id: String, seat: String, controller_id: String) -> Dictionary:
	return await request_json(
		"POST",
		"/tables/%s/pass" % table_id.uri_encode(),
		{"seat": seat, "controller_id": controller_id}
	)


func request_json(method: String, path: String, body = null, query: Dictionary = {}) -> Dictionary:
	var http := HTTPRequest.new()
	http.timeout = timeout_seconds
	add_child(http)

	var url := "%s%s" % [base_url, path]
	if not query.is_empty():
		url = "%s?%s" % [url, _encode_query(query)]

	var headers: PackedStringArray = ["Accept: application/json"]
	var request_body := ""
	if body != null:
		headers.append("Content-Type: application/json")
		request_body = JSON.stringify(body)

	var error := http.request(url, headers, _method_code(method), request_body)
	if error != OK:
		http.queue_free()
		return {"ok": false, "status": 0, "data": {}, "error": "request failed to start: %s" % error}

	var completed: Array = await http.request_completed
	http.queue_free()

	var response_code := int(completed[1])
	var raw_body: PackedByteArray = completed[3]
	var text := raw_body.get_string_from_utf8()
	var parsed = {}
	if text != "":
		var json = JSON.parse_string(text)
		if json is Dictionary:
			parsed = json
		else:
			return {
				"ok": false,
				"status": response_code,
				"data": {},
				"error": "server returned non-object JSON"
			}

	var ok := response_code >= 200 and response_code < 300
	return {
		"ok": ok,
		"status": response_code,
		"data": parsed,
		"error": "" if ok else _error_message(parsed)
	}


func _method_code(method: String) -> int:
	match method:
		"GET":
			return HTTPClient.METHOD_GET
		"POST":
			return HTTPClient.METHOD_POST
		_:
			return HTTPClient.METHOD_GET


func _encode_query(query: Dictionary) -> String:
	var parts: Array[String] = []
	for key in query.keys():
		parts.append("%s=%s" % [str(key).uri_encode(), str(query[key]).uri_encode()])
	return "&".join(parts)


func _error_message(payload: Dictionary) -> String:
	if payload.has("rejection") and payload["rejection"] is Dictionary:
		var rejection: Dictionary = payload["rejection"]
		return "%s: %s" % [rejection.get("code", "rejected"), rejection.get("message", "")]
	if payload.has("detail"):
		return str(payload["detail"])
	if payload.has("error"):
		return str(payload["error"])
	return "request failed"
