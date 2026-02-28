## Predictive Coding Gamestate Engine — Godot 4.x client.
##
## Attach this script to a Node2D in your scene.
##
## Renders a 10×10 destructible grid.  Left-click a block to strike
## (destroy) it.  The Python server resolves the physics via free-energy
## minimization and returns the new world state.
##
## Materials (Phase 8):
##   Stone (left half, x < 5):
##     +1.0 (solid)  → dark slate     rgb(60, 65, 85)
##     −1.0 (air)    → sky blue       rgb(135, 200, 235)
##   Sand (right half, x ≥ 5):
##     +1.0 (solid)  → golden brown   rgb(190, 160, 80)
##     −1.0 (air)    → sky blue       rgb(135, 200, 235)

extends Node2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const GRID_W: int = 10
const GRID_H: int = 10
const BLOCK_SIZE: int = 40
const SERVER_URL: String = "http://127.0.0.1:5001"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var blocks: Array[ColorRect] = []
var materials: Array = []           # "stone" or "sand" per cell (from server)
var http_request: HTTPRequest
var request_pending: bool = false   # guard against overlapping requests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

func _ready() -> void:
	# --- Build the visual grid ---
	for y in range(GRID_H):
		for x in range(GRID_W):
			var block := ColorRect.new()
			block.size = Vector2(38, 38)                         # 2 px gap
			block.position = Vector2(x * BLOCK_SIZE + 1, y * BLOCK_SIZE + 1)
			if x < 5:
				block.color = Color(60.0/255, 65.0/255, 85.0/255)  # stone = dark slate
			else:
				block.color = Color(190.0/255, 160.0/255, 80.0/255)  # sand = golden brown
			add_child(block)
			blocks.append(block)

	# --- Networking ---
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

	# Fetch the initial world state from the server.
	request_pending = true
	http_request.request(SERVER_URL + "/get_state")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton \
			and event.pressed \
			and event.button_index == MOUSE_BUTTON_LEFT:

		var grid_x: int = int(event.position.x) / BLOCK_SIZE
		var grid_y: int = int(event.position.y) / BLOCK_SIZE

		if grid_x >= 0 and grid_x < GRID_W \
				and grid_y >= 0 and grid_y < GRID_H:
			strike_block(grid_x, grid_y)

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

func strike_block(x: int, y: int) -> void:
	if request_pending:
		return                          # wait for the previous response

	var body: String = JSON.stringify({"x": x, "y": y})
	var headers: PackedStringArray = ["Content-Type: application/json"]

	request_pending = true
	http_request.request(
		SERVER_URL + "/strike",
		headers,
		HTTPClient.METHOD_POST,
		body
	)


func _on_request_completed(
		result: int,
		response_code: int,
		_headers: PackedStringArray,
		body: PackedByteArray
) -> void:
	request_pending = false

	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		push_error(
			"Server request failed: result=%d  code=%d" % [result, response_code]
		)
		return

	var json := JSON.new()
	var err := json.parse(body.get_string_from_utf8())
	if err != OK:
		push_error("Failed to parse JSON response")
		return

	# /get_state returns {"state": [...], "materials": [...]}.
	# /strike and /reset return a flat array.
	if json.data is Dictionary:
		var state: Array = json.data["state"]
		if json.data.has("materials"):
			materials = json.data["materials"]
		update_grid(state)
	else:
		var state: Array = json.data
		update_grid(state)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

func update_grid(state: Array) -> void:
	for i in range(state.size()):
		if i >= blocks.size():
			break

		var value: float = clampf(float(state[i]), -1.0, 1.0)
		var t: float = (1.0 - value) / 2.0   # 0 = solid, 1 = air

		# Determine material for this cell.
		var mat: String = "stone"
		if i < materials.size():
			mat = materials[i]
		else:
			# Fallback: left half = stone, right half = sand.
			mat = "stone" if (i % GRID_W) < 5 else "sand"

		var r: float
		var g: float
		var b: float

		if t > 0.9:
			# Air / destroyed — sky blue (same for both materials)
			r = 135.0; g = 200.0; b = 235.0
		elif mat == "stone":
			# Stone: dark slate-gray → light gray-blue
			r = 60.0 + t * 120.0
			g = 65.0 + t * 120.0
			b = 85.0 + t * 130.0
		else:
			# Sand: warm golden-brown → pale tan
			r = 190.0 + t * 40.0
			g = 160.0 + t * 60.0
			b = 80.0 + t * 100.0

		blocks[i].color = Color(r / 255.0, g / 255.0, b / 255.0)
