## Predictive Coding Gamestate Engine — Godot 4.x client.
##
## Attach this script to a Node2D in your scene.
##
## Renders a 10×10 destructible grid.  Left-click a block to strike
## (destroy) it.  The Python server resolves the physics via free-energy
## minimization and returns the new world state.
##
## Colour mapping (earthy theme):
##     +1.0 (solid)  → earthy brown  rgb(90, 65, 40)
##     ~0.5 (weak)   → cracked tan   rgb(180, 160, 120)
##      0.0 (failing) → pale sand     rgb(220, 210, 180)
##     −1.0 (air)    → sky blue      rgb(135, 200, 235)

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
			block.color = Color(90.0/255, 65.0/255, 40.0/255)      # solid = earthy brown
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

	var state: Array = json.data
	update_grid(state)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

func update_grid(state: Array) -> void:
	for i in range(state.size()):
		if i >= blocks.size():
			break

		# Map value from [-1.0 … +1.0] to earthy colour gradient:
		#   +1.0 (solid)  → earthy brown  rgb(90, 65, 40)
		#   ~0.5 (weak)   → cracked tan   rgb(180, 160, 120)
		#    0.0 (failing) → pale sand     rgb(220, 210, 180)
		#   −1.0 (air)    → sky blue      rgb(135, 200, 235)
		var value: float = clampf(float(state[i]), -1.0, 1.0)
		var t: float = (1.0 - value) / 2.0   # 0 = solid, 1 = air

		var r: float
		var g: float
		var b: float

		if t > 0.9:
			# Air / destroyed — sky blue
			r = 135.0; g = 200.0; b = 235.0
		else:
			# Solid → cracked gradient
			r = 90.0 + t * 140.0
			g = 65.0 + t * 155.0
			b = 40.0 + t * 150.0

		blocks[i].color = Color(r / 255.0, g / 255.0, b / 255.0)
