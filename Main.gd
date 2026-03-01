## Predictive Coding Gamestate Engine — Godot 4.x client.
##
## Attach this script to a Node2D in your scene.
##
## Renders a 10×10 destructible grid.  Left-click a block to strike
## (destroy) it.  The Python server resolves the physics via free-energy
## minimization and returns the new world state.
##
## Dual-state (Phase 9):
##   State tensor is 200-dim: [density×100, thermal×100].
##   Density: +1.0 = solid, −1.0 = air.
##   Thermal: −1.0 = cold, +1.0 = hot.
##
## Materials (Phase 8):
##   Stone (left half, x < 5):  dark slate  rgb(60, 65, 85)
##   Sand  (right half, x ≥ 5): golden brown rgb(190, 160, 80)
##
## Heat overlay: hot cells blend toward fiery orange rgb(255, 76, 0).
## Left-click = strike (density).  Right-click / Shift+click = heat.

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
	if event is InputEventMouseButton and event.pressed:
		var grid_x: int = int(event.position.x) / BLOCK_SIZE
		var grid_y: int = int(event.position.y) / BLOCK_SIZE

		if grid_x >= 0 and grid_x < GRID_W \
				and grid_y >= 0 and grid_y < GRID_H:
			if event.button_index == MOUSE_BUTTON_RIGHT \
					or (event.button_index == MOUSE_BUTTON_LEFT and event.shift_pressed):
				heat_block(grid_x, grid_y)
			elif event.button_index == MOUSE_BUTTON_LEFT:
				strike_block(grid_x, grid_y)

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

func heat_block(x: int, y: int) -> void:
	if request_pending:
		return
	var body: String = JSON.stringify({"x": x, "y": y})
	var headers: PackedStringArray = ["Content-Type: application/json"]
	request_pending = true
	http_request.request(
		SERVER_URL + "/heat",
		headers,
		HTTPClient.METHOD_POST,
		body
	)


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
	var grid_cells: int = GRID_W * GRID_H   # 100

	for i in range(grid_cells):
		if i >= blocks.size():
			break

		# Density value: indices 0–99.
		var density_val: float = clampf(float(state[i]), -1.0, 1.0)
		# Thermal value: indices 100–199.
		var heat_val: float = -1.0
		if i + grid_cells < state.size():
			heat_val = clampf(float(state[i + grid_cells]), -1.0, 1.0)

		var t: float = (1.0 - density_val) / 2.0   # 0 = solid, 1 = air

		# Determine material for this cell.
		var mat: String = "stone"
		if i < materials.size():
			mat = materials[i]
		else:
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

		# Heat overlay: blend toward fiery orange (255, 76, 0).
		if heat_val > -0.8:
			var heat_t: float = (heat_val + 1.0) / 2.0   # 0 = cold, 1 = hot
			var blend: float = heat_t * 0.8               # max 80% blend
			r = r * (1.0 - blend) + 255.0 * blend
			g = g * (1.0 - blend) + 76.0 * blend
			b = b * (1.0 - blend) + 0.0 * blend

		blocks[i].color = Color(r / 255.0, g / 255.0, b / 255.0)
