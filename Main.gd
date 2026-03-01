## Predictive Coding Gamestate Engine — Godot 4.x client.
##
## Attach this script to a Node2D in your scene.
##
## Renders a 10×10 destructible grid with procedural textures.
## Left-click = strike (destroy).  Right-click / Shift+click = heat.
## The Python server resolves physics via free-energy minimization.
##
## Dual-state (Phase 9):
##   State tensor is 200-dim: [density×100, thermal×100].
##   Density: +1.0 = solid, −1.0 = air.
##   Thermal: −1.0 = cold, +1.0 = hot.
##
## Materials (Phase 10):
##   Stone (x < 3):     Simplex noise, dark slate gray
##   Wood  (3 ≤ x < 7): Cellular noise (vertical grain), warm brown
##   Ice   (x ≥ 7):     Value noise, pale translucent cyan
##
## Background inference (Phase 11):
##   A Timer fires every 0.5 s → POST /tick.
##   Server applies thermal dissipation, ambient entropy, continuous
##   inference (healing), then resolves cascades.
##   Uses a dedicated HTTPRequest so ticks don't block player actions.
##
## Procedural textures (Phase 12):
##   All material textures generated synchronously in _ready() using
##   FastNoiseLite + Image + ImageTexture (no async await needed).
##   Heat overlay: radial GradientTexture2D with additive blending.
##   Heat sprites pulse subtly via _process() for fire animation.

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

var blocks: Array = []              # Array of { "base": Sprite2D, "heat": Sprite2D }
var materials: Array = []           # "stone", "wood", or "ice" per cell
var http_request: HTTPRequest
var request_pending: bool = false   # guard against overlapping requests

# Tick metronome (Phase 11)
var tick_http_request: HTTPRequest
var tick_request_pending: bool = false
var tick_timer: Timer

# Procedural textures (Phase 12)
var tex_stone: ImageTexture
var tex_wood: ImageTexture
var tex_ice: ImageTexture
var tex_heat: GradientTexture2D


# ---------------------------------------------------------------------------
# Procedural texture generation (Phase 12)
# ---------------------------------------------------------------------------

func _generate_stone_texture() -> ImageTexture:
	# Simplex noise → dark slate gray shades (40×40).
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.frequency = 0.05
	noise.seed = randi()
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = 3

	var img := Image.create(40, 40, false, Image.FORMAT_RGBA8)
	for y in range(40):
		for x in range(40):
			var val: float = (noise.get_noise_2d(float(x), float(y)) + 1.0) / 2.0
			# Dark slate: base rgb(55, 60, 80) with ±25 variation.
			var r: float = (55.0 + val * 50.0) / 255.0
			var g: float = (60.0 + val * 45.0) / 255.0
			var b: float = (80.0 + val * 50.0) / 255.0
			img.set_pixel(x, y, Color(r, g, b, 1.0))

	return ImageTexture.create_from_image(img)


func _generate_wood_texture() -> ImageTexture:
	# Cellular noise, vertically stretched → warm brown wood grain (40×40).
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_CELLULAR
	noise.frequency = 0.08
	noise.seed = randi()
	noise.cellular_return_type = FastNoiseLite.RETURN_DISTANCE

	var img := Image.create(40, 40, false, Image.FORMAT_RGBA8)
	for y in range(40):
		for x in range(40):
			# Y × 0.3 stretches noise vertically → tall thin cells = grain.
			var val: float = (noise.get_noise_2d(float(x), float(y) * 0.3) + 1.0) / 2.0
			# Warm brown: base rgb(80, 50, 25) with grain variation.
			var r: float = (80.0 + val * 65.0) / 255.0
			var g: float = (50.0 + val * 45.0) / 255.0
			var b: float = (25.0 + val * 30.0) / 255.0
			img.set_pixel(x, y, Color(r, g, b, 1.0))

	return ImageTexture.create_from_image(img)


func _generate_ice_texture() -> ImageTexture:
	# Value noise → pale translucent cyan/white (40×40).
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_VALUE
	noise.frequency = 0.06
	noise.seed = randi()

	var img := Image.create(40, 40, false, Image.FORMAT_RGBA8)
	for y in range(40):
		for x in range(40):
			var val: float = (noise.get_noise_2d(float(x), float(y)) + 1.0) / 2.0
			# Pale cyan/white: base rgb(175, 235, 240) with shimmer.
			var r: float = (175.0 + val * 60.0) / 255.0
			var g: float = (235.0 + val * 20.0) / 255.0
			var b: float = (240.0 + val * 15.0) / 255.0
			img.set_pixel(x, y, Color(r, g, b, 1.0))

	return ImageTexture.create_from_image(img)


func _generate_heat_texture() -> GradientTexture2D:
	# Radial gradient: bright yellow center → fiery orange → transparent (60×60).
	var grad := Gradient.new()
	grad.colors = PackedColorArray([
		Color(1.0, 0.95, 0.4, 1.0),    # bright yellow-white center
		Color(1.0, 0.45, 0.0, 0.8),    # fiery orange ring
		Color(0.6, 0.1, 0.0, 0.3),     # dark ember
		Color(0.0, 0.0, 0.0, 0.0),     # transparent black edge
	])
	grad.offsets = PackedFloat32Array([0.0, 0.25, 0.6, 1.0])

	var tex := GradientTexture2D.new()
	tex.width = 60
	tex.height = 60
	tex.fill = GradientTexture2D.FILL_RADIAL
	tex.fill_from = Vector2(0.5, 0.5)   # center
	tex.fill_to = Vector2(1.0, 0.5)     # radius = half width
	tex.gradient = grad

	return tex


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

func _ready() -> void:
	# --- Sky background (visible through destroyed/transparent cells) ---
	var sky := ColorRect.new()
	sky.size = Vector2(GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE)
	sky.color = Color(0.53, 0.78, 0.92, 1.0)    # rgb(135, 200, 235) sky blue
	add_child(sky)

	# --- Generate procedural textures ---
	tex_stone = _generate_stone_texture()
	tex_wood  = _generate_wood_texture()
	tex_ice   = _generate_ice_texture()
	tex_heat  = _generate_heat_texture()

	# --- Build the node grid ---
	for y in range(GRID_H):
		for x in range(GRID_W):
			var container := Node2D.new()
			container.position = Vector2(x * BLOCK_SIZE, y * BLOCK_SIZE)
			add_child(container)

			# Base material sprite (40×40, top-left aligned).
			var base_sprite := Sprite2D.new()
			base_sprite.centered = false
			if x < 3:
				base_sprite.texture = tex_stone
			elif x < 7:
				base_sprite.texture = tex_wood
			else:
				base_sprite.texture = tex_ice
			container.add_child(base_sprite)

			# Heat overlay sprite (60×60, centered on cell, additive).
			var heat_sprite := Sprite2D.new()
			heat_sprite.texture = tex_heat
			heat_sprite.centered = true
			heat_sprite.position = Vector2(BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0)
			var heat_mat := CanvasItemMaterial.new()
			heat_mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
			heat_sprite.material = heat_mat
			heat_sprite.modulate = Color(1, 1, 1, 0)    # invisible initially
			container.add_child(heat_sprite)

			blocks.append({ "base": base_sprite, "heat": heat_sprite })

	# --- Networking (player actions) ---
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

	# --- Networking (tick metronome — Phase 11) ---
	tick_http_request = HTTPRequest.new()
	add_child(tick_http_request)
	tick_http_request.request_completed.connect(_on_tick_request_completed)

	# --- Tick timer — 0.5 s (2 ticks/sec), autostart ---
	tick_timer = Timer.new()
	tick_timer.wait_time = 0.5
	tick_timer.autostart = true
	add_child(tick_timer)
	tick_timer.timeout.connect(_on_tick_timeout)

	# Fetch the initial world state from the server.
	request_pending = true
	http_request.request(SERVER_URL + "/get_state")


# ---------------------------------------------------------------------------
# Animation — heat pulse (Phase 12)
# ---------------------------------------------------------------------------

func _process(_delta: float) -> void:
	var time: float = Time.get_ticks_msec() / 1000.0
	for i in range(blocks.size()):
		var heat: Sprite2D = blocks[i]["heat"]
		if heat.modulate.a > 0.01:
			# Subtle pulsing scale so the fire looks alive.
			# Per-cell phase offset (i * 0.7) prevents uniform pulsing.
			var pulse: float = 1.0 + 0.15 * sin(time * 6.0 + float(i) * 0.7)
			heat.scale = Vector2(pulse, pulse)


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
# Networking — player actions
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
# Networking — tick metronome (Phase 11)
# ---------------------------------------------------------------------------

func _on_tick_timeout() -> void:
	if tick_request_pending:
		return                          # previous tick still in-flight
	var headers: PackedStringArray = ["Content-Type: application/json"]
	tick_request_pending = true
	tick_http_request.request(
		SERVER_URL + "/tick",
		headers,
		HTTPClient.METHOD_POST,
		"{}"
	)


func _on_tick_request_completed(
		result: int,
		response_code: int,
		_headers: PackedStringArray,
		body: PackedByteArray
) -> void:
	tick_request_pending = false

	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		return                          # silently skip failed ticks

	var json := JSON.new()
	var err := json.parse(body.get_string_from_utf8())
	if err != OK:
		return

	# /tick returns a flat 200-element array.
	if json.data is Array:
		update_grid(json.data)

# ---------------------------------------------------------------------------
# Rendering (Phase 12 — procedural textures)
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

		# Determine material for this cell.
		var mat: String = "stone"
		if i < materials.size():
			mat = materials[i]
		else:
			# Fallback: stone / wood / ice by column.
			var col: int = i % GRID_W
			if col < 3:
				mat = "stone"
			elif col < 7:
				mat = "wood"
			else:
				mat = "ice"

		var base: Sprite2D = blocks[i]["base"]
		var heat: Sprite2D = blocks[i]["heat"]

		# --- Material texture swap ---
		if mat == "stone":
			base.texture = tex_stone
		elif mat == "wood":
			base.texture = tex_wood
		else:
			base.texture = tex_ice

		# --- Structural damage (density → opacity + darkening) ---
		if density_val <= -0.9:
			# Air / destroyed — fully transparent, sky shows through.
			base.modulate = Color(1, 1, 1, 0)
		elif density_val >= 0.95:
			# Full health — pristine, no tint.
			base.modulate = Color(1, 1, 1, 1)
		else:
			# Damaged: map density [-0.9 .. 0.95] → [0 .. 1].
			var health: float = (density_val + 0.9) / 1.85
			health = clampf(health, 0.0, 1.0)
			# Alpha fades as structure crumbles.
			var alpha: float = 0.15 + health * 0.85
			# RGB darkens to simulate cracking / weakening.
			var darken: float = 0.4 + health * 0.6
			base.modulate = Color(darken, darken, darken, alpha)

		# --- Thermodynamics (heat → overlay opacity) ---
		if heat_val < -0.8:
			# Cold — heat overlay invisible.
			heat.modulate = Color(1, 1, 1, 0)
			heat.scale = Vector2(1.0, 1.0)
		else:
			# Map [-0.8 .. 1.0] → [0.0 .. 1.0] for smooth fade-in.
			var heat_t: float = (heat_val + 0.8) / 1.8
			heat_t = clampf(heat_t, 0.0, 1.0)
			heat.modulate = Color(1, 1, 1, heat_t)
			# Scale pulse is handled by _process() for smooth 60fps animation.
