import ctypes
import math
import random
import sys
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective


Vec3i = Tuple[int, int, int]
Vec3f = Tuple[float, float, float]
ChunkKey = Tuple[int, int]


WORLD_X = 96
WORLD_Y = 96
WORLD_Z = 96
CHUNK_SIZE = 16
RENDER_DISTANCE = 70
GRAVITY = 26.0
JUMP_SPEED = 10.0
PLAYER_SPEED = 7.0
MOUSE_SENS = 0.0027
LOOK_POLE_LIMIT = 0.992
PLAYER_RADIUS = 0.35
PLAYER_EYE_OFFSET = 0.62
PLAYER_BODY_LOWER = -0.20
PLAYER_BODY_UPPER = 0.40
PLAYER_HEAD_RADIUS_SCALE = 0.92
REACH = 2.0
GRAVITY_BLEND_POWER = 6.0
FOV_DEG = 75.0
MINE_FUSE_SECONDS = 10.0
MINE_RENDER_DISTANCE = 42.0
MINE_SPHERE_STACKS = 4
MINE_SPHERE_SLICES = 8
PROX_MINE_TRIGGER_RANGE = 2.0
MINE_THROW_SPEED = 8.0
MINE_THROW_UP_BIAS = 1.0
MINE_THROW_COOLDOWN = 1.0
MINE_TIME_DIRT = 0.48
MINE_TIME_STONE = 0.95
MINE_TIME_WOOD = 0.72
BREAK_PARTICLE_COUNT = 20
BREAK_PARTICLE_LIFE = 0.65
BREAK_PARTICLE_SPEED = 3.6
DIRT_PARTICLE_BONUS = 34
EXPLOSION_PARTICLE_COUNT = 190
EXPLOSION_PARTICLE_LIFE = 0.90
EXPLOSION_PARTICLE_SPEED = 13.0
EXPLOSION_SPARK_COUNT = 120
EXPLOSION_RING_POINTS = 64
EXPLOSION_AXIS_SPREAD = 0.14
GAMEPAD_DEADZONE = 0.22
GAMEPAD_LOOK_SPEED = 3.0
MOVE_INPUT_DEADZONE = 0.18
COYOTE_TIME = 0.12
JUMP_BUFFER_TIME = 0.12
FALL_GRAVITY_MULT = 1.18
JUMP_CUT_GRAVITY_MULT = 1.95
PLAYER_MAX_HEALTH = 100.0
HAZARD_CONTACT_DPS = 10.0
HAZARD_TINT = (0.82, 0.92, 0.16)
HAZARD_TEX_INDEX = 4
HAZARD_LAYER_THICKNESS = 2
UP_SMOOTH_RATE = 10.0
DEATH_BLACKOUT_SECONDS = 5.0
DEATH_FADE_SECONDS = 2.0
TITLE_MIN_SECONDS = 3.0

CUBE_HALF = 20
FACE_RELIEF = 10

TEX_SIZE = 16
ATLAS_COLS = 2
ATLAS_ROWS = 3

AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
WOOD = 4
MINE = 5
CHARRED_STONE = 6

BLOCK_COLORS = {
    GRASS: (0.20, 0.18, 0.08),
    DIRT: (0.24, 0.16, 0.10),
    STONE: (0.22, 0.20, 0.18),
    WOOD: (0.25, 0.17, 0.11),
    CHARRED_STONE: (0.05, 0.04, 0.04),
}

FACE_DELTAS = [
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
    ((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 0, 1), [(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]

TRI_IDX = (0, 1, 2, 0, 2, 3)
UV_QUAD = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

BLOCK_TEX_INDEX = {
    GRASS: 0,
    DIRT: 1,
    STONE: 2,
    WOOD: 3,
    CHARRED_STONE: 2,
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hash2(x: int, z: int, seed: int) -> float:
    n = x * 374761393 + z * 668265263 + seed * 700001
    n = (n ^ (n >> 13)) * 1274126177
    n ^= n >> 16
    return (n & 0xFFFF) / 65535.0


def hash3(x: int, y: int, z: int, seed: int) -> float:
    n = x * 73856093 ^ y * 19349663 ^ z * 83492791 ^ (seed * 2654435761)
    n = (n ^ (n >> 13)) * 1274126177
    n ^= n >> 16
    return (n & 0xFFFF) / 65535.0


def value_noise2(x: float, z: float, seed: int) -> float:
    x0 = math.floor(x)
    z0 = math.floor(z)
    tx = x - x0
    tz = z - z0
    u = smoothstep(tx)
    v = smoothstep(tz)

    n00 = hash2(x0, z0, seed)
    n10 = hash2(x0 + 1, z0, seed)
    n01 = hash2(x0, z0 + 1, seed)
    n11 = hash2(x0 + 1, z0 + 1, seed)

    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    return lerp(nx0, nx1, v)


def chunk_for(x: int, z: int) -> ChunkKey:
    return (x // CHUNK_SIZE, z // CHUNK_SIZE)


def v_add(a: Vec3f, b: Vec3f) -> Vec3f:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3f, b: Vec3f) -> Vec3f:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(v: Vec3f, s: float) -> Vec3f:
    return (v[0] * s, v[1] * s, v[2] * s)


def v_dot(a: Vec3f, b: Vec3f) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3f, b: Vec3f) -> Vec3f:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_len(v: Vec3f) -> float:
    return math.sqrt(v_dot(v, v))


def v_norm(v: Vec3f) -> Vec3f:
    mag = v_len(v)
    if mag < 1e-7:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / mag
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def rotate_axis(v: Vec3f, axis: Vec3f, angle_rad: float) -> Vec3f:
    # Rodrigues rotation formula.
    ax = v_norm(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    term1 = v_scale(v, c)
    term2 = v_scale(v_cross(ax, v), s)
    term3 = v_scale(ax, v_dot(ax, v) * (1.0 - c))
    return v_add(v_add(term1, term2), term3)


class World:
    def __init__(self, seed: int = 1337):
        self.seed = seed
        self.blocks: Dict[Vec3i, int] = {}
        self.cx = WORLD_X // 2
        self.cy = WORLD_Y // 2
        self.cz = WORLD_Z // 2

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < WORLD_X and 0 <= y < WORLD_Y and 0 <= z < WORLD_Z

    def block_at(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            return AIR
        return self.blocks.get((x, y, z), AIR)

    def set_block(self, x: int, y: int, z: int, block: int) -> None:
        if not self.in_bounds(x, y, z):
            return
        key = (x, y, z)
        if block == AIR:
            self.blocks.pop(key, None)
        else:
            self.blocks[key] = block

    def _terrain_height(self, u: int, v: int, seed_off: int) -> int:
        n1 = value_noise2(u * 0.085, v * 0.085, self.seed + seed_off)
        n2 = value_noise2(u * 0.19, v * 0.19, self.seed + seed_off * 3 + 7)
        h = 1.5 + n1 * FACE_RELIEF * 0.75 + n2 * FACE_RELIEF * 0.45
        return int(clamp(h, 1, FACE_RELIEF))

    def carve_axis_path(self, axis: int, seed_off: int) -> None:
        start = -(CUBE_HALF + FACE_RELIEF)
        end = CUBE_HALF + FACE_RELIEF
        r = 2

        for t in range(start, end + 1):
            wobble_a = int(round((value_noise2(t * 0.12, seed_off * 0.11, self.seed + seed_off) - 0.5) * 8.0))
            wobble_b = int(round((value_noise2(t * 0.12, seed_off * 0.19, self.seed + seed_off + 17) - 0.5) * 8.0))

            if axis == 0:
                cx = self.cx + t
                cy = self.cy + wobble_a
                cz = self.cz + wobble_b
            elif axis == 1:
                cx = self.cx + wobble_a
                cy = self.cy + t
                cz = self.cz + wobble_b
            else:
                cx = self.cx + wobble_a
                cy = self.cy + wobble_b
                cz = self.cz + t

            for dx in range(-r - 1, r + 2):
                for dy in range(-r - 1, r + 2):
                    for dz in range(-r - 1, r + 2):
                        x = cx + dx
                        y = cy + dy
                        z = cz + dz
                        if not self.in_bounds(x, y, z):
                            continue
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= r * r:
                            self.set_block(x, y, z, AIR)
                        elif d2 <= (r + 1) * (r + 1):
                            self.set_block(x, y, z, DIRT)

            # Leave rubble-like rock intrusions mostly near edges, without sealing the path.
            rubble = value_noise2(t * 0.18, seed_off * 0.07, self.seed + seed_off + 73)
            if rubble > 0.67:
                rx = int(round((value_noise2(t * 0.21, seed_off * 0.03, self.seed + seed_off + 91) - 0.5) * 4.0))
                ry = int(round((value_noise2(t * 0.17, seed_off * 0.09, self.seed + seed_off + 97) - 0.5) * 4.0))
                rz = int(round((value_noise2(t * 0.23, seed_off * 0.15, self.seed + seed_off + 101) - 0.5) * 4.0))
                if axis == 0:
                    px, py, pz = cx, cy + ry, cz + rz
                elif axis == 1:
                    px, py, pz = cx + rx, cy, cz + rz
                else:
                    px, py, pz = cx + rx, cy + ry, cz
                if self.in_bounds(px, py, pz) and (abs(rx) + abs(ry) + abs(rz) >= 2):
                    self.set_block(px, py, pz, STONE)

    def carve_cross_face_paths(self) -> None:
        # Guaranteed traversable routes linking opposite faces through the cube.
        self.carve_axis_path(0, 401)
        self.carve_axis_path(1, 557)
        self.carve_axis_path(2, 709)

    def generate(self) -> None:
        # Solid central cube.
        for x in range(self.cx - CUBE_HALF, self.cx + CUBE_HALF + 1):
            for y in range(self.cy - CUBE_HALF, self.cy + CUBE_HALF + 1):
                for z in range(self.cz - CUBE_HALF, self.cz + CUBE_HALF + 1):
                    if self.in_bounds(x, y, z):
                        self.blocks[(x, y, z)] = STONE

        # Add 3-D terrain relief on each face of the cube.
        faces = [
            ((1, 0, 0), (0, 1, 0), (0, 0, 1), 11),
            ((-1, 0, 0), (0, 1, 0), (0, 0, -1), 23),
            ((0, 1, 0), (1, 0, 0), (0, 0, 1), 31),
            ((0, -1, 0), (1, 0, 0), (0, 0, -1), 41),
            ((0, 0, 1), (1, 0, 0), (0, 1, 0), 53),
            ((0, 0, -1), (-1, 0, 0), (0, 1, 0), 67),
        ]

        for normal, u_axis, v_axis, seed_off in faces:
            for u in range(-CUBE_HALF, CUBE_HALF + 1):
                for v in range(-CUBE_HALF, CUBE_HALF + 1):
                    h = self._terrain_height(u, v, seed_off)
                    sparse_moss = value_noise2(u * 0.22, v * 0.22, self.seed + seed_off + 111)
                    for d in range(h + 1):
                        x = self.cx + normal[0] * (CUBE_HALF + d) + u_axis[0] * u + v_axis[0] * v
                        y = self.cy + normal[1] * (CUBE_HALF + d) + u_axis[1] * u + v_axis[1] * v
                        z = self.cz + normal[2] * (CUBE_HALF + d) + u_axis[2] * u + v_axis[2] * v

                        if not self.in_bounds(x, y, z):
                            continue

                        if d == h:
                            block = GRASS if sparse_moss > 0.84 else DIRT
                        elif d >= h - 3:
                            block = DIRT
                        else:
                            block = STONE
                        self.blocks[(x, y, z)] = block

        self.carve_cross_face_paths()

        # Sparse muddy wood pillars on top-ish areas to preserve build materials.
        for x in range(self.cx - CUBE_HALF + 2, self.cx + CUBE_HALF - 1):
            for z in range(self.cz - CUBE_HALF + 2, self.cz + CUBE_HALF - 1):
                spawn = value_noise2(x * 0.25, z * 0.25, self.seed + 177)
                if spawn > 0.89:
                    top = self.find_surface_y(x, z)
                    if top is not None:
                        for t in range(3):
                            self.set_block(x, top + 1 + t, z, WOOD)

    def find_surface_y(self, x: int, z: int) -> Optional[int]:
        for y in range(WORLD_Y - 2, 1, -1):
            if self.block_at(x, y, z) != AIR:
                return y
        return None


@dataclass
class Player:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    look_x: float = 0.0
    look_y: float = 0.0
    look_z: float = -1.0
    on_ground: bool = False
    mining_target: Optional[Vec3i] = None
    mining_progress: float = 0.0
    break_pulse: float = 0.0
    mouse_settle_frames: int = 0
    coyote_timer: float = 0.0
    jump_buffer_timer: float = 0.0
    jump_was_down: bool = False
    health: float = PLAYER_MAX_HEALTH
    health_display: float = PLAYER_MAX_HEALTH
    health_hit_flash: float = 0.0
    up_x: float = 0.0
    up_y: float = 1.0
    up_z: float = 0.0
    death_fade_timer: float = 0.0
    death_hold_timer: float = 0.0


class ChunkMesh:
    def __init__(self, key: ChunkKey):
        self.key = key
        self.vbo = glGenBuffers(1)
        self.vertex_count = 0

    def upload(self, vertices: List[float]) -> None:
        self.vertex_count = len(vertices) // 11
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if vertices:
            data = array("f", vertices)
            glBufferData(GL_ARRAY_BUFFER, len(data) * 4, data.tobytes(), GL_STATIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)

    def delete(self) -> None:
        glDeleteBuffers(1, [self.vbo])


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()
        self.width = 1280
        self.height = 720
        # Player 1 uses the left half of split-screen; lock mouse to that view.
        self.screen_center = (self.width // 4, self.height // 2)
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Not Minecraft (OpenGL)")
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(self.screen_center)
        pygame.mouse.get_rel()
        pygame.event.clear(pygame.MOUSEMOTION)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.07, 0.04, 0.03, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (3.25, 1.65, 1.02, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.92, 0.40, 0.22, 1.0))
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.6)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.085)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.032)
        glClearColor(0.26, 0.09, 0.07, 1.0)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(FOV_DEG, self.width / self.height, 0.05, 450.0)
        glMatrixMode(GL_MODELVIEW)

        self.texture_atlas = self.create_texture_atlas()

        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("Menlo", 18)
        self.ui_font_small = pygame.font.SysFont("Menlo", 15)
        self.ui_font_title = pygame.font.SysFont("Georgia", 62, bold=True)
        self.ui_font_subtitle = pygame.font.SysFont("Georgia", 28, bold=True)
        self.world = World(seed=2106)
        self.world.generate()

        axis = random.randint(0, 2)
        sign_a = 1 if random.random() < 0.5 else -1
        sign_b = -sign_a
        self.spawn_faces = [(axis, sign_a), (axis, sign_b)]
        all_faces: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
        self.hazard_faces: Set[Tuple[int, int]] = {f for f in all_faces if f not in self.spawn_faces}
        self.hazard_surface_faces: Dict[Tuple[int, int, int, int], int] = {}
        self.rebuild_hazard_surface_cache()
        spawn_a = self.find_spawn_point(axis, sign_a)
        spawn_b = self.find_spawn_point(axis, sign_b)
        self.players: List[Player] = [
            Player(spawn_a[0], spawn_a[1], spawn_a[2], look_x=-1.0, look_y=0.0, look_z=0.0, mouse_settle_frames=3),
            Player(spawn_b[0], spawn_b[1], spawn_b[2], look_x=1.0, look_y=0.0, look_z=0.0, mouse_settle_frames=0),
        ]
        self.reset_player_up(0)
        self.reset_player_up(1)
        self.stabilize_player_spawn(0)
        self.stabilize_player_spawn(1)
        self.reset_player_up(0)
        self.reset_player_up(1)
        self.initial_spawn_points: List[Vec3f] = [self.player_pos(0), self.player_pos(1)]
        self.reset_players_for_match_start()

        self.chunk_meshes: Dict[ChunkKey, ChunkMesh] = {}
        self.dirty_chunks: Set[ChunkKey] = set()
        self.build_all_chunk_meshes()
        self.mines: Dict[Vec3i, Dict[str, object]] = {}
        self.thrown_mines: List[Dict[str, float]] = []
        self.mine_cook_active: List[bool] = [False, False]
        self.mine_cook_time: List[float] = [0.0, 0.0]
        self.next_mine_throw_time: List[float] = [0.0, 0.0]
        self.populate_proximity_mines()

        self.selected_block = DIRT
        self.break_particles: List[Dict[str, float]] = []
        self.explosion_particles: List[Dict[str, float]] = []
        self.gamepads: List[pygame.joystick.Joystick] = []
        self.p1_pad: Optional[pygame.joystick.Joystick] = None
        self.p2_pad: Optional[pygame.joystick.Joystick] = None
        self.p1_pad_slot: Optional[int] = None
        self.p2_pad_slot: Optional[int] = None
        self.pad_prev_buttons: List[Dict[int, bool]] = [{}, {}]
        self.refresh_gamepads()
        self.show_player_lines = False
        self.show_controls_help = False
        self.in_title_screen = True
        self.title_screen_started_at = pygame.time.get_ticks() * 0.001
        self.running = True

    def refresh_gamepads(self) -> None:
        self.gamepads = []
        count = pygame.joystick.get_count()
        for i in range(count):
            js = pygame.joystick.Joystick(i)
            js.init()
            self.gamepads.append(js)

        max_idx = len(self.gamepads) - 1
        if self.p1_pad_slot is not None and self.p1_pad_slot > max_idx:
            self.p1_pad_slot = None
        if self.p2_pad_slot is not None and self.p2_pad_slot > max_idx:
            self.p2_pad_slot = None

        if len(self.gamepads) == 0:
            self.p1_pad_slot = None
            self.p2_pad_slot = None
        elif self.p1_pad_slot is None and self.p2_pad_slot is None:
            self.auto_assign_gamepads()
        else:
            self.ensure_valid_pad_assignments()
        self.apply_pad_assignments()

    def auto_assign_gamepads(self) -> None:
        self.p1_pad_slot = None
        self.p2_pad_slot = 0 if len(self.gamepads) >= 1 else None

    def ensure_valid_pad_assignments(self) -> None:
        self.p1_pad_slot = None
        if len(self.gamepads) == 0:
            self.p2_pad_slot = None
            return

        if self.p2_pad_slot is not None and self.p2_pad_slot >= len(self.gamepads):
            self.p2_pad_slot = None

        if self.p2_pad_slot is None and len(self.gamepads) > 0:
            self.p2_pad_slot = 0

    def apply_pad_assignments(self) -> None:
        self.p1_pad = None
        self.p2_pad = self.gamepads[self.p2_pad_slot] if self.p2_pad_slot is not None and self.p2_pad_slot < len(self.gamepads) else None

    def cycle_pad_slot(self, player_idx: int, step: int) -> None:
        if player_idx == 0:
            return
        slots: List[Optional[int]]
        other_slot: Optional[int]
        current: Optional[int]
        other_slot = self.p1_pad_slot
        current = self.p2_pad_slot
        slots = [i for i in range(len(self.gamepads)) if i != other_slot]
        if not slots:
            slots = [None]

        if not slots:
            return
        if current not in slots:
            current = slots[0]
        idx = slots.index(current)
        nxt = slots[(idx + step) % len(slots)]
        self.p2_pad_slot = nxt
        if self.p2_pad_slot is None and len(self.gamepads) > 0:
            self.p2_pad_slot = 0
        self.ensure_valid_pad_assignments()
        self.apply_pad_assignments()

    def read_axis(self, pad: Optional[pygame.joystick.Joystick], idx: int) -> float:
        if pad is None or idx >= pad.get_numaxes():
            return 0.0
        v = float(pad.get_axis(idx))
        if abs(v) < GAMEPAD_DEADZONE:
            return 0.0
        return v

    def read_button(self, pad: Optional[pygame.joystick.Joystick], idx: int) -> bool:
        return bool(pad is not None and idx < pad.get_numbuttons() and pad.get_button(idx))

    def edge_button(self, player_idx: int, btn: int, now: bool) -> bool:
        prev = self.pad_prev_buttons[player_idx].get(btn, False)
        self.pad_prev_buttons[player_idx][btn] = now
        return now and not prev

    def make_lofi_tile(self, seed: int, base: Tuple[int, int, int], hi: Tuple[int, int, int], lo: Tuple[int, int, int]) -> bytearray:
        tile = bytearray(TEX_SIZE * TEX_SIZE * 3)
        idx = 0
        for y in range(TEX_SIZE):
            for x in range(TEX_SIZE):
                n1 = hash3(x, y, seed, seed + 11)
                n2 = hash3(x // 2, y // 2, seed + 31, seed + 47)
                n3 = hash3(x + y, y - x, seed + 61, seed + 3)
                if n1 > 0.72:
                    c = hi
                elif n2 < 0.24 or (y % 3 == 0 and n3 > 0.35):
                    c = lo
                else:
                    c = base
                tile[idx] = c[0]
                tile[idx + 1] = c[1]
                tile[idx + 2] = c[2]
                idx += 3
        return tile

    def create_texture_atlas(self) -> int:
        atlas_w = TEX_SIZE * ATLAS_COLS
        atlas_h = TEX_SIZE * ATLAS_ROWS
        atlas = bytearray(atlas_w * atlas_h * 3)

        tiles = {
            0: self.make_lofi_tile(901, (56, 46, 22), (88, 66, 28), (36, 28, 14)),
            1: self.make_lofi_tile(903, (78, 42, 28), (118, 60, 36), (49, 24, 17)),
            2: self.make_lofi_tile(907, (64, 54, 49), (88, 74, 66), (40, 32, 29)),
            3: self.make_lofi_tile(911, (82, 50, 30), (112, 70, 42), (52, 31, 19)),
            4: self.make_lofi_tile(947, (108, 138, 22), (190, 164, 34), (58, 84, 14)),
        }

        for tile_idx, tile_data in tiles.items():
            tx = tile_idx % ATLAS_COLS
            ty = tile_idx // ATLAS_COLS
            for y in range(TEX_SIZE):
                src_row = y * TEX_SIZE * 3
                dst_y = ty * TEX_SIZE + y
                dst_row = (dst_y * atlas_w + tx * TEX_SIZE) * 3
                atlas[dst_row:dst_row + TEX_SIZE * 3] = tile_data[src_row:src_row + TEX_SIZE * 3]

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, atlas_w, atlas_h, 0, GL_RGB, GL_UNSIGNED_BYTE, bytes(atlas))
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex

    def atlas_uv(self, tex_idx: int, u: float, v: float) -> Tuple[float, float]:
        tx = tex_idx % ATLAS_COLS
        ty = tex_idx // ATLAS_COLS
        return ((tx + u) / ATLAS_COLS, (ty + v) / ATLAS_ROWS)

    def chunk_bounds(self, key: ChunkKey) -> Tuple[int, int, int, int]:
        cx, cz = key
        x0 = cx * CHUNK_SIZE
        z0 = cz * CHUNK_SIZE
        return (x0, min(x0 + CHUNK_SIZE, WORLD_X), z0, min(z0 + CHUNK_SIZE, WORLD_Z))

    def build_all_chunk_meshes(self) -> None:
        for cx in range((WORLD_X + CHUNK_SIZE - 1) // CHUNK_SIZE):
            for cz in range((WORLD_Z + CHUNK_SIZE - 1) // CHUNK_SIZE):
                key = (cx, cz)
                self.chunk_meshes[key] = ChunkMesh(key)
                self.rebuild_chunk_mesh(key)

    def rebuild_chunk_mesh(self, key: ChunkKey) -> None:
        mesh = self.chunk_meshes.get(key)
        if mesh is None:
            mesh = ChunkMesh(key)
            self.chunk_meshes[key] = mesh

        x0, x1, z0, z1 = self.chunk_bounds(key)
        verts: List[float] = []

        for x in range(x0, x1):
            for z in range(z0, z1):
                for y in range(WORLD_Y):
                    block = self.world.block_at(x, y, z)
                    if block == AIR:
                        continue

                    tex_idx = BLOCK_TEX_INDEX[block]
                    base = BLOCK_COLORS[block]

                    for (nx, ny, nz), face in FACE_DELTAS:
                        if self.world.block_at(x + nx, y + ny, z + nz) != AIR:
                            continue

                        shade = 1.0
                        if ny == -1:
                            shade = 0.62
                        elif ny == 0:
                            shade = 0.82

                        face_tex_idx = tex_idx
                        r = base[0] * shade
                        g = base[1] * shade
                        b = base[2] * shade
                        if self.is_hazard_outer_face(x, y, z, nx, ny, nz):
                            face_tex_idx = HAZARD_TEX_INDEX
                            r = lerp(r, HAZARD_TINT[0], 0.95)
                            g = lerp(g, HAZARD_TINT[1], 0.98)
                            b = lerp(b, HAZARD_TINT[2], 0.92)

                        for idx in TRI_IDX:
                            vx, vy, vz = face[idx]
                            uv = UV_QUAD[idx]
                            uu, vv = self.atlas_uv(face_tex_idx, uv[0], uv[1])
                            verts.extend((x + vx, y + vy, z + vz, nx, ny, nz, uu, vv, r, g, b))

        mesh.upload(verts)

    def is_hazard_outer_face(self, x: int, y: int, z: int, nx: int, ny: int, nz: int) -> bool:
        if nx != 0:
            axis = 0
            sign = 1 if nx > 0 else -1
            face_coord = x + 1 if nx > 0 else x
            center = self.world.cx
        elif ny != 0:
            axis = 1
            sign = 1 if ny > 0 else -1
            face_coord = y + 1 if ny > 0 else y
            center = self.world.cy
        else:
            axis = 2
            sign = 1 if nz > 0 else -1
            face_coord = z + 1 if nz > 0 else z
            center = self.world.cz

        if (axis, sign) not in self.hazard_faces:
            return False

        # Restrict toxicity to the true outer cube face for this axis/sign,
        # not arbitrary exposed side faces from local terrain relief.
        fx = x + 0.5 + 0.5 * nx
        fy = y + 0.5 + 0.5 * ny
        fz = z + 0.5 + 0.5 * nz
        rel = (fx - self.world.cx, fy - self.world.cy, fz - self.world.cz)
        dom_axis = 0
        if abs(rel[1]) > abs(rel[dom_axis]):
            dom_axis = 1
        if abs(rel[2]) > abs(rel[dom_axis]):
            dom_axis = 2
        dom_sign = 1 if rel[dom_axis] >= 0.0 else -1
        if dom_axis != axis or dom_sign != sign:
            return False

        key = self.hazard_surface_key(axis, sign, x, y, z)
        if key is None:
            return False
        surface_face = self.hazard_surface_faces.get(key)
        if surface_face is None:
            return False
        depth = sign * (surface_face - face_coord)
        return 0 <= depth < HAZARD_LAYER_THICKNESS

    def hazard_surface_key(self, axis: int, sign: int, x: int, y: int, z: int) -> Optional[Tuple[int, int, int, int]]:
        _ = sign
        if axis == 0:
            return (axis, sign, y, z)
        if axis == 1:
            return (axis, sign, x, z)
        if axis == 2:
            return (axis, sign, x, y)
        return None

    def rebuild_hazard_surface_cache(self) -> None:
        self.hazard_surface_faces = {}
        if not self.hazard_faces:
            return

        for axis, sign in self.hazard_faces:
            if axis == 0:
                for y in range(WORLD_Y):
                    for z in range(WORLD_Z):
                        if sign > 0:
                            for x in range(WORLD_X - 1, -1, -1):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, y, z)] = x + 1
                                    break
                        else:
                            for x in range(WORLD_X):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, y, z)] = x
                                    break
            elif axis == 1:
                for x in range(WORLD_X):
                    for z in range(WORLD_Z):
                        if sign > 0:
                            for y in range(WORLD_Y - 1, -1, -1):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, x, z)] = y + 1
                                    break
                        else:
                            for y in range(WORLD_Y):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, x, z)] = y
                                    break
            else:
                for x in range(WORLD_X):
                    for y in range(WORLD_Y):
                        if sign > 0:
                            for z in range(WORLD_Z - 1, -1, -1):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, x, y)] = z + 1
                                    break
                        else:
                            for z in range(WORLD_Z):
                                if self.world.block_at(x, y, z) != AIR:
                                    self.hazard_surface_faces[(axis, sign, x, y)] = z
                                    break

    def mark_chunk_dirty(self, key: ChunkKey) -> None:
        cx, cz = key
        max_cx = (WORLD_X + CHUNK_SIZE - 1) // CHUNK_SIZE
        max_cz = (WORLD_Z + CHUNK_SIZE - 1) // CHUNK_SIZE
        if 0 <= cx < max_cx and 0 <= cz < max_cz:
            self.dirty_chunks.add(key)

    def mark_neighbors_dirty(self, x: int, z: int) -> None:
        key = chunk_for(x, z)
        self.mark_chunk_dirty(key)

        lx = x % CHUNK_SIZE
        lz = z % CHUNK_SIZE
        if lx == 0:
            self.mark_chunk_dirty((key[0] - 1, key[1]))
        elif lx == CHUNK_SIZE - 1:
            self.mark_chunk_dirty((key[0] + 1, key[1]))

        if lz == 0:
            self.mark_chunk_dirty((key[0], key[1] - 1))
        elif lz == CHUNK_SIZE - 1:
            self.mark_chunk_dirty((key[0], key[1] + 1))

    def set_world_block(self, x: int, y: int, z: int, block: int) -> None:
        self.world.set_block(x, y, z, block)
        self.mark_neighbors_dirty(x, z)

    def update_dirty_meshes(self, per_frame: int = 4) -> None:
        for _ in range(per_frame):
            if not self.dirty_chunks:
                break
            key = self.dirty_chunks.pop()
            self.rebuild_chunk_mesh(key)

    def player_pos(self, idx: int) -> Vec3f:
        p = self.players[idx]
        return (p.x, p.y, p.z)

    def set_player_pos(self, idx: int, p: Vec3f) -> None:
        self.players[idx].x, self.players[idx].y, self.players[idx].z = p

    def player_vel(self, idx: int) -> Vec3f:
        p = self.players[idx]
        return (p.vx, p.vy, p.vz)

    def set_player_vel(self, idx: int, v: Vec3f) -> None:
        self.players[idx].vx, self.players[idx].vy, self.players[idx].vz = v

    def player_alive(self, idx: int) -> bool:
        p = self.players[idx]
        return p.death_fade_timer <= 0.0 and p.death_hold_timer <= 0.0

    def look_dir(self, idx: int) -> Vec3f:
        p = self.players[idx]
        return v_norm((p.look_x, p.look_y, p.look_z))

    def set_look_dir(self, idx: int, d: Vec3f) -> None:
        dn = v_norm(d)
        self.players[idx].look_x, self.players[idx].look_y, self.players[idx].look_z = dn

    def snap_axis_dir(self, vec: Vec3f, avoid: Tuple[Vec3i, ...] = ()) -> Vec3i:
        candidates = [
            (abs(vec[0]), (1 if vec[0] >= 0 else -1, 0, 0)),
            (abs(vec[1]), (0, 1 if vec[1] >= 0 else -1, 0)),
            (abs(vec[2]), (0, 0, 1 if vec[2] >= 0 else -1)),
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, axis in candidates:
            blocked = False
            for a in avoid:
                if abs(axis[0] * a[0] + axis[1] * a[1] + axis[2] * a[2]) == 1:
                    blocked = True
                    break
            if not blocked:
                return axis
        return candidates[0][1]

    def place_mine(self, player_idx: int, cooked_time: float = 0.0) -> None:
        if not self.player_alive(player_idx):
            return
        now = pygame.time.get_ticks() * 0.001
        if now < self.next_mine_throw_time[player_idx]:
            return
        pos = self.player_pos(player_idx)
        fwd, right, up = self.camera_basis(player_idx, pos)
        up_i = self.snap_axis_dir(up)
        right_i = self.snap_axis_dir(right, (up_i,))
        fwd_i = self.snap_axis_dir(fwd, (up_i, right_i))
        eye = self.eye_pos(player_idx)
        launch_pos = v_add(eye, v_add(v_scale(fwd, 0.75), v_scale(up, -0.08)))
        launch_vel = v_add(v_scale(fwd, MINE_THROW_SPEED), v_scale(up, MINE_THROW_UP_BIAS))
        remaining_fuse = max(0.05, MINE_FUSE_SECONDS - max(0.0, cooked_time))
        self.thrown_mines.append(
            {
                "x": launch_pos[0],
                "y": launch_pos[1],
                "z": launch_pos[2],
                "vx": launch_vel[0],
                "vy": launch_vel[1],
                "vz": launch_vel[2],
                "owner": float(player_idx),
                "upx": float(up_i[0]),
                "upy": float(up_i[1]),
                "upz": float(up_i[2]),
                "rightx": float(right_i[0]),
                "righty": float(right_i[1]),
                "rightz": float(right_i[2]),
                "fwdx": float(fwd_i[0]),
                "fwdy": float(fwd_i[1]),
                "fwdz": float(fwd_i[2]),
                "fuse": remaining_fuse,
                "age": 0.0,
            }
        )
        self.next_mine_throw_time[player_idx] = now + MINE_THROW_COOLDOWN

    def update_mine_throw(self, player_idx: int, dt: float, throw_held: bool) -> None:
        if not self.player_alive(player_idx):
            self.mine_cook_active[player_idx] = False
            self.mine_cook_time[player_idx] = 0.0
            return

        now = pygame.time.get_ticks() * 0.001
        if throw_held:
            if not self.mine_cook_active[player_idx]:
                if now >= self.next_mine_throw_time[player_idx]:
                    self.mine_cook_active[player_idx] = True
                    self.mine_cook_time[player_idx] = 0.0
            else:
                self.mine_cook_time[player_idx] = clamp(
                    self.mine_cook_time[player_idx] + dt,
                    0.0,
                    MINE_FUSE_SECONDS,
                )
            return

        if self.mine_cook_active[player_idx]:
            self.place_mine(player_idx, self.mine_cook_time[player_idx])
            self.mine_cook_active[player_idx] = False
            self.mine_cook_time[player_idx] = 0.0

    def arm_thrown_mine(self, thrown: Dict[str, float], key: Vec3i, support: Vec3i, normal: Vec3i) -> bool:
        if not self.world.in_bounds(*key) or not self.world.in_bounds(*support):
            return False
        if self.world.block_at(*key) != AIR or self.world.block_at(*support) == AIR:
            return False
        if key in self.mines:
            return False
        owner = int(thrown.get("owner", -1))
        if 0 <= owner < len(self.players) and self.player_intersects(owner, *key):
            return False

        up_i = (
            int(round(thrown.get("upx", 0.0))),
            int(round(thrown.get("upy", 1.0))),
            int(round(thrown.get("upz", 0.0))),
        )
        right_i = (
            int(round(thrown.get("rightx", 1.0))),
            int(round(thrown.get("righty", 0.0))),
            int(round(thrown.get("rightz", 0.0))),
        )
        fwd_i = (
            int(round(thrown.get("fwdx", 0.0))),
            int(round(thrown.get("fwdy", 0.0))),
            int(round(thrown.get("fwdz", 1.0))),
        )

        self.mines[key] = {
            "kind": "timed",
            "owner": owner,
            "pos": key,
            "timer": float(thrown.get("fuse", MINE_FUSE_SECONDS)),
            "up": up_i,
            "right": right_i,
            "forward": fwd_i,
            "normal": normal,
            "support": support,
        }
        return True

    def populate_proximity_mines(self) -> None:
        if not self.hazard_faces:
            return
        for axis, sign in self.hazard_faces:
            placed = False
            tries = 0
            while tries < 160:
                tries += 1
                support = self.random_face_support(axis, sign)
                if support is None:
                    continue
                if self.try_place_proximity_mine(axis, sign, support):
                    placed = True
                    break
            if placed:
                continue

            # Deterministic fallback so each toxic face always gets one mine.
            support = self.find_face_support_fallback(axis, sign)
            if support is not None:
                self.try_place_proximity_mine(axis, sign, support)

    def try_place_proximity_mine(self, axis: int, sign: int, support: Vec3i) -> bool:
        mine_pos = (
            support[0] + (sign if axis == 0 else 0),
            support[1] + (sign if axis == 1 else 0),
            support[2] + (sign if axis == 2 else 0),
        )
        if not self.world.in_bounds(*mine_pos):
            return False
        if self.world.block_at(*mine_pos) != AIR or mine_pos in self.mines:
            return False
        if any(self.player_intersects(i, *mine_pos) for i in range(len(self.players))):
            return False

        n = (sign if axis == 0 else 0, sign if axis == 1 else 0, sign if axis == 2 else 0)
        ref = (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0)
        tangent_u = v_norm(v_cross(ref, (float(n[0]), float(n[1]), float(n[2]))))
        tangent_v = v_norm(v_cross((float(n[0]), float(n[1]), float(n[2])), tangent_u))
        up_i = self.snap_axis_dir((float(n[0]), float(n[1]), float(n[2])))
        right_i = self.snap_axis_dir(tangent_u, (up_i,))
        fwd_i = self.snap_axis_dir(tangent_v, (up_i, right_i))

        self.mines[mine_pos] = {
            "kind": "proximity",
            "owner": -1,
            "pos": mine_pos,
            "normal": n,
            "support": support,
            "up": up_i,
            "right": right_i,
            "forward": fwd_i,
        }
        return True

    def find_face_support_fallback(self, axis: int, sign: int) -> Optional[Vec3i]:
        center = [self.world.cx, self.world.cy, self.world.cz]
        outer = int(round(clamp(center[axis] + sign * (CUBE_HALF + FACE_RELIEF + 1), 0, (WORLD_X if axis == 0 else WORLD_Y if axis == 1 else WORLD_Z) - 1)))
        other_axes = [0, 1, 2]
        other_axes.remove(axis)
        max_step = max(WORLD_X, WORLD_Y, WORLD_Z)
        for radius in range(max_step):
            for du in range(-radius, radius + 1):
                for dv in range(-radius, radius + 1):
                    if max(abs(du), abs(dv)) != radius:
                        continue
                    coords = [0, 0, 0]
                    coords[other_axes[0]] = clamp(center[other_axes[0]] + du, 1, (WORLD_X if other_axes[0] == 0 else WORLD_Y if other_axes[0] == 1 else WORLD_Z) - 2)
                    coords[other_axes[1]] = clamp(center[other_axes[1]] + dv, 1, (WORLD_X if other_axes[1] == 0 else WORLD_Y if other_axes[1] == 1 else WORLD_Z) - 2)
                    for step in range(max_step):
                        coords[axis] = outer - sign * step
                        x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
                        if not self.world.in_bounds(x, y, z):
                            continue
                        if self.world.block_at(x, y, z) != AIR:
                            return (x, y, z)
        return None

    def mine_anchor_cell(self, mine: Dict[str, object]) -> Optional[Vec3i]:
        support = mine.get("support")
        if not isinstance(support, tuple):
            return None
        return support

    def mine_has_support(self, mine: Dict[str, object]) -> bool:
        anchor = self.mine_anchor_cell(mine)
        if anchor is None or not self.world.in_bounds(*anchor):
            return False
        return self.world.block_at(*anchor) != AIR

    def detonate_mine(self, mine: Dict[str, object]) -> None:
        pos = mine["pos"]
        if not isinstance(pos, tuple):
            return

        if pos not in self.mines:
            return
        self.mines.pop(pos, None)

        fx_owner_obj = mine.get("owner")
        fx_owner = int(fx_owner_obj) if isinstance(fx_owner_obj, int) and 0 <= fx_owner_obj < len(self.players) else None
        cells: Set[Vec3i] = set()
        cell_dirs: Dict[Vec3i, Vec3i] = {}
        up = mine["up"]
        right = mine["right"]
        fwd = mine["forward"]
        if not isinstance(up, tuple) or not isinstance(right, tuple) or not isinstance(fwd, tuple):
            return

        cells.add(pos)
        cell_dirs[pos] = (0, 0, 0)
        for i in range(1, 4):
            c_up = (pos[0] + up[0] * i, pos[1] + up[1] * i, pos[2] + up[2] * i)
            c_dn = (pos[0] - up[0] * i, pos[1] - up[1] * i, pos[2] - up[2] * i)
            cells.add(c_up)
            cells.add(c_dn)
            cell_dirs.setdefault(c_up, up)
            cell_dirs.setdefault(c_dn, (-up[0], -up[1], -up[2]))

        # Bomberman-style rays: travel through empty cells up to 10, then include
        # (and stop at) the first obstruction (+1 cell when blocked).
        ray_dirs = (
            right,
            (-right[0], -right[1], -right[2]),
            fwd,
            (-fwd[0], -fwd[1], -fwd[2]),
        )
        for d in ray_dirs:
            for i in range(1, 12):
                cell = (pos[0] + d[0] * i, pos[1] + d[1] * i, pos[2] + d[2] * i)
                if not self.world.in_bounds(*cell):
                    break
                block = self.world.block_at(*cell)
                if i <= 10:
                    cells.add(cell)
                    cell_dirs.setdefault(cell, d)
                    if block != AIR:
                        break
                else:
                    # i == 11: only include if this is the first obstruction.
                    if block != AIR:
                        cells.add(cell)
                        cell_dirs.setdefault(cell, d)
                    break

        hit_players: Set[int] = set()
        for cx, cy, cz in cells:
            if not self.world.in_bounds(cx, cy, cz):
                continue
            block = self.world.block_at(cx, cy, cz)
            for i in range(len(self.players)):
                if not self.player_alive(i):
                    continue
                if self.player_intersects(i, cx, cy, cz) or self.player_in_blast_cell(i, cx, cy, cz):
                    hit_players.add(i)

        for i in hit_players:
            self.kill_player(i)

        # Visual blast follows each mine's damage pattern.
        self.spawn_explosion_effect((pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5), scale=0.55, axis=None, player_idx=fx_owner)
        for (cx, cy, cz), axis_dir in cell_dirs.items():
            if (cx, cy, cz) == pos:
                continue
            dist = abs(cx - pos[0]) + abs(cy - pos[1]) + abs(cz - pos[2])
            scale = max(0.22, 1.0 - dist * 0.06)
            self.spawn_explosion_effect((cx + 0.5, cy + 0.5, cz + 0.5), scale=scale, axis=axis_dir, player_idx=fx_owner)

        for cx, cy, cz in cells:
            if not self.world.in_bounds(cx, cy, cz):
                continue
            block = self.world.block_at(cx, cy, cz)
            if block == AIR:
                continue
            if block == STONE:
                self.set_world_block(cx, cy, cz, CHARRED_STONE)
            else:
                self.set_world_block(cx, cy, cz, AIR)
            self.spawn_break_feedback((cx, cy, cz), block, fx_owner)

        # Chain reaction: any armed mine inside blast cells detonates immediately.
        chained_positions = [cell for cell in cells if cell in self.mines]
        for chained_pos in chained_positions:
            chained = self.mines.get(chained_pos)
            if chained is not None:
                self.detonate_mine(chained)

    def kill_player(self, player_idx: int) -> None:
        p = self.players[player_idx]
        if not self.player_alive(player_idx):
            return
        p.vx = p.vy = p.vz = 0.0
        p.on_ground = False
        p.mining_target = None
        p.mining_progress = 0.0
        p.break_pulse = 0.0
        p.health = 0.0
        p.death_fade_timer = DEATH_FADE_SECONDS
        p.death_hold_timer = DEATH_BLACKOUT_SECONDS

    def respawn_player(self, player_idx: int) -> None:
        spawn = self.initial_spawn_points[player_idx]
        p = self.players[player_idx]
        p.x, p.y, p.z = spawn
        p.vx = p.vy = p.vz = 0.0
        p.on_ground = False
        p.mining_target = None
        p.mining_progress = 0.0
        p.break_pulse = 0.0
        p.health = PLAYER_MAX_HEALTH
        p.health_display = PLAYER_MAX_HEALTH
        p.health_hit_flash = 0.0
        p.death_fade_timer = 0.0
        p.death_hold_timer = 0.0
        self.reset_player_up(player_idx)
        self.stabilize_player_spawn(player_idx)
        self.reset_player_up(player_idx)

    def update_respawns(self, dt: float) -> None:
        for i, p in enumerate(self.players):
            if self.player_alive(i):
                continue
            if p.death_fade_timer > 0.0:
                p.death_fade_timer = max(0.0, p.death_fade_timer - dt)
            elif p.death_hold_timer > 0.0:
                p.death_hold_timer = max(0.0, p.death_hold_timer - dt)
            if p.death_fade_timer <= 0.0 and p.death_hold_timer <= 0.0:
                self.respawn_player(i)

    def update_health_feedback(self, dt: float) -> None:
        for p in self.players:
            target = clamp(p.health, 0.0, PLAYER_MAX_HEALTH)
            if target < p.health_display:
                p.health_hit_flash = max(p.health_hit_flash, 1.0)
                # Keep a visible lag so HP drops are unmistakable.
                p.health_display = max(target, p.health_display - dt * 28.0)
            else:
                p.health_display = target
            if p.health_hit_flash > 0.0:
                p.health_hit_flash = max(0.0, p.health_hit_flash - dt * 1.5)

    def update_mines(self, dt: float) -> None:
        if self.thrown_mines:
            still_flying: List[Dict[str, float]] = []
            for mine in self.thrown_mines:
                mine["age"] = float(mine.get("age", 0.0)) + dt
                mine["fuse"] = float(mine.get("fuse", MINE_FUSE_SECONDS)) - dt
                prev = (mine["x"], mine["y"], mine["z"])
                gdir = self.gravity_dir(prev)
                mine["vx"] += gdir[0] * GRAVITY * dt
                mine["vy"] += gdir[1] * GRAVITY * dt
                mine["vz"] += gdir[2] * GRAVITY * dt
                cur = (
                    prev[0] + mine["vx"] * dt,
                    prev[1] + mine["vy"] * dt,
                    prev[2] + mine["vz"] * dt,
                )
                cur_cell = (math.floor(cur[0]), math.floor(cur[1]), math.floor(cur[2]))
                prev_cell = (math.floor(prev[0]), math.floor(prev[1]), math.floor(prev[2]))

                if not self.world.in_bounds(*cur_cell):
                    continue

                hit_block = self.world.block_at(*cur_cell) != AIR
                if hit_block:
                    if prev_cell != cur_cell and self.world.in_bounds(*prev_cell):
                        diff = (prev_cell[0] - cur_cell[0], prev_cell[1] - cur_cell[1], prev_cell[2] - cur_cell[2])
                        if abs(diff[0]) + abs(diff[1]) + abs(diff[2]) != 1:
                            normal = self.snap_axis_dir((prev[0] - cur[0], prev[1] - cur[1], prev[2] - cur[2]))
                        else:
                            normal = diff
                        if self.arm_thrown_mine(mine, prev_cell, cur_cell, normal):
                            continue

                    n = self.snap_axis_dir((prev[0] - cur[0], prev[1] - cur[1], prev[2] - cur[2]))
                    fallback_key = (cur_cell[0] + n[0], cur_cell[1] + n[1], cur_cell[2] + n[2])
                    if self.arm_thrown_mine(mine, fallback_key, cur_cell, n):
                        continue
                    continue

                # If moving through air but adjacent to any solid surface, allow
                # mine to settle onto that surface after a brief travel window.
                if mine["age"] > 0.08:
                    dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
                    settle_normals: List[Vec3i] = []
                    for d in dirs:
                        support = (cur_cell[0] - d[0], cur_cell[1] - d[1], cur_cell[2] - d[2])
                        if not self.world.in_bounds(*support):
                            continue
                        if self.world.block_at(*support) != AIR:
                            settle_normals.append(d)
                    if settle_normals:
                        # Prefer surface opposite gravity so floor landings feel natural.
                        down = self.gravity_dir(cur)
                        down_i = self.snap_axis_dir(down)
                        best = settle_normals[0]
                        best_score = -999.0
                        for n in settle_normals:
                            score = n[0] * (-down_i[0]) + n[1] * (-down_i[1]) + n[2] * (-down_i[2])
                            if score > best_score:
                                best_score = score
                                best = n
                        support = (cur_cell[0] - best[0], cur_cell[1] - best[1], cur_cell[2] - best[2])
                        if self.arm_thrown_mine(mine, cur_cell, support, best):
                            continue

                mine["x"], mine["y"], mine["z"] = cur
                still_flying.append(mine)
            self.thrown_mines = still_flying

        if not self.mines:
            return
        unsupported: List[Vec3i] = []
        to_detonate: List[Dict[str, object]] = []
        for mine in list(self.mines.values()):
            if not self.mine_has_support(mine):
                pos = mine.get("pos")
                if isinstance(pos, tuple):
                    unsupported.append(pos)
                continue
            kind = str(mine.get("kind", "timed"))
            if kind == "proximity":
                pos = mine.get("pos")
                if not isinstance(pos, tuple):
                    continue
                center = (pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5)
                for p in self.players:
                    d2 = (p.x - center[0]) ** 2 + (p.y - center[1]) ** 2 + (p.z - center[2]) ** 2
                    if d2 <= PROX_MINE_TRIGGER_RANGE * PROX_MINE_TRIGGER_RANGE:
                        to_detonate.append(mine)
                        break
            else:
                t = float(mine["timer"]) - dt
                mine["timer"] = t
                if t <= 0.0:
                    to_detonate.append(mine)

        for pos in unsupported:
            self.mines.pop(pos, None)

        for mine in to_detonate:
            self.detonate_mine(mine)

    def face_for_position(self, pos: Vec3f) -> Tuple[int, int]:
        rel = (pos[0] - self.world.cx, pos[1] - self.world.cy, pos[2] - self.world.cz)
        axis = 0
        if abs(rel[1]) > abs(rel[axis]):
            axis = 1
        if abs(rel[2]) > abs(rel[axis]):
            axis = 2
        sign = 1 if rel[axis] >= 0.0 else -1
        return (axis, sign)

    def update_hazard_damage(self, dt: float) -> None:
        for i, p in enumerate(self.players):
            if not self.player_alive(i):
                continue
            pos = self.player_pos(i)
            face = self.face_for_position(pos)
            if face not in self.hazard_faces:
                continue
            g = self.player_gravity_dir(i, pos)
            # Robust contact test: use movement-grounded state plus two collision probes.
            touching = (
                p.on_ground
                or self.collides_body(v_add(pos, v_scale(g, 0.10)))
                or self.collides_body(v_add(pos, v_scale(g, 0.24)))
            )
            if not touching:
                continue
            p.health -= HAZARD_CONTACT_DPS * dt
            if p.health <= 0.0:
                self.kill_player(i)

    def outward_up(self, pos: Vec3f) -> Vec3f:
        rel = (pos[0] - self.world.cx, pos[1] - self.world.cy, pos[2] - self.world.cz)
        ax = abs(rel[0]) + 1e-6
        ay = abs(rel[1]) + 1e-6
        az = abs(rel[2]) + 1e-6
        m = max(ax, ay, az)
        nx = math.copysign((ax / m) ** GRAVITY_BLEND_POWER, rel[0])
        ny = math.copysign((ay / m) ** GRAVITY_BLEND_POWER, rel[1])
        nz = math.copysign((az / m) ** GRAVITY_BLEND_POWER, rel[2])
        return v_norm((nx, ny, nz))

    def reset_player_up(self, player_idx: int) -> None:
        up = self.outward_up(self.player_pos(player_idx))
        p = self.players[player_idx]
        p.up_x, p.up_y, p.up_z = up

    def player_up(self, player_idx: int, pos: Optional[Vec3f] = None) -> Vec3f:
        p = self.players[player_idx]
        up = v_norm((p.up_x, p.up_y, p.up_z))
        if v_len(up) < 1e-6:
            return self.outward_up(self.player_pos(player_idx) if pos is None else pos)
        return up

    def update_player_up_smoothing(self, dt: float) -> None:
        alpha = 1.0 - math.exp(-UP_SMOOTH_RATE * dt)
        for i, p in enumerate(self.players):
            pos = self.player_pos(i)
            target = self.outward_up(pos)
            cur = self.player_up(i, pos)
            blended = v_norm(v_add(v_scale(cur, 1.0 - alpha), v_scale(target, alpha)))
            p.up_x, p.up_y, p.up_z = blended

    def gravity_dir(self, pos: Vec3f) -> Vec3f:
        up = self.outward_up(pos)
        return (-up[0], -up[1], -up[2])

    def player_gravity_dir(self, player_idx: int, pos: Vec3f) -> Vec3f:
        up = self.player_up(player_idx, pos)
        return (-up[0], -up[1], -up[2])

    def camera_basis(self, player_idx: int, pos: Vec3f) -> Tuple[Vec3f, Vec3f, Vec3f]:
        up = self.player_up(player_idx, pos)
        look = self.look_dir(player_idx)

        fwd = v_sub(look, v_scale(up, v_dot(look, up)))
        if v_len(fwd) < 1e-5:
            seed = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 0.0, 1.0)
            fwd = v_cross(seed, up)
        fwd = v_norm(fwd)

        right = v_norm(v_cross(fwd, up))
        return (fwd, right, up)

    def eye_pos(self, player_idx: int) -> Vec3f:
        pos = self.player_pos(player_idx)
        up = self.player_up(player_idx, pos)
        return v_add(pos, v_scale(up, PLAYER_EYE_OFFSET))

    def handle_mouse_look(self) -> None:
        if not self.player_alive(0):
            pygame.mouse.set_pos(self.screen_center)
            pygame.mouse.get_rel()
            return
        p = self.players[0]
        if p.mouse_settle_frames > 0:
            pygame.mouse.set_pos(self.screen_center)
            pygame.event.clear(pygame.MOUSEMOTION)
            pygame.mouse.get_rel()
            p.mouse_settle_frames -= 1
            return

        mx_pos, my_pos = pygame.mouse.get_pos()
        mx = mx_pos - self.screen_center[0]
        my = my_pos - self.screen_center[1]
        pygame.mouse.set_pos(self.screen_center)
        if mx == 0 and my == 0:
            return

        pos = self.player_pos(0)
        fwd, right, up = self.camera_basis(0, pos)

        look = self.look_dir(0)
        look = rotate_axis(look, up, -mx * MOUSE_SENS)
        look = rotate_axis(look, right, my * MOUSE_SENS)
        look = self.stabilize_view_direction(0, look)

        self.set_look_dir(0, look)

    def handle_gamepad_look(self, player_idx: int, look_x: float, look_y: float, dt: float) -> None:
        if not self.player_alive(player_idx):
            return
        if look_x == 0 and look_y == 0:
            return

        pos = self.player_pos(player_idx)
        _, right, up = self.camera_basis(player_idx, pos)
        look = self.look_dir(player_idx)
        look = rotate_axis(look, up, -look_x * GAMEPAD_LOOK_SPEED * dt)
        look = rotate_axis(look, right, look_y * GAMEPAD_LOOK_SPEED * dt)
        look = self.stabilize_view_direction(player_idx, look)
        self.set_look_dir(player_idx, look)

    def process_title_input(self) -> None:
        elapsed = pygame.time.get_ticks() * 0.001 - self.title_screen_started_at
        can_dismiss = elapsed >= TITLE_MIN_SECONDS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif can_dismiss and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.in_title_screen = False
            elif can_dismiss and event.type == pygame.MOUSEBUTTONDOWN:
                self.in_title_screen = False
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                self.refresh_gamepads()

        if self.in_title_screen and can_dismiss:
            for pad in self.gamepads:
                if pad is not None and pad.get_numbuttons() > 0 and pad.get_button(0):
                    self.in_title_screen = False
                    break

        if not self.in_title_screen:
            self.reset_players_for_match_start()
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            pygame.mouse.set_pos(self.screen_center)
            pygame.mouse.get_rel()
            pygame.event.clear(pygame.MOUSEMOTION)
            self.players[0].mouse_settle_frames = max(self.players[0].mouse_settle_frames, 2)

    def reset_players_for_match_start(self) -> None:
        for i, p in enumerate(self.players):
            spawn = self.initial_spawn_points[i]
            p.x, p.y, p.z = spawn
            p.vx = p.vy = p.vz = 0.0
            p.on_ground = False
            p.mining_target = None
            p.mining_progress = 0.0
            p.break_pulse = 0.0
            p.coyote_timer = 0.0
            p.jump_buffer_timer = 0.0
            p.jump_was_down = False
            p.health = PLAYER_MAX_HEALTH
            p.health_display = PLAYER_MAX_HEALTH
            p.health_hit_flash = 0.0
            p.death_fade_timer = 0.0
            p.death_hold_timer = 0.0
            self.reset_player_up(i)
            self.stabilize_player_spawn(i)
            self.reset_player_up(i)

    def render_title_screen(self) -> None:
        t = pygame.time.get_ticks() * 0.001
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 2.1))
        ember_wobble = math.sin(t * 0.8) * 22.0

        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Layered background gradient.
        bands = 44
        for i in range(bands):
            y0 = (self.height * i) / bands
            y1 = (self.height * (i + 1)) / bands
            k = i / max(1, bands - 1)
            r = 0.05 + 0.14 * k
            g = 0.03 + 0.06 * k
            b = 0.03 + 0.04 * k
            glColor3f(r, g, b)
            glBegin(GL_QUADS)
            glVertex2f(0, y0)
            glVertex2f(self.width, y0)
            glVertex2f(self.width, y1)
            glVertex2f(0, y1)
            glEnd()

        # Hellish horizon wedge.
        glColor4f(0.26, 0.10, 0.06, 0.92)
        glBegin(GL_TRIANGLES)
        glVertex2f(-40.0, self.height)
        glVertex2f(self.width * 0.56 + ember_wobble, self.height * 0.42)
        glVertex2f(self.width, self.height)
        glEnd()

        # Slanted smoke streaks.
        glColor4f(0.38, 0.17, 0.10, 0.20)
        for x in range(0, self.width, 18):
            glBegin(GL_LINES)
            glVertex2f(float(x), 0.0)
            glVertex2f(float(x - 210), float(self.height))
            glEnd()

        # Vignette to darken edges.
        edge = 120.0
        glColor4f(0.0, 0.0, 0.0, 0.35)
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(edge, 0.0)
        glVertex2f(edge, self.height)
        glVertex2f(0.0, self.height)
        glVertex2f(self.width - edge, 0.0)
        glVertex2f(self.width, 0.0)
        glVertex2f(self.width, self.height)
        glVertex2f(self.width - edge, self.height)
        glVertex2f(0.0, 0.0)
        glVertex2f(self.width, 0.0)
        glVertex2f(self.width, edge)
        glVertex2f(0.0, edge)
        glVertex2f(0.0, self.height - edge)
        glVertex2f(self.width, self.height - edge)
        glVertex2f(self.width, self.height)
        glVertex2f(0.0, self.height)
        glEnd()

        # Center panel and trim.
        panel_w = self.width * 0.70
        panel_h = self.height * 0.55
        panel_x = (self.width - panel_w) * 0.5
        panel_y = (self.height - panel_h) * 0.34
        glColor4f(0.03, 0.03, 0.03, 0.66)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_w, panel_y)
        glVertex2f(panel_x + panel_w, panel_y + panel_h)
        glVertex2f(panel_x, panel_y + panel_h)
        glEnd()
        glColor4f(0.62, 0.44, 0.22, 0.65)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_w, panel_y)
        glVertex2f(panel_x + panel_w, panel_y + panel_h)
        glVertex2f(panel_x, panel_y + panel_h)
        glEnd()

        cx = self.width * 0.5
        cy = self.height * 0.44
        tx = cx - 290.0
        self.draw_screen_text(tx, cy - 112.0, "THE WAR BELOW", (240, 216, 138, 255), font=self.ui_font_title)
        self.draw_screen_text(tx + 1.0, cy - 111.0, "THE WAR BELOW", (90, 25, 15, 120), font=self.ui_font_title)
        self.draw_screen_text(tx + 10.0, cy - 46.0, "This ain't Minecraft anymore!", (255, 228, 112, 255), font=self.ui_font_subtitle)
        self.draw_screen_text(tx + 10.0, cy - 6.0, "Two players. One cube-world. Dig or be buried.", (222, 194, 136, 255), small=True)

        deploy_col = int(185 + 70 * pulse)
        self.draw_screen_text(
            tx + 10.0,
            cy + 48.0,
            "Press Enter / Space / Click / Gamepad A to deploy",
            (255, deploy_col, 96, 255),
            small=True,
        )
        remaining = max(0.0, TITLE_MIN_SECONDS - (pygame.time.get_ticks() * 0.001 - self.title_screen_started_at))
        if remaining > 0.0:
            self.draw_screen_text(tx + 10.0, cy + 64.0, f"Stand by... {remaining:.1f}s", (255, 208, 120, 255), small=True)
        self.draw_screen_text(tx + 10.0, cy + 80.0, "Esc to quit", (190, 171, 130, 255), small=True)
        self.draw_screen_text(tx + 10.0, cy + 112.0, "C in-game opens full control reference", (176, 159, 124, 255), small=True)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    def stabilize_view_direction(self, player_idx: int, look: Vec3f) -> Vec3f:
        look_n = v_norm(look)
        pos = self.player_pos(player_idx)
        up = self.player_up(player_idx, pos)
        dot_up = v_dot(look_n, up)
        if abs(dot_up) <= LOOK_POLE_LIMIT:
            return look_n

        # Keep near-pole looking stable by preserving current heading (azimuth)
        # while clamping away from the exact singularity.
        cur = self.look_dir(player_idx)
        tangent = v_sub(cur, v_scale(up, v_dot(cur, up)))
        if v_len(tangent) < 1e-5:
            seed = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 0.0, 1.0)
            tangent = v_norm(v_cross(seed, up))
        else:
            tangent = v_norm(tangent)

        clamped_dot = math.copysign(LOOK_POLE_LIMIT, dot_up)
        tangent_mag = math.sqrt(max(0.0, 1.0 - clamped_dot * clamped_dot))
        return v_norm(v_add(v_scale(up, clamped_dot), v_scale(tangent, tangent_mag)))

    def place_block(self, player_idx: int) -> None:
        if not self.player_alive(player_idx):
            return
        _, prev = self.raycast_block(REACH, player_idx)
        if prev:
            px, py, pz = prev
            if self.world.block_at(px, py, pz) == AIR and not self.player_intersects(player_idx, px, py, pz):
                self.set_world_block(px, py, pz, self.selected_block)

    def process_input(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                self.refresh_gamepads()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.selected_block = DIRT
                elif event.key == pygame.K_2:
                    self.selected_block = STONE
                elif event.key == pygame.K_3:
                    self.selected_block = WOOD
                elif event.key == pygame.K_c:
                    self.show_controls_help = not self.show_controls_help
                elif self.show_controls_help and event.key == pygame.K_F2:
                    self.cycle_pad_slot(1, 1)
                elif self.show_controls_help and event.key == pygame.K_F3:
                    self.auto_assign_gamepads()
                    self.apply_pad_assignments()
                elif self.show_controls_help and event.key == pygame.K_F5:
                    self.refresh_gamepads()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    self.place_block(0)

        self.handle_mouse_look()
        keys = pygame.key.get_pressed()
        self.show_player_lines = bool(keys[pygame.K_PERIOD])

        p1_move_x = float((1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0))
        p1_move_z = float((1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0))
        p1_jump = bool(keys[pygame.K_SPACE])
        p1_mine_hold = bool(pygame.mouse.get_pressed(3)[0])
        p1_throw_hold = bool(keys[pygame.K_e])

        p2_move_x = 0.0
        p2_move_z = 0.0
        p2_jump = False
        p2_mine_hold = False
        p2_throw_hold = False

        # Player 2 must use gamepad.
        if self.p2_pad is not None:
            p2_move_x = self.read_axis(self.p2_pad, 0)
            p2_move_z = -self.read_axis(self.p2_pad, 1)
            p2_jump = self.read_button(self.p2_pad, 0)
            p2_mine_hold = self.read_button(self.p2_pad, 5)
            p2_throw_hold = self.read_button(self.p2_pad, 3)

            lx2 = self.read_axis(self.p2_pad, 2)
            ly2 = self.read_axis(self.p2_pad, 3)
            self.handle_gamepad_look(1, lx2, ly2, dt)

            if self.edge_button(1, 2, self.read_button(self.p2_pad, 2)):
                self.place_block(1)
            if self.edge_button(1, 1, self.read_button(self.p2_pad, 1)):
                self.selected_block = DIRT if self.selected_block == WOOD else self.selected_block + 1
            if self.edge_button(1, 4, self.read_button(self.p2_pad, 4)):
                self.selected_block = WOOD if self.selected_block == DIRT else self.selected_block - 1

        self.update_player_movement(
            player_idx=0,
            move_x=p1_move_x,
            move_z=p1_move_z,
            jump_pressed=p1_jump,
            dt=dt,
        )
        self.update_player_movement(
            player_idx=1,
            move_x=p2_move_x,
            move_z=p2_move_z,
            jump_pressed=p2_jump,
            dt=dt,
        )

        self.update_mining(0, dt, p1_mine_hold)
        self.update_mining(1, dt, p2_mine_hold)
        self.update_mine_throw(0, dt, p1_throw_hold)
        self.update_mine_throw(1, dt, p2_throw_hold)

    def update_player_movement(self, player_idx: int, move_x: float, move_z: float, jump_pressed: bool, dt: float) -> None:
        p = self.players[player_idx]
        if not self.player_alive(player_idx):
            self.set_player_vel(player_idx, (0.0, 0.0, 0.0))
            return
        pos = self.player_pos(player_idx)
        fwd, right, up = self.camera_basis(player_idx, pos)
        gravity = self.player_gravity_dir(player_idx, pos)
        no_input = False

        if math.sqrt(move_x * move_x + move_z * move_z) < MOVE_INPUT_DEADZONE:
            move_x = 0.0
            move_z = 0.0
            no_input = True

        support_now = self.collides_body(v_add(pos, v_scale(gravity, 0.14)))
        if support_now:
            p.coyote_timer = COYOTE_TIME
        else:
            p.coyote_timer = max(0.0, p.coyote_timer - dt)

        jump_pressed_edge = jump_pressed and not p.jump_was_down
        p.jump_was_down = jump_pressed
        if jump_pressed_edge:
            p.jump_buffer_timer = JUMP_BUFFER_TIME
        else:
            p.jump_buffer_timer = max(0.0, p.jump_buffer_timer - dt)

        if no_input and support_now and not jump_pressed:
            p.on_ground = True
            self.set_player_vel(player_idx, (0.0, 0.0, 0.0))
            return

        wish = (0.0, 0.0, 0.0)
        wish = v_add(wish, v_scale(fwd, float(move_z)))
        wish = v_add(wish, v_scale(right, float(move_x)))
        if v_len(wish) > 0.001:
            wish = v_scale(v_norm(wish), PLAYER_SPEED)
        else:
            wish = (0.0, 0.0, 0.0)

        vel = self.player_vel(player_idx)
        gspeed = v_dot(vel, gravity)
        vel = v_add(wish, v_scale(gravity, gspeed))
        if no_input:
            # Hard-remove any sideways creep when there is no intended movement.
            vel = v_scale(gravity, gspeed)

        can_jump = p.coyote_timer > 0.0
        if can_jump and p.jump_buffer_timer > 0.0:
            vel = v_add(vel, v_scale(up, JUMP_SPEED))
            p.on_ground = False
            p.coyote_timer = 0.0
            p.jump_buffer_timer = 0.0

        gmult = 1.0
        if gspeed > 0.0:
            gmult = FALL_GRAVITY_MULT
        elif gspeed < 0.0 and not jump_pressed:
            gmult = JUMP_CUT_GRAVITY_MULT
        vel = v_add(vel, v_scale(gravity, GRAVITY * gmult * dt))
        self.set_player_vel(player_idx, vel)

        delta = v_scale(vel, dt)
        self.move_and_collide(player_idx, delta)

        pos2 = self.player_pos(player_idx)
        g2 = self.player_gravity_dir(player_idx, pos2)
        support_near = self.collides_body(v_add(pos2, v_scale(g2, 0.14)))
        p.on_ground = support_near
        if no_input and support_near:
            self.set_player_vel(player_idx, (0.0, 0.0, 0.0))

    def mine_time_for(self, block: int) -> float:
        if block in (STONE, CHARRED_STONE):
            return MINE_TIME_STONE
        if block == WOOD:
            return MINE_TIME_WOOD
        return MINE_TIME_DIRT

    def spawn_break_feedback(self, cell: Vec3i, block: int, player_idx: Optional[int] = None) -> None:
        cx = cell[0] + 0.5
        cy = cell[1] + 0.5
        cz = cell[2] + 0.5
        base = BLOCK_COLORS.get(block, (0.45, 0.35, 0.25))
        eye_idx = 0 if player_idx is None else player_idx
        eye = self.eye_pos(eye_idx)
        outward = v_norm((cx - eye[0], cy - eye[1], cz - eye[2]))
        if v_len(outward) < 1e-5:
            outward = self.look_dir(eye_idx)
        is_dirt_break = block in (DIRT, GRASS)
        count = BREAK_PARTICLE_COUNT + (DIRT_PARTICLE_BONUS if is_dirt_break else 0)

        for _ in range(count):
            vx = (random.random() - 0.5) * 2.0
            vy = (random.random() - 0.5) * 2.0
            vz = (random.random() - 0.5) * 2.0
            v = v_norm((vx, vy, vz))
            v = v_norm(v_add(v, v_scale(outward, 0.75)))
            speed = BREAK_PARTICLE_SPEED * (0.85 + random.random() * (1.25 if is_dirt_break else 1.0))
            r_mul = 0.7 + random.random() * 0.45 if is_dirt_break else (0.8 + random.random() * 0.4)
            g_mul = 0.66 + random.random() * 0.38 if is_dirt_break else (0.8 + random.random() * 0.4)
            b_mul = 0.62 + random.random() * 0.32 if is_dirt_break else (0.8 + random.random() * 0.4)
            self.break_particles.append(
                {
                    "x": cx + outward[0] * 0.26 + (random.random() - 0.5) * 0.18,
                    "y": cy + outward[1] * 0.26 + (random.random() - 0.5) * 0.18,
                    "z": cz + outward[2] * 0.26 + (random.random() - 0.5) * 0.18,
                    "vx": v[0] * speed,
                    "vy": v[1] * speed,
                    "vz": v[2] * speed,
                    "life": BREAK_PARTICLE_LIFE,
                    "ttl": BREAK_PARTICLE_LIFE,
                    "r": base[0] * r_mul,
                    "g": base[1] * g_mul,
                    "b": base[2] * b_mul,
                    "d": 1.0 if is_dirt_break else 0.0,
                    "owner": -1 if player_idx is None else player_idx,
                }
            )
        if player_idx is None:
            for p in self.players:
                p.break_pulse = max(p.break_pulse, 0.55)
        else:
            self.players[player_idx].break_pulse = max(self.players[player_idx].break_pulse, 1.0)

    def update_break_feedback(self, dt: float) -> None:
        for p in self.players:
            if p.break_pulse > 0.0:
                p.break_pulse = max(0.0, p.break_pulse - dt * 4.8)

        if not self.break_particles:
            return

        for p in self.break_particles:
            p["life"] -= dt
            gdir = self.gravity_dir((p["x"], p["y"], p["z"]))
            gscale = 9.0 if p["d"] > 0.5 else 6.0
            p["vx"] += gdir[0] * gscale * dt
            p["vy"] += gdir[1] * gscale * dt
            p["vz"] += gdir[2] * gscale * dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["z"] += p["vz"] * dt
            drag = 0.90 if p["d"] > 0.5 else 0.94
            p["vx"] *= drag
            p["vy"] *= drag
            p["vz"] *= drag

        self.break_particles = [p for p in self.break_particles if p["life"] > 0.0]

    def spawn_explosion_effect(self, center: Vec3f, scale: float = 1.0, axis: Optional[Vec3i] = None, player_idx: Optional[int] = None) -> None:
        core_count = max(10, int(EXPLOSION_PARTICLE_COUNT * scale))
        spark_count = max(4, int(EXPLOSION_SPARK_COUNT * scale))
        axis_dir = None
        if axis is not None:
            axis_dir = v_norm((float(axis[0]), float(axis[1]), float(axis[2])))
            if v_len(axis_dir) < 1e-5:
                axis_dir = None
            else:
                # Directional arm cells should read as beams, not blobs.
                core_count = max(8, int(core_count * 0.55))
                spark_count = max(3, int(spark_count * 0.45))
        for _ in range(core_count):
            rnd = v_norm(
                (
                    (random.random() - 0.5) * 2.0,
                    (random.random() - 0.5) * 2.0,
                    (random.random() - 0.5) * 2.0,
                )
            )
            if axis_dir is not None:
                perp = v_sub(rnd, v_scale(axis_dir, v_dot(rnd, axis_dir)))
                if v_len(perp) > 1e-5:
                    perp = v_norm(perp)
                else:
                    perp = (0.0, 0.0, 0.0)
                v = v_norm(v_add(v_scale(axis_dir, 1.0 - EXPLOSION_AXIS_SPREAD), v_scale(perp, EXPLOSION_AXIS_SPREAD)))
            else:
                v = rnd
            speed = EXPLOSION_PARTICLE_SPEED * scale * (0.45 + random.random() * 0.9)
            hot = random.random()
            if hot > 0.78:
                col = (0.98, 0.55, 0.22)
            elif hot > 0.45:
                col = (0.82, 0.22, 0.07)
            else:
                col = (0.22, 0.08, 0.06)
            self.explosion_particles.append(
                {
                    "x": center[0] + (random.random() - 0.5) * 0.25,
                    "y": center[1] + (random.random() - 0.5) * 0.25,
                    "z": center[2] + (random.random() - 0.5) * 0.25,
                    "vx": v[0] * speed,
                    "vy": v[1] * speed,
                    "vz": v[2] * speed,
                    "life": EXPLOSION_PARTICLE_LIFE * (0.7 + 0.3 * scale),
                    "ttl": EXPLOSION_PARTICLE_LIFE * (0.7 + 0.3 * scale),
                    "r": col[0],
                    "g": col[1],
                    "b": col[2],
                    "owner": -1 if player_idx is None else player_idx,
                }
            )
        # Fast bright sparks.
        for _ in range(spark_count):
            rnd = v_norm(
                (
                    (random.random() - 0.5) * 2.0,
                    (random.random() - 0.5) * 2.0,
                    (random.random() - 0.5) * 2.0,
                )
            )
            if axis_dir is not None:
                perp = v_sub(rnd, v_scale(axis_dir, v_dot(rnd, axis_dir)))
                if v_len(perp) > 1e-5:
                    perp = v_norm(perp)
                else:
                    perp = (0.0, 0.0, 0.0)
                v = v_norm(v_add(v_scale(axis_dir, 0.985), v_scale(perp, 0.09)))
            else:
                v = rnd
            speed = EXPLOSION_PARTICLE_SPEED * scale * (1.4 + random.random() * 1.1)
            self.explosion_particles.append(
                {
                    "x": center[0],
                    "y": center[1],
                    "z": center[2],
                    "vx": v[0] * speed,
                    "vy": v[1] * speed,
                    "vz": v[2] * speed,
                    "life": 0.28 * (0.7 + 0.3 * scale),
                    "ttl": 0.28 * (0.7 + 0.3 * scale),
                    "r": 1.0,
                    "g": 0.36,
                    "b": 0.10,
                    "spark": 1.0,
                    "owner": -1 if player_idx is None else player_idx,
                }
            )
        if player_idx is None:
            for p in self.players:
                p.break_pulse = max(p.break_pulse, 1.4 + 0.8 * scale)
        elif 0 <= player_idx < len(self.players):
            self.players[player_idx].break_pulse = max(self.players[player_idx].break_pulse, 1.4 + 0.8 * scale)

    def update_explosion_effects(self, dt: float) -> None:
        if not self.explosion_particles:
            return

        for p in self.explosion_particles:
            p["life"] -= dt
            gdir = self.gravity_dir((p["x"], p["y"], p["z"]))
            spark = float(p.get("spark", 0.0))
            drag = 0.84 if spark > 0.5 else 0.92
            grav = 2.5 if spark > 0.5 else 5.0
            p["vx"] *= drag
            p["vy"] *= drag
            p["vz"] *= drag
            p["vx"] += gdir[0] * grav * dt
            p["vy"] += gdir[1] * grav * dt
            p["vz"] += gdir[2] * grav * dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["z"] += p["vz"] * dt

        self.explosion_particles = [p for p in self.explosion_particles if p["life"] > 0.0]

    def update_mining(self, player_idx: int, dt: float, is_mining: bool) -> None:
        p = self.players[player_idx]
        if not self.player_alive(player_idx):
            p.mining_target = None
            p.mining_progress = 0.0
            return
        if not is_mining:
            p.mining_target = None
            p.mining_progress = 0.0
            return

        hit, _ = self.raycast_block(REACH, player_idx)
        if hit is None:
            p.mining_target = None
            p.mining_progress = 0.0
            return

        if hit != p.mining_target:
            p.mining_target = hit
            p.mining_progress = 0.0

        block = self.world.block_at(*hit)
        if block == AIR:
            p.mining_target = None
            p.mining_progress = 0.0
            return

        p.mining_progress += dt
        if p.mining_progress >= self.mine_time_for(block):
            broken_block = block
            self.set_world_block(*hit, AIR)
            self.spawn_break_feedback(hit, broken_block, player_idx)
            p.mining_target = None
            p.mining_progress = 0.0

    def player_intersects(self, player_idx: int, bx: int, by: int, bz: int) -> bool:
        pos = self.player_pos(player_idx)
        up = self.outward_up(pos)
        lower = v_add(pos, v_scale(up, PLAYER_BODY_LOWER))
        upper = v_add(pos, v_scale(up, PLAYER_BODY_UPPER))
        return (
            self.sphere_aabb_intersect(lower, PLAYER_RADIUS * 1.05, bx, by, bz)
            or self.sphere_aabb_intersect(upper, PLAYER_RADIUS * PLAYER_HEAD_RADIUS_SCALE, bx, by, bz)
        )

    def player_in_blast_cell(self, player_idx: int, bx: int, by: int, bz: int) -> bool:
        # Slightly expanded blast check so close-proximity self-hits are reliable.
        pos = self.player_pos(player_idx)
        up = self.player_up(player_idx, pos)
        lower = v_add(pos, v_scale(up, PLAYER_BODY_LOWER))
        upper = v_add(pos, v_scale(up, PLAYER_BODY_UPPER))
        return (
            self.sphere_aabb_intersect(lower, PLAYER_RADIUS * 1.35, bx, by, bz)
            or self.sphere_aabb_intersect(upper, PLAYER_RADIUS * 1.22, bx, by, bz)
        )

    def sphere_aabb_intersect(self, pos: Vec3f, radius: float, bx: int, by: int, bz: int) -> bool:
        cx = clamp(pos[0], bx, bx + 1.0)
        cy = clamp(pos[1], by, by + 1.0)
        cz = clamp(pos[2], bz, bz + 1.0)
        dx = pos[0] - cx
        dy = pos[1] - cy
        dz = pos[2] - cz
        return dx * dx + dy * dy + dz * dz < radius * radius

    def collides_sphere(self, pos: Vec3f, radius: float) -> bool:
        min_x = math.floor(pos[0] - radius)
        max_x = math.floor(pos[0] + radius)
        min_y = math.floor(pos[1] - radius)
        max_y = math.floor(pos[1] + radius)
        min_z = math.floor(pos[2] - radius)
        max_z = math.floor(pos[2] + radius)

        for bx in range(min_x, max_x + 1):
            for by in range(min_y, max_y + 1):
                for bz in range(min_z, max_z + 1):
                    if self.world.block_at(bx, by, bz) == AIR:
                        continue
                    if self.sphere_aabb_intersect(pos, radius, bx, by, bz):
                        return True
        return False

    def collides_body(self, pos: Vec3f) -> bool:
        up = self.outward_up(pos)
        lower = v_add(pos, v_scale(up, PLAYER_BODY_LOWER))
        upper = v_add(pos, v_scale(up, PLAYER_BODY_UPPER))
        return (
            self.collides_sphere(lower, PLAYER_RADIUS)
            or self.collides_sphere(upper, PLAYER_RADIUS * PLAYER_HEAD_RADIUS_SCALE)
        )

    def move_and_collide(self, player_idx: int, delta: Vec3f) -> None:
        pos = self.player_pos(player_idx)
        vel = list(self.player_vel(player_idx))

        for axis in range(3):
            step = [0.0, 0.0, 0.0]
            step[axis] = delta[axis]
            trial = (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2])
            if not self.collides_body(trial):
                pos = trial
            else:
                vel[axis] = 0.0

        self.set_player_pos(player_idx, pos)
        self.set_player_vel(player_idx, (vel[0], vel[1], vel[2]))

    def raycast_block(self, max_dist: float, player_idx: int) -> Tuple[Optional[Vec3i], Optional[Vec3i]]:
        ox, oy, oz = self.eye_pos(player_idx)
        dx, dy, dz = self.look_dir(player_idx)
        last_empty = None
        t = 0.0
        step = 0.05
        while t <= max_dist:
            x = ox + dx * t
            y = oy + dy * t
            z = oz + dz * t
            cell = (math.floor(x), math.floor(y), math.floor(z))
            if self.world.block_at(*cell) != AIR:
                return cell, last_empty
            last_empty = cell
            t += step
        return None, last_empty

    def random_face_support(self, axis: int, sign: int) -> Optional[Vec3i]:
        center = [self.world.cx, self.world.cy, self.world.cz]
        axis_lo = center[axis] - (CUBE_HALF + FACE_RELIEF + 2)
        axis_hi = center[axis] + (CUBE_HALF + FACE_RELIEF + 2)
        outer = clamp(center[axis] + sign * (CUBE_HALF + FACE_RELIEF + 1), axis_lo, axis_hi)
        outer_i = int(round(outer))
        other_axes = [0, 1, 2]
        other_axes.remove(axis)
        ranges = [
            (max(1, center[other_axes[0]] - CUBE_HALF), min((WORLD_X if other_axes[0] == 0 else WORLD_Y if other_axes[0] == 1 else WORLD_Z) - 2, center[other_axes[0]] + CUBE_HALF)),
            (max(1, center[other_axes[1]] - CUBE_HALF), min((WORLD_X if other_axes[1] == 0 else WORLD_Y if other_axes[1] == 1 else WORLD_Z) - 2, center[other_axes[1]] + CUBE_HALF)),
        ]
        max_steps = CUBE_HALF * 2 + FACE_RELIEF * 2 + 6

        for _ in range(120):
            a = random.randint(ranges[0][0], ranges[0][1])
            b = random.randint(ranges[1][0], ranges[1][1])
            coords = [0, 0, 0]
            coords[other_axes[0]] = a
            coords[other_axes[1]] = b
            for step in range(max_steps):
                coords[axis] = outer_i - sign * step
                x, y, z = coords[0], coords[1], coords[2]
                if not self.world.in_bounds(x, y, z):
                    continue
                if self.world.block_at(x, y, z) != AIR:
                    return (x, y, z)
        return None

    def find_spawn_point(self, axis: int, sign: int) -> Vec3f:
        support = self.random_face_support(axis, sign)
        n = (sign if axis == 0 else 0, sign if axis == 1 else 0, sign if axis == 2 else 0)
        if support is not None:
            return (
                support[0] + 0.5 + n[0] * 1.05,
                support[1] + 0.5 + n[1] * 1.05,
                support[2] + 0.5 + n[2] * 1.05,
            )

        # Deterministic fallback near the requested face center.
        return (
            self.world.cx + n[0] * (CUBE_HALF + FACE_RELIEF + 3.0),
            self.world.cy + n[1] * (CUBE_HALF + FACE_RELIEF + 3.0),
            self.world.cz + n[2] * (CUBE_HALF + FACE_RELIEF + 3.0),
        )

    def stabilize_player_spawn(self, player_idx: int) -> None:
        # Resolve any tiny initial overlap with terrain so movement works immediately.
        for _ in range(40):
            pos = self.player_pos(player_idx)
            if not self.collides_body(pos):
                break
            up = self.outward_up(pos)
            self.set_player_pos(player_idx, v_add(pos, v_scale(up, 0.08)))

        pos = self.player_pos(player_idx)
        self.players[player_idx].on_ground = self.collides_body(v_add(pos, v_scale(self.gravity_dir(pos), 0.10)))

    def render_world(self, player_idx: int) -> None:
        px, _, pz = self.player_pos(player_idx)
        r2 = RENDER_DISTANCE * RENDER_DISTANCE

        stride = 44
        glBindTexture(GL_TEXTURE_2D, self.texture_atlas)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        for key, mesh in self.chunk_meshes.items():
            if mesh.vertex_count == 0:
                continue
            cx, cz = key
            center_x = cx * CHUNK_SIZE + CHUNK_SIZE * 0.5
            center_z = cz * CHUNK_SIZE + CHUNK_SIZE * 0.5
            dx = center_x - px
            dz = center_z - pz
            if dx * dx + dz * dz > r2:
                continue

            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(32))
            glDrawArrays(GL_TRIANGLES, 0, mesh.vertex_count)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        self.render_mines(player_idx)
        self.render_thrown_mines(player_idx)
        self.render_break_particles(player_idx)
        self.render_explosion_particles(player_idx)

    def render_thrown_mines(self, player_idx: int) -> None:
        if not self.thrown_mines:
            return
        cam = self.player_pos(player_idx)
        max_r2 = MINE_RENDER_DISTANCE * MINE_RENDER_DISTANCE
        was_blend = glIsEnabled(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

        for mine in self.thrown_mines:
            center = (mine["x"], mine["y"], mine["z"])
            d2 = (center[0] - cam[0]) ** 2 + (center[1] - cam[1]) ** 2 + (center[2] - cam[2]) ** 2
            if d2 > max_r2:
                continue
            n = v_norm(v_scale(self.gravity_dir(center), -1.0))
            ref = (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0)
            u = v_norm(v_cross(ref, n))
            v = v_norm(v_cross(n, u))
            r = 0.18
            glColor3f(0.02, 0.02, 0.02)
            for i in range(MINE_SPHERE_STACKS):
                t0 = math.pi * i / MINE_SPHERE_STACKS
                t1 = math.pi * (i + 1) / MINE_SPHERE_STACKS
                glBegin(GL_TRIANGLE_STRIP)
                for j in range(MINE_SPHERE_SLICES + 1):
                    p = 2.0 * math.pi * j / MINE_SPHERE_SLICES
                    for t in (t0, t1):
                        st = math.sin(t)
                        ct = math.cos(t)
                        cp = math.cos(p)
                        sp = math.sin(p)
                        normal_vec = v_norm(v_add(v_scale(n, ct), v_add(v_scale(u, st * cp), v_scale(v, st * sp))))
                        point = v_add(center, v_scale(normal_vec, r))
                        glNormal3f(normal_vec[0], normal_vec[1], normal_vec[2])
                        glVertex3f(point[0], point[1], point[2])
                glEnd()

            pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.015))
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glColor4f(1.0 * pulse, 0.42 * pulse, 0.08 * pulse, 0.65)
            glow_center = v_add(center, v_scale(n, r + 0.005))
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(glow_center[0], glow_center[1], glow_center[2])
            for i in range(15):
                a = (math.pi * 2.0 * i) / 14.0
                c = math.cos(a)
                s = math.sin(a)
                edge = v_add(glow_center, v_add(v_scale(u, c * 0.12), v_scale(v, s * 0.12)))
                glVertex3f(edge[0], edge[1], edge[2])
            glEnd()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if not was_blend:
                glDisable(GL_BLEND)
            glEnable(GL_LIGHTING)

        glEnable(GL_CULL_FACE)
        if was_blend:
            glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)

    def render_mines(self, player_idx: int) -> None:
        if not self.mines:
            return

        cam = self.player_pos(player_idx)
        max_r2 = MINE_RENDER_DISTANCE * MINE_RENDER_DISTANCE
        visible: List[Tuple[Dict[str, object], Vec3f, Vec3f, Vec3f, Vec3f, float]] = []

        for mine in self.mines.values():
            pos = mine.get("pos")
            normal = mine.get("normal")
            support = mine.get("support")
            if not isinstance(pos, tuple) or not isinstance(normal, tuple) or not isinstance(support, tuple):
                continue

            n = v_norm((float(normal[0]), float(normal[1]), float(normal[2])))
            if v_len(n) < 1e-6:
                continue

            support_center = (support[0] + 0.5, support[1] + 0.5, support[2] + 0.5)
            center = v_add(support_center, v_scale(n, 0.5 + 0.22 * 0.72))
            d2 = (center[0] - cam[0]) ** 2 + (center[1] - cam[1]) ** 2 + (center[2] - cam[2]) ** 2
            if d2 > max_r2:
                continue

            ref = (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0)
            u = v_norm(v_cross(ref, n))
            v = v_norm(v_cross(n, u))
            visible.append((mine, center, n, u, v, d2))

        if not visible:
            return

        was_blend = glIsEnabled(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        t_now = pygame.time.get_ticks() * 0.001

        # Opaque mine bodies.
        for mine, center, n, u, v, _ in visible:
            kind = str(mine.get("kind", "timed"))
            timer = float(mine.get("timer", 0.0)) if kind == "timed" else 0.0

            r = 0.22

            # Closed low-poly sphere body (round mine, fully opaque).
            glColor3f(0.015, 0.012, 0.01)
            for i in range(MINE_SPHERE_STACKS):
                t0 = math.pi * i / MINE_SPHERE_STACKS
                t1 = math.pi * (i + 1) / MINE_SPHERE_STACKS
                glBegin(GL_TRIANGLE_STRIP)
                for j in range(MINE_SPHERE_SLICES + 1):
                    p = 2.0 * math.pi * j / MINE_SPHERE_SLICES
                    for t in (t0, t1):
                        st = math.sin(t)
                        ct = math.cos(t)
                        cp = math.cos(p)
                        sp = math.sin(p)
                        normal_vec = v_norm(v_add(v_scale(n, ct), v_add(v_scale(u, st * cp), v_scale(v, st * sp))))
                        point = v_add(center, v_scale(normal_vec, r))
                        glNormal3f(normal_vec[0], normal_vec[1], normal_vec[2])
                        glVertex3f(point[0], point[1], point[2])
                glEnd()

            # Flashing indicator on top.
            if kind == "proximity":
                elapsed = pygame.time.get_ticks() * 0.001
                flash = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(elapsed * 8.0))
                # 2x larger dome-like orange glow for strong visibility.
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                dome_center = v_add(center, v_scale(n, r * 0.10))
                dome_r = 4.20
                dome_stacks = 6
                dome_slices = 24
                for si in range(dome_stacks):
                    t0 = (math.pi * 0.5) * si / dome_stacks
                    t1 = (math.pi * 0.5) * (si + 1) / dome_stacks
                    fade = 1.0 - (si / max(1.0, float(dome_stacks)))
                    glColor4f(1.0, 0.50 + 0.20 * fade, 0.10 + 0.12 * fade, (0.08 + 0.22 * flash) * fade)
                    glBegin(GL_TRIANGLE_STRIP)
                    for j in range(dome_slices + 1):
                        p_ang = (math.pi * 2.0 * j) / dome_slices
                        cp = math.cos(p_ang)
                        sp = math.sin(p_ang)
                        for t_ang in (t0, t1):
                            st = math.sin(t_ang)
                            ct = math.cos(t_ang)
                            radial = dome_r * st
                            h = dome_r * ct
                            point = v_add(
                                dome_center,
                                v_add(
                                    v_scale(n, h),
                                    v_add(v_scale(u, cp * radial), v_scale(v, sp * radial)),
                                ),
                            )
                            glVertex3f(point[0], point[1], point[2])
                    glEnd()
                glDisable(GL_LIGHTING)
                glColor4f(1.0 * flash, 0.45 * flash, 0.08 * flash, 1.0)
                led_r = 0.24
            else:
                elapsed = MINE_FUSE_SECONDS - timer
                flash = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(elapsed * 11.0))
                glDisable(GL_LIGHTING)
                glColor4f(0.82 * flash, 0.07 * flash, 0.03 * flash, 1.0)
                led_r = 0.08
            led_center = v_add(center, v_scale(n, r + 0.004))
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(led_center[0], led_center[1], led_center[2])
            for i in range(13):
                a = (math.pi * 2.0 * i) / 12.0
                c = math.cos(a)
                s = math.sin(a)
                edge = v_add(led_center, v_add(v_scale(u, c * led_r), v_scale(v, s * led_r)))
                glVertex3f(edge[0], edge[1], edge[2])
            glEnd()
            if kind == "proximity":
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                if not was_blend:
                    glDisable(GL_BLEND)
            glEnable(GL_LIGHTING)

        glEnable(GL_CULL_FACE)
        if was_blend:
            glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)

    def render_break_particles(self, player_idx: int) -> None:
        if not self.break_particles:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for p in self.break_particles:
            owner = int(p.get("owner", -1))
            if owner >= 0 and owner != player_idx:
                continue
            if p["d"] > 0.5:
                continue
            alpha = clamp(p["life"] / p["ttl"], 0.0, 1.0)
            glColor4f(p["r"], p["g"], p["b"], alpha)
            glVertex3f(p["x"], p["y"], p["z"])
        glEnd()

        glPointSize(7.0)
        glBegin(GL_POINTS)
        for p in self.break_particles:
            owner = int(p.get("owner", -1))
            if owner >= 0 and owner != player_idx:
                continue
            if p["d"] <= 0.5:
                continue
            alpha = clamp(p["life"] / p["ttl"], 0.0, 1.0)
            glColor4f(p["r"], p["g"], p["b"], alpha * 0.9)
            glVertex3f(p["x"], p["y"], p["z"])
        glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

    def render_explosion_particles(self, player_idx: int) -> None:
        if not self.explosion_particles:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glPointSize(16.0)
        glBegin(GL_POINTS)
        for p in self.explosion_particles:
            owner = int(p.get("owner", -1))
            if owner >= 0 and owner != player_idx:
                continue
            alpha = clamp(p["life"] / p["ttl"], 0.0, 1.0)
            spark = float(p.get("spark", 0.0))
            glow = 1.1 if spark > 0.5 else 0.95
            ember = 1.0 - alpha
            rr = min(1.0, p["r"] * glow + ember * 0.06)
            gg = max(0.0, p["g"] * glow - ember * 0.30)
            bb = max(0.0, p["b"] * glow - ember * 0.34)
            glColor4f(rr, gg, bb, alpha)
            glVertex3f(p["x"], p["y"], p["z"])
        glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

    def update_lantern_light(self, player_idx: int) -> None:
        ex, ey, ez = self.eye_pos(player_idx)
        glLightfv(GL_LIGHT0, GL_POSITION, (ex, ey, ez, 1.0))

    def render_crosshair(self, player_idx: int, view_w: int, view_h: int) -> None:
        p = self.players[player_idx]
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, view_w, view_h, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)
        cx = view_w // 2
        cy = view_h // 2
        size = 8.0 + p.break_pulse * 5.0
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()

        # Mining progress line under reticle.
        if p.mining_target is not None:
            target_block = self.world.block_at(*p.mining_target)
            if target_block != AIR:
                req = self.mine_time_for(target_block)
                if req > 0.0:
                    progress = clamp(p.mining_progress / req, 0.0, 1.0)
                    bar_w = 44.0
                    y = cy + 18.0
                    glColor3f(0.25, 0.23, 0.2)
                    glBegin(GL_LINES)
                    glVertex2f(cx - bar_w * 0.5, y)
                    glVertex2f(cx + bar_w * 0.5, y)
                    glEnd()
                    glColor3f(1.0, 0.84, 0.58)
                    glBegin(GL_LINES)
                    glVertex2f(cx - bar_w * 0.5, y)
                    glVertex2f(cx - bar_w * 0.5 + bar_w * progress, y)
                    glEnd()

        self.render_horizon_marker(player_idx, view_w, view_h)
        self.render_health_bar(player_idx, view_w, view_h)
        self.render_other_player_dot(player_idx, view_w, view_h)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def render_horizon_marker(self, player_idx: int, view_w: int, view_h: int) -> None:
        pos = self.player_pos(player_idx)
        fwd = self.look_dir(player_idx)
        up = self.player_up(player_idx, pos)
        right = v_norm(v_cross(fwd, up))
        if v_len(right) < 1e-5:
            right = (1.0, 0.0, 0.0)

        # Project global up into camera plane so marker changes across cube faces.
        global_up = (0.0, 1.0, 0.0)
        h = v_sub(global_up, v_scale(fwd, v_dot(global_up, fwd)))
        if v_len(h) < 1e-5:
            h = right
        h = v_norm(h)
        hx = v_dot(h, right)
        hy = v_dot(h, up)
        h2 = math.sqrt(hx * hx + hy * hy)
        if h2 < 1e-5:
            hx, hy = 1.0, 0.0
        else:
            hx /= h2
            hy /= h2

        cx = view_w * 0.5
        cy = view_h * 0.5
        line_len = 24.0
        tick = 6.0

        glLineWidth(3.0)
        glColor3f(1.0, 0.95, 0.18)
        glBegin(GL_LINES)
        glVertex2f(cx - hx * line_len, cy + hy * line_len)
        glVertex2f(cx + hx * line_len, cy - hy * line_len)
        # center tick normal to the horizon line
        glVertex2f(cx - hy * tick, cy - hx * tick)
        glVertex2f(cx + hy * tick, cy + hx * tick)
        glEnd()
        glLineWidth(1.0)

    def render_health_bar(self, player_idx: int, view_w: int, view_h: int) -> None:
        p = self.players[player_idx]
        pct_raw = p.health / max(1e-5, PLAYER_MAX_HEALTH)
        pct = clamp(pct_raw if math.isfinite(pct_raw) else 1.0, 0.0, 1.0)
        shown_raw = p.health_display / max(1e-5, PLAYER_MAX_HEALTH)
        shown_pct = clamp(shown_raw if math.isfinite(shown_raw) else 1.0, 0.0, 1.0)
        w = min(260.0, view_w * 0.50)
        h = 22.0
        margin = 14.0
        x = view_w - w - margin
        y = view_h - h - margin
        was_blend = glIsEnabled(GL_BLEND)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Outer panel.
        glColor4f(0.0, 0.0, 0.0, 0.72)
        glBegin(GL_QUADS)
        glVertex2f(x - 5.0, y - 5.0)
        glVertex2f(x + w + 5.0, y - 5.0)
        glVertex2f(x + w + 5.0, y + h + 5.0)
        glVertex2f(x - 5.0, y + h + 5.0)
        glEnd()

        # Empty bar background.
        glColor3f(0.11, 0.07, 0.03)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        # Main fill (solid yellow when full).
        fill_w = w * pct
        if fill_w > 0.0:
            glColor3f(1.0, 0.96, 0.10)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + fill_w, y)
            glVertex2f(x + fill_w, y + h)
            glVertex2f(x, y + h)
            glEnd()

        # Trailing damage segment.
        if shown_pct > pct + 1e-4:
            seg_x0 = x + w * pct
            seg_x1 = x + w * shown_pct
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.02))
            alpha = clamp(0.34 + p.health_hit_flash * pulse, 0.34, 0.95)
            glColor4f(1.0, 0.24, 0.10, alpha)
            glBegin(GL_QUADS)
            glVertex2f(seg_x0, y + 2.0)
            glVertex2f(seg_x1, y + 2.0)
            glVertex2f(seg_x1, y + h - 2.0)
            glVertex2f(seg_x0, y + h - 2.0)
            glEnd()

        # Current health marker.
        edge_x = x + w * pct
        glLineWidth(2.5)
        glColor3f(1.0, 1.0, 0.92)
        glBegin(GL_LINES)
        glVertex2f(edge_x, y - 3.0)
        glVertex2f(edge_x, y + h + 3.0)
        glEnd()
        glLineWidth(1.0)

        # Quarter ticks to make drop distance obvious.
        glColor4f(0.36, 0.29, 0.10, 0.75)
        for i in range(1, 4):
            tx = x + w * (i * 0.25)
            glBegin(GL_LINES)
            glVertex2f(tx, y + 2.0)
            glVertex2f(tx, y + h - 2.0)
            glEnd()

        if p.health_hit_flash > 0.0:
            pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.03))
            flash_a = clamp(p.health_hit_flash * pulse, 0.0, 0.9)
            glColor4f(1.0, 0.14, 0.06, flash_a * 0.42)
            glBegin(GL_QUADS)
            glVertex2f(x - 10.0, y - 8.0)
            glVertex2f(x + w + 10.0, y - 8.0)
            glVertex2f(x + w + 10.0, y + h + 8.0)
            glVertex2f(x - 10.0, y + h + 8.0)
            glEnd()

        # Bright border to ensure visibility in dark scenes.
        glLineWidth(2.5)
        border_r = 1.0
        border_g = 0.95 - 0.45 * p.health_hit_flash
        border_b = 0.34 - 0.22 * p.health_hit_flash
        glColor3f(border_r, max(0.18, border_g), max(0.10, border_b))
        glBegin(GL_LINE_LOOP)
        glVertex2f(x - 1.5, y - 1.5)
        glVertex2f(x + w + 1.5, y - 1.5)
        glVertex2f(x + w + 1.5, y + h + 1.5)
        glVertex2f(x - 1.5, y + h + 1.5)
        glEnd()
        glLineWidth(1.0)

        if not was_blend:
            glDisable(GL_BLEND)

    def render_death_overlay(self, player_idx: int, view_w: int, view_h: int) -> None:
        alpha = self.death_overlay_alpha(player_idx)
        if alpha <= 0.0:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, view_w, view_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        was_blend = glIsEnabled(GL_BLEND)
        if alpha >= 0.999:
            glDisable(GL_BLEND)
        else:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glColor4f(0.0, 0.0, 0.0, alpha)
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(float(view_w), 0.0)
        glVertex2f(float(view_w), float(view_h))
        glVertex2f(0.0, float(view_h))
        glEnd()
        if was_blend:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def death_overlay_alpha(self, player_idx: int) -> float:
        p = self.players[player_idx]
        if self.player_alive(player_idx):
            return 0.0
        if p.death_fade_timer > 0.0:
            fade_t = 1.0 - clamp(p.death_fade_timer / max(1e-5, DEATH_FADE_SECONDS), 0.0, 1.0)
            # Strongly visible fade from frame one, then up to full black.
            return clamp(0.35 + 0.65 * fade_t, 0.0, 1.0)
        return 1.0

    def apply_camera(self, player_idx: int) -> None:
        glLoadIdentity()
        eye = self.eye_pos(player_idx)
        look = self.look_dir(player_idx)
        up = self.player_up(player_idx)
        target = v_add(eye, look)
        gluLookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2], up[0], up[1], up[2])

    def render_other_player_dot(self, player_idx: int, view_w: int, view_h: int) -> None:
        other_idx = 1 - player_idx
        eye = self.eye_pos(player_idx)
        target = self.player_pos(other_idx)
        rel = v_sub(target, eye)

        fwd = self.look_dir(player_idx)
        up = self.player_up(player_idx)
        right = v_norm(v_cross(fwd, up))
        if v_len(right) < 1e-5:
            right = (1.0, 0.0, 0.0)
        up = v_norm(v_cross(right, fwd))
        z = v_dot(rel, fwd)
        x = v_dot(rel, right)
        y = v_dot(rel, up)

        tan_half = math.tan(math.radians(FOV_DEG * 0.5))
        aspect = view_w / max(1.0, float(view_h))

        behind = z <= 0.01
        if behind:
            z = 0.01
            x = -x
            y = -y

        x_ndc = x / (z * tan_half * aspect)
        y_ndc = y / (z * tan_half)
        offscreen = behind or abs(x_ndc) > 1.0 or abs(y_ndc) > 1.0

        x_ndc = clamp(x_ndc, -0.92, 0.92)
        y_ndc = clamp(y_ndc, -0.92, 0.92)
        sx = (x_ndc * 0.5 + 0.5) * view_w
        sy = (-y_ndc * 0.5 + 0.5) * view_h

        t = pygame.time.get_ticks() * 0.001
        pulse = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(t * 9.0))
        cx = view_w * 0.5
        cy = view_h * 0.5

        if offscreen:
            fill = (1.0, 0.12, 0.08)
            ring = (1.0, 0.78, 0.25)
            radius = 18.0
        else:
            fill = (0.95, 1.0, 0.12)
            ring = (0.15, 1.0, 0.35)
            radius = 15.0

        # Dotted, semi-transparent direction line (visible only while '.' is held).
        if self.show_player_lines:
            glLineWidth(3.5)
            glColor4f(ring[0] * 0.9, ring[1] * 0.9, ring[2] * 0.5, 0.5)
            dx = sx - cx
            dy = sy - cy
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len > 1e-4:
                ux = dx / seg_len
                uy = dy / seg_len
                dash = 14.0
                gap = 10.0
                tpos = 0.0
                glBegin(GL_LINES)
                while tpos < seg_len:
                    a = tpos
                    b = min(seg_len, tpos + dash)
                    glVertex2f(cx + ux * a, cy + uy * a)
                    glVertex2f(cx + ux * b, cy + uy * b)
                    tpos += dash + gap
                glEnd()

        # Pulsing outer ring.
        glColor3f(ring[0] * pulse, ring[1] * pulse, ring[2] * pulse)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(sx, sy)
        for i in range(25):
            a = (math.pi * 2.0 * i) / 24.0
            glVertex2f(sx + math.cos(a) * (radius + 10.0 * pulse), sy + math.sin(a) * (radius + 10.0 * pulse))
        glEnd()

        # Bold black stroke.
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(sx, sy)
        for i in range(25):
            a = (math.pi * 2.0 * i) / 24.0
            glVertex2f(sx + math.cos(a) * (radius + 4.0), sy + math.sin(a) * (radius + 4.0))
        glEnd()

        # Main fill.
        glColor3f(fill[0], fill[1], fill[2])
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(sx, sy)
        for i in range(25):
            a = (math.pi * 2.0 * i) / 24.0
            glVertex2f(sx + math.cos(a) * radius, sy + math.sin(a) * radius)
        glEnd()

        # Bright center.
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(sx, sy)
        for i in range(25):
            a = (math.pi * 2.0 * i) / 24.0
            glVertex2f(sx + math.cos(a) * (radius * 0.32), sy + math.sin(a) * (radius * 0.32))
        glEnd()

        self.render_player_radar(player_idx, view_w, view_h)

    def render_player_radar(self, player_idx: int, view_w: int, view_h: int) -> None:
        other_idx = 1 - player_idx
        center_x = 70.0
        center_y = 70.0
        radar_r = 46.0
        max_range = 42.0

        pos = self.player_pos(player_idx)
        other = self.player_pos(other_idx)
        rel = v_sub(other, pos)
        fwd, right, _ = self.camera_basis(player_idx, pos)
        rx = v_dot(rel, right)
        rz = v_dot(rel, fwd)

        # Radar frame.
        glColor4f(0.0, 0.0, 0.0, 0.45)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(center_x, center_y)
        for i in range(33):
            a = (math.pi * 2.0 * i) / 32.0
            glVertex2f(center_x + math.cos(a) * radar_r, center_y + math.sin(a) * radar_r)
        glEnd()

        glColor3f(0.92, 0.96, 0.22)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(33):
            a = (math.pi * 2.0 * i) / 32.0
            glVertex2f(center_x + math.cos(a) * radar_r, center_y + math.sin(a) * radar_r)
        glEnd()

        glColor4f(0.9, 0.9, 0.2, 0.35)
        glBegin(GL_LINES)
        glVertex2f(center_x - radar_r, center_y)
        glVertex2f(center_x + radar_r, center_y)
        glVertex2f(center_x, center_y - radar_r)
        glVertex2f(center_x, center_y + radar_r)
        glEnd()

        # Other-player blip.
        scale = radar_r / max_range
        bx = clamp(rx * scale, -radar_r + 4.0, radar_r - 4.0)
        by = clamp(-rz * scale, -radar_r + 4.0, radar_r - 4.0)
        dist = math.sqrt(rx * rx + rz * rz)
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.012))
        if dist > max_range:
            color = (1.0, 0.28, 0.16)
        else:
            color = (0.25, 1.0, 0.28)
        glColor3f(color[0] * pulse, color[1] * pulse, color[2] * pulse)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(center_x + bx, center_y + by)
        for i in range(17):
            a = (math.pi * 2.0 * i) / 16.0
            glVertex2f(center_x + bx + math.cos(a) * 5.0, center_y + by + math.sin(a) * 5.0)
        glEnd()

        # Proximity-mine blips (light blue).
        mine_pulse = 0.68 + 0.32 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.010))
        for mine in self.mines.values():
            if str(mine.get("kind", "timed")) != "proximity":
                continue
            mpos = mine.get("pos")
            if not isinstance(mpos, tuple):
                continue
            mcenter = (mpos[0] + 0.5, mpos[1] + 0.5, mpos[2] + 0.5)
            mrel = v_sub(mcenter, pos)
            mx = v_dot(mrel, right)
            mz = v_dot(mrel, fwd)
            mbx = clamp(mx * scale, -radar_r + 3.0, radar_r - 3.0)
            mby = clamp(-mz * scale, -radar_r + 3.0, radar_r - 3.0)
            mdist = math.sqrt(mx * mx + mz * mz)
            mine_col = (0.48, 0.95, 1.0) if mdist <= max_range else (0.30, 0.75, 0.90)
            glColor3f(mine_col[0] * mine_pulse, mine_col[1] * mine_pulse, mine_col[2] * mine_pulse)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(center_x + mbx, center_y + mby)
            for i in range(13):
                a = (math.pi * 2.0 * i) / 12.0
                glVertex2f(center_x + mbx + math.cos(a) * 3.2, center_y + mby + math.sin(a) * 3.2)
            glEnd()

        # Show the other player's facing direction on radar.
        other_fwd = self.look_dir(other_idx)
        hx = v_dot(other_fwd, right)
        hz = v_dot(other_fwd, fwd)
        hlen = math.sqrt(hx * hx + hz * hz)
        if hlen > 1e-5:
            hx /= hlen
            hz /= hlen
            tip_x = center_x + bx + hx * 10.0
            tip_y = center_y + by - hz * 10.0
            left_x = center_x + bx - hz * 3.5
            left_y = center_y + by - hx * 3.5
            right_x = center_x + bx + hz * 3.5
            right_y = center_y + by + hx * 3.5
            # Dark outline under bright arrow for contrast.
            glColor3f(0.05, 0.05, 0.05)
            glBegin(GL_TRIANGLES)
            glVertex2f(tip_x + hx * 1.4, tip_y - hz * 1.4)
            glVertex2f(left_x - hz * 1.3, left_y - hx * 1.3)
            glVertex2f(right_x + hz * 1.3, right_y + hx * 1.3)
            glEnd()
            glColor3f(1.0, 0.98, 0.16)
            glBegin(GL_TRIANGLES)
            glVertex2f(tip_x, tip_y)
            glVertex2f(left_x, left_y)
            glVertex2f(right_x, right_y)
            glEnd()

        self.render_face_indicator(player_idx, center_x + 92.0, center_y)

    def render_face_indicator(self, player_idx: int, cx: float, cy: float) -> None:
        face = self.face_for_position(self.player_pos(player_idx))
        tile = 15.0
        t = pygame.time.get_ticks() * 0.001
        pulse = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(t * 7.0))

        # Compact unfolded-cube map around +Z (front) face.
        layout: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
            ((0, 1), (-1, 0)),   # -X
            ((0, -1), (1, 0)),   # +X
            ((1, 1), (0, -1)),   # +Y
            ((1, -1), (0, 1)),   # -Y
            ((2, 1), (0, 0)),    # +Z
            ((2, -1), (2, 0)),   # -Z
        ]

        panel_w = tile * 5.2
        panel_h = tile * 3.9
        glColor4f(0.12, 0.10, 0.02, 0.35)
        glBegin(GL_QUADS)
        glVertex2f(cx - panel_w * 0.5, cy - panel_h * 0.5)
        glVertex2f(cx + panel_w * 0.5, cy - panel_h * 0.5)
        glVertex2f(cx + panel_w * 0.5, cy + panel_h * 0.5)
        glVertex2f(cx - panel_w * 0.5, cy + panel_h * 0.5)
        glEnd()

        for f, (gx, gy) in layout:
            x0 = cx + gx * tile - tile * 0.46
            y0 = cy + gy * tile - tile * 0.46
            x1 = x0 + tile * 0.92
            y1 = y0 + tile * 0.92

            # Base cube representation is yellow for all faces.
            glColor3f(0.98, 0.88, 0.22)

            glBegin(GL_QUADS)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

            if f == face:
                # Active face: darker yellow shade overlay.
                inset = tile * 0.10
                glColor3f(0.72, 0.54, 0.05)
                glBegin(GL_QUADS)
                glVertex2f(x0 + inset, y0 + inset)
                glVertex2f(x1 - inset, y0 + inset)
                glVertex2f(x1 - inset, y1 - inset)
                glVertex2f(x0 + inset, y1 - inset)
                glEnd()

                # Bright pulsing center overlay for clarity.
                inset2 = tile * 0.28
                glColor3f(1.0, 0.95, 0.24 + 0.22 * pulse)
                glBegin(GL_QUADS)
                glVertex2f(x0 + inset2, y0 + inset2)
                glVertex2f(x1 - inset2, y0 + inset2)
                glVertex2f(x1 - inset2, y1 - inset2)
                glVertex2f(x0 + inset2, y1 - inset2)
                glEnd()

                # Thick active border.
                glLineWidth(3.0)
                glColor3f(1.0, 0.95, 0.15)
                glBegin(GL_LINE_LOOP)
                glVertex2f(x0 - 1.0, y0 - 1.0)
                glVertex2f(x1 + 1.0, y0 - 1.0)
                glVertex2f(x1 + 1.0, y1 + 1.0)
                glVertex2f(x0 - 1.0, y1 + 1.0)
                glEnd()
                glLineWidth(1.0)

            glColor3f(0.42, 0.31, 0.08)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            glEnd()

    def shutdown(self) -> None:
        for mesh in self.chunk_meshes.values():
            mesh.delete()
        glDeleteTextures(1, [self.texture_atlas])
        pygame.quit()

    def render_player_view(self, player_idx: int, vx: int, vy: int, vw: int, vh: int) -> None:
        glViewport(vx, vy, vw, vh)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV_DEG, vw / max(1.0, float(vh)), 0.05, 450.0)
        glMatrixMode(GL_MODELVIEW)
        self.apply_camera(player_idx)
        self.update_lantern_light(player_idx)
        self.render_world(player_idx)
        self.render_crosshair(player_idx, vw, vh)

    def render_death_overlays_screen(self, half_w: int) -> None:
        a0 = self.death_overlay_alpha(0)
        a1 = self.death_overlay_alpha(1)
        if a0 <= 0.0 and a1 <= 0.0:
            return

        was_blend = glIsEnabled(GL_BLEND)
        was_cull = glIsEnabled(GL_CULL_FACE)
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if a0 > 0.0:
            glColor4f(0.0, 0.0, 0.0, a0)
            glBegin(GL_QUADS)
            glVertex2f(0.0, 0.0)
            glVertex2f(float(half_w), 0.0)
            glVertex2f(float(half_w), float(self.height))
            glVertex2f(0.0, float(self.height))
            glEnd()

        if a1 > 0.0:
            glColor4f(0.0, 0.0, 0.0, a1)
            glBegin(GL_QUADS)
            glVertex2f(float(half_w), 0.0)
            glVertex2f(float(self.width), 0.0)
            glVertex2f(float(self.width), float(self.height))
            glVertex2f(float(half_w), float(self.height))
            glEnd()

        if not was_blend:
            glDisable(GL_BLEND)
        if was_cull:
            glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

    def draw_screen_text(
        self,
        x: float,
        y: float,
        text: str,
        color: Tuple[int, int, int, int] = (255, 240, 180, 255),
        small: bool = True,
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        if font is None:
            font = self.ui_font_small if small else self.ui_font
        surf = font.render(text, True, color[:3])
        rgba = pygame.image.tostring(surf, "RGBA", True)
        glWindowPos2f(float(x), float(self.height - y - surf.get_height()))
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, rgba)

    def gamepad_live_status(self, pad: Optional[pygame.joystick.Joystick]) -> str:
        if pad is None:
            return "not connected"
        lx = self.read_axis(pad, 0)
        ly = self.read_axis(pad, 1)
        rx = self.read_axis(pad, 2)
        ry = self.read_axis(pad, 3)
        pressed: List[str] = []
        max_btn = min(10, pad.get_numbuttons())
        for i in range(max_btn):
            if pad.get_button(i):
                pressed.append(str(i))
        btn_text = ",".join(pressed) if pressed else "-"
        return f"LS({lx:+.2f},{ly:+.2f}) RS({rx:+.2f},{ry:+.2f}) BTN[{btn_text}]"

    def render_controls_overlay_screen(self) -> None:
        if not self.show_controls_help:
            return

        was_blend = glIsEnabled(GL_BLEND)
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pad = 26.0
        glColor4f(0.0, 0.0, 0.0, 0.76)
        glBegin(GL_QUADS)
        glVertex2f(pad, pad)
        glVertex2f(self.width - pad, pad)
        glVertex2f(self.width - pad, self.height - pad)
        glVertex2f(pad, self.height - pad)
        glEnd()

        col_w = (self.width - pad * 3.0) * 0.5
        left_x = pad + 18.0
        right_x = pad * 2.0 + col_w + 18.0
        top_y = pad + 14.0

        self.draw_screen_text(left_x, top_y, "Controls (Press C to close)", (255, 242, 120, 255), small=False)
        self.draw_screen_text(left_x, top_y + 36.0, "Player 1 - Keyboard/Mouse only", (255, 255, 170, 255), small=True)
        self.draw_screen_text(left_x, top_y + 58.0, "W/A/S/D move    Mouse look    Space jump", (235, 230, 210, 255))
        self.draw_screen_text(left_x, top_y + 80.0, "LMB hold mine   RMB place block   E place mine", (235, 230, 210, 255))
        self.draw_screen_text(left_x, top_y + 102.0, "1/2/3 select block   . show player lines", (235, 230, 210, 255))
        self.draw_screen_text(left_x, top_y + 136.0, "Player 1 does not use gamepad", (220, 225, 205, 255))
        self.draw_screen_text(left_x, top_y + 158.0, "Use USB gamepad for Player 2 only", (220, 225, 205, 255))

        self.draw_screen_text(right_x, top_y + 36.0, "Player 2 - Gamepad required", (255, 255, 170, 255), small=True)
        self.draw_screen_text(right_x, top_y + 58.0, "LS move    RS look    A jump", (235, 230, 210, 255))
        self.draw_screen_text(right_x, top_y + 80.0, "RB mine    X place block    Y place mine", (235, 230, 210, 255))
        self.draw_screen_text(right_x, top_y + 102.0, "B/LB cycle block", (235, 230, 210, 255))

        self.draw_screen_text(right_x, top_y + 136.0, "Shared", (255, 255, 170, 255), small=True)
        self.draw_screen_text(right_x, top_y + 158.0, "Esc quit   C toggle this help", (235, 230, 210, 255))

        status_y = top_y + 206.0
        self.draw_screen_text(left_x, status_y, "USB Gamepad Setup", (255, 232, 130, 255), small=True)
        self.draw_screen_text(left_x, status_y + 22.0, "F2 cycle P2 pad", (225, 220, 205, 255))
        self.draw_screen_text(left_x, status_y + 44.0, "F3 auto-assign P2 pad    F5 rescan USB pads", (225, 220, 205, 255))

        p1_name = "Keyboard/Mouse"
        p2_name = "None"
        if self.p2_pad is not None:
            p2_name = f"#{self.p2_pad_slot} {self.p2_pad.get_name()}"
        self.draw_screen_text(left_x, status_y + 72.0, f"P1 assigned: {p1_name}", (255, 246, 170, 255))
        self.draw_screen_text(left_x, status_y + 94.0, f"P2 assigned: {p2_name}", (255, 246, 170, 255))
        self.draw_screen_text(left_x, status_y + 124.0, "P1 live: keyboard/mouse", (242, 238, 208, 255))
        self.draw_screen_text(left_x, status_y + 146.0, f"P2 live: {self.gamepad_live_status(self.p2_pad)}", (242, 238, 208, 255))

        detected = len(self.gamepads)
        detect_col = (255, 246, 170, 255) if detected > 0 else (255, 120, 120, 255)
        self.draw_screen_text(right_x, status_y, f"Detected USB gamepads: {detected}", detect_col, small=True)
        max_list = min(5, len(self.gamepads))
        for i in range(max_list):
            pad = self.gamepads[i]
            flags: List[str] = []
            if self.p2_pad_slot == i:
                flags.append("P2")
            label = ",".join(flags) if flags else "free"
            axes = pad.get_numaxes()
            btns = pad.get_numbuttons()
            self.draw_screen_text(
                right_x,
                status_y + 24.0 + i * 22.0,
                f"#{i} [{label}] {pad.get_name()}  ({axes} axes, {btns} btn)",
                (230, 226, 206, 255),
            )
        if len(self.gamepads) == 0:
            self.draw_screen_text(right_x, status_y + 24.0, "No controller detected. Plug in USB/Bluetooth pad, then press F5.", (255, 156, 130, 255))

        if not was_blend:
            glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            if self.in_title_screen:
                if not pygame.mouse.get_visible():
                    pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                self.process_title_input()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.render_title_screen()
                pygame.display.flip()
                continue

            if pygame.mouse.get_visible():
                pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            self.update_player_up_smoothing(dt)
            self.process_input(dt)
            self.update_hazard_damage(dt)
            self.update_mines(dt)
            self.update_break_feedback(dt)
            self.update_explosion_effects(dt)
            self.update_respawns(dt)
            self.update_health_feedback(dt)
            self.update_dirty_meshes(per_frame=5)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            half_w = self.width // 2
            self.render_player_view(0, 0, 0, half_w, self.height)
            self.render_player_view(1, half_w, 0, self.width - half_w, self.height)
            self.render_death_overlays_screen(half_w)
            self.render_controls_overlay_screen()

            # Divider between splits.
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_DEPTH_TEST)
            glColor3f(0.1, 0.08, 0.08)
            glBegin(GL_QUADS)
            glVertex2f(half_w - 1, 0)
            glVertex2f(half_w + 1, 0)
            glVertex2f(half_w + 1, self.height)
            glVertex2f(half_w - 1, self.height)
            glEnd()
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_LIGHTING)
            pygame.display.flip()

        self.shutdown()


if __name__ == "__main__":
    try:
        Game().run()
    except Exception as exc:
        pygame.quit()
        print(f"Fatal error: {exc}")
        sys.exit(1)
