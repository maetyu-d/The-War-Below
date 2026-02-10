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
GRAVITY = 52.0
JUMP_SPEED = 10.0
PLAYER_SPEED = 7.0
MOUSE_SENS = 0.0027
PLAYER_RADIUS = 0.35
PLAYER_EYE_OFFSET = 0.62
PLAYER_BODY_LOWER = -0.20
PLAYER_BODY_UPPER = 0.40
PLAYER_HEAD_RADIUS_SCALE = 0.92
REACH = 2.0
GRAVITY_BLEND_POWER = 6.0
MINE_FUSE_SECONDS = 10.0
MINE_TIME_DIRT = 0.28
MINE_TIME_STONE = 0.55
MINE_TIME_WOOD = 0.42
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

CUBE_HALF = 20
FACE_RELIEF = 10

TEX_SIZE = 16
ATLAS_COLS = 2
ATLAS_ROWS = 2

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
        self.width = 1280
        self.height = 720
        self.screen_center = (self.width // 2, self.height // 2)
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
        gluPerspective(75.0, self.width / self.height, 0.05, 450.0)
        glMatrixMode(GL_MODELVIEW)

        self.texture_atlas = self.create_texture_atlas()

        self.clock = pygame.time.Clock()
        self.world = World(seed=2106)
        self.world.generate()

        spawn = self.find_spawn_point()
        self.player = Player(spawn[0], spawn[1], spawn[2], look_x=0.0, look_y=0.0, look_z=-1.0)
        self.stabilize_player_spawn()

        self.chunk_meshes: Dict[ChunkKey, ChunkMesh] = {}
        self.dirty_chunks: Set[ChunkKey] = set()
        self.build_all_chunk_meshes()
        self.mines: Dict[Vec3i, Dict[str, object]] = {}

        self.selected_block = DIRT
        self.mining_target: Optional[Vec3i] = None
        self.mining_progress = 0.0
        self.break_pulse = 0.0
        self.break_particles: List[Dict[str, float]] = []
        self.explosion_particles: List[Dict[str, float]] = []
        self.mouse_settle_frames = 3
        self.running = True

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

                        r = base[0] * shade
                        g = base[1] * shade
                        b = base[2] * shade

                        for idx in TRI_IDX:
                            vx, vy, vz = face[idx]
                            uv = UV_QUAD[idx]
                            uu, vv = self.atlas_uv(tex_idx, uv[0], uv[1])
                            verts.extend((x + vx, y + vy, z + vz, nx, ny, nz, uu, vv, r, g, b))

        mesh.upload(verts)

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

    def player_pos(self) -> Vec3f:
        return (self.player.x, self.player.y, self.player.z)

    def set_player_pos(self, p: Vec3f) -> None:
        self.player.x, self.player.y, self.player.z = p

    def player_vel(self) -> Vec3f:
        return (self.player.vx, self.player.vy, self.player.vz)

    def set_player_vel(self, v: Vec3f) -> None:
        self.player.vx, self.player.vy, self.player.vz = v

    def look_dir(self) -> Vec3f:
        return v_norm((self.player.look_x, self.player.look_y, self.player.look_z))

    def set_look_dir(self, d: Vec3f) -> None:
        dn = v_norm(d)
        self.player.look_x, self.player.look_y, self.player.look_z = dn

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

    def place_mine(self) -> None:
        pos = self.player_pos()
        fwd, right, up = self.camera_basis(pos)
        up_i = self.snap_axis_dir(up)
        right_i = self.snap_axis_dir(right, (up_i,))
        fwd_i = self.snap_axis_dir(fwd, (up_i, right_i))

        base = (math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))
        key = (base[0] + fwd_i[0], base[1] + fwd_i[1], base[2] + fwd_i[2])
        if not self.world.in_bounds(*key):
            return
        if self.player_intersects(*key):
            return
        if key in self.mines:
            return

        target_block = self.world.block_at(*key)
        normal = up_i
        support: Vec3i
        if target_block != AIR:
            # One forward is a wall; attach to the wall face toward the player.
            normal = (-fwd_i[0], -fwd_i[1], -fwd_i[2])
            support = key
        else:
            # In air: prefer sitting on ground below local up.
            below = (key[0] - up_i[0], key[1] - up_i[1], key[2] - up_i[2])
            if self.world.in_bounds(*below) and self.world.block_at(*below) != AIR:
                normal = up_i
                support = below
            else:
                return

        self.mines[key] = {
            "pos": key,
            "timer": MINE_FUSE_SECONDS,
            "up": up_i,
            "right": right_i,
            "forward": fwd_i,
            "normal": normal,
            "support": support,
        }

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
        up = mine["up"]
        right = mine["right"]
        fwd = mine["forward"]
        if not isinstance(pos, tuple) or not isinstance(up, tuple) or not isinstance(right, tuple) or not isinstance(fwd, tuple):
            return

        if pos not in self.mines:
            return
        self.mines.pop(pos, None)

        cells: Set[Vec3i] = set()
        cell_dirs: Dict[Vec3i, Vec3i] = {}
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

        blast_breaks_rock = False
        player_hit = False
        for cx, cy, cz in cells:
            if not self.world.in_bounds(cx, cy, cz):
                continue
            block = self.world.block_at(cx, cy, cz)
            if block == STONE:
                blast_breaks_rock = True
            if self.player_intersects(cx, cy, cz):
                player_hit = True

        if blast_breaks_rock and player_hit:
            self.kill_player()

        # Visual blast follows the same pattern: restrained center, stronger directional arms.
        self.spawn_explosion_effect((pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5), scale=0.55, axis=None)
        for (cx, cy, cz), axis_dir in cell_dirs.items():
            if (cx, cy, cz) == pos:
                continue
            dist = abs(cx - pos[0]) + abs(cy - pos[1]) + abs(cz - pos[2])
            scale = max(0.22, 1.0 - dist * 0.06)
            self.spawn_explosion_effect((cx + 0.5, cy + 0.5, cz + 0.5), scale=scale, axis=axis_dir)

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
            self.spawn_break_feedback((cx, cy, cz), block)

        # Chain reaction: any armed mine inside blast cells detonates immediately.
        chained_positions = [cell for cell in cells if cell in self.mines]
        for chained_pos in chained_positions:
            chained = self.mines.get(chained_pos)
            if chained is not None:
                self.detonate_mine(chained)

    def kill_player(self) -> None:
        self.running = False

    def update_mines(self, dt: float) -> None:
        if not self.mines:
            return
        unsupported: List[Vec3i] = []
        to_detonate: List[Dict[str, object]] = []
        for mine in self.mines.values():
            if not self.mine_has_support(mine):
                pos = mine.get("pos")
                if isinstance(pos, tuple):
                    unsupported.append(pos)
                continue
            t = float(mine["timer"]) - dt
            mine["timer"] = t
            if t <= 0.0:
                to_detonate.append(mine)

        for pos in unsupported:
            self.mines.pop(pos, None)

        for mine in to_detonate:
            self.detonate_mine(mine)

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

    def gravity_dir(self, pos: Vec3f) -> Vec3f:
        up = self.outward_up(pos)
        return (-up[0], -up[1], -up[2])

    def camera_basis(self, pos: Vec3f) -> Tuple[Vec3f, Vec3f, Vec3f]:
        up = self.outward_up(pos)
        look = self.look_dir()

        fwd = v_sub(look, v_scale(up, v_dot(look, up)))
        if v_len(fwd) < 1e-5:
            seed = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 0.0, 1.0)
            fwd = v_cross(seed, up)
        fwd = v_norm(fwd)

        right = v_norm(v_cross(fwd, up))
        return (fwd, right, up)

    def eye_pos(self) -> Vec3f:
        pos = self.player_pos()
        up = self.outward_up(pos)
        return v_add(pos, v_scale(up, PLAYER_EYE_OFFSET))

    def handle_mouse_look(self) -> None:
        if self.mouse_settle_frames > 0:
            pygame.mouse.set_pos(self.screen_center)
            pygame.event.clear(pygame.MOUSEMOTION)
            pygame.mouse.get_rel()
            self.mouse_settle_frames -= 1
            return

        mx, my = pygame.mouse.get_rel()
        if mx == 0 and my == 0:
            return

        pos = self.player_pos()
        fwd, right, up = self.camera_basis(pos)

        look = self.look_dir()
        look = rotate_axis(look, up, -mx * MOUSE_SENS)
        look = rotate_axis(look, right, -my * MOUSE_SENS)

        self.set_look_dir(look)

    def process_input(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.selected_block = DIRT
                elif event.key == pygame.K_2:
                    self.selected_block = STONE
                elif event.key == pygame.K_3:
                    self.selected_block = WOOD
                elif event.key == pygame.K_e:
                    self.place_mine()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                hit, prev = self.raycast_block(REACH)
                if event.button == 3 and prev:
                    px, py, pz = prev
                    if self.world.block_at(px, py, pz) == AIR and not self.player_intersects(px, py, pz):
                        self.set_world_block(px, py, pz, self.selected_block)

        self.handle_mouse_look()

        keys = pygame.key.get_pressed()
        move_x = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
        move_z = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)

        pos = self.player_pos()
        fwd, right, up = self.camera_basis(pos)
        gravity = self.gravity_dir(pos)

        wish = (0.0, 0.0, 0.0)
        wish = v_add(wish, v_scale(fwd, float(move_z)))
        wish = v_add(wish, v_scale(right, float(move_x)))
        if v_len(wish) > 0.001:
            wish = v_scale(v_norm(wish), PLAYER_SPEED)

        vel = self.player_vel()
        gspeed = v_dot(vel, gravity)
        vel = v_add(wish, v_scale(gravity, gspeed))

        if self.player.on_ground and keys[pygame.K_SPACE]:
            vel = v_add(vel, v_scale(up, JUMP_SPEED))
            self.player.on_ground = False

        vel = v_add(vel, v_scale(gravity, GRAVITY * dt))
        self.set_player_vel(vel)

        delta = v_scale(vel, dt)
        self.move_and_collide(delta)
        self.update_mining(dt, pygame.mouse.get_pressed(3)[0])

        pos = self.player_pos()
        self.player.on_ground = self.collides_body(v_add(pos, v_scale(self.gravity_dir(pos), 0.08)))

    def mine_time_for(self, block: int) -> float:
        if block in (STONE, CHARRED_STONE):
            return MINE_TIME_STONE
        if block == WOOD:
            return MINE_TIME_WOOD
        return MINE_TIME_DIRT

    def spawn_break_feedback(self, cell: Vec3i, block: int) -> None:
        cx = cell[0] + 0.5
        cy = cell[1] + 0.5
        cz = cell[2] + 0.5
        base = BLOCK_COLORS.get(block, (0.45, 0.35, 0.25))
        eye = self.eye_pos()
        outward = v_norm((cx - eye[0], cy - eye[1], cz - eye[2]))
        if v_len(outward) < 1e-5:
            outward = self.look_dir()
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
                }
            )
        self.break_pulse = 1.0

    def update_break_feedback(self, dt: float) -> None:
        if self.break_pulse > 0.0:
            self.break_pulse = max(0.0, self.break_pulse - dt * 4.8)

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

    def spawn_explosion_effect(self, center: Vec3f, scale: float = 1.0, axis: Optional[Vec3i] = None) -> None:
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
                }
            )
        self.break_pulse = max(self.break_pulse, 1.4 + 0.8 * scale)

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

    def update_mining(self, dt: float, is_mining: bool) -> None:
        if not is_mining:
            self.mining_target = None
            self.mining_progress = 0.0
            return

        hit, _ = self.raycast_block(REACH)
        if hit is None:
            self.mining_target = None
            self.mining_progress = 0.0
            return

        if hit != self.mining_target:
            self.mining_target = hit
            self.mining_progress = 0.0

        block = self.world.block_at(*hit)
        if block == AIR:
            self.mining_target = None
            self.mining_progress = 0.0
            return

        self.mining_progress += dt
        if self.mining_progress >= self.mine_time_for(block):
            broken_block = block
            self.set_world_block(*hit, AIR)
            self.spawn_break_feedback(hit, broken_block)
            self.mining_target = None
            self.mining_progress = 0.0

    def player_intersects(self, bx: int, by: int, bz: int) -> bool:
        pos = self.player_pos()
        up = self.outward_up(pos)
        lower = v_add(pos, v_scale(up, PLAYER_BODY_LOWER))
        upper = v_add(pos, v_scale(up, PLAYER_BODY_UPPER))
        return (
            self.sphere_aabb_intersect(lower, PLAYER_RADIUS * 1.05, bx, by, bz)
            or self.sphere_aabb_intersect(upper, PLAYER_RADIUS * PLAYER_HEAD_RADIUS_SCALE, bx, by, bz)
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

    def move_and_collide(self, delta: Vec3f) -> None:
        pos = self.player_pos()
        vel = list(self.player_vel())

        for axis in range(3):
            step = [0.0, 0.0, 0.0]
            step[axis] = delta[axis]
            trial = (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2])
            if not self.collides_body(trial):
                pos = trial
            else:
                vel[axis] = 0.0

        self.set_player_pos(pos)
        self.set_player_vel((vel[0], vel[1], vel[2]))

    def raycast_block(self, max_dist: float) -> Tuple[Optional[Vec3i], Optional[Vec3i]]:
        ox, oy, oz = self.eye_pos()
        dx, dy, dz = self.look_dir()
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

    def find_spawn_point(self) -> Vec3f:
        x = self.world.cx
        z = self.world.cz
        top = self.world.find_surface_y(x, z)
        if top is None:
            return (self.world.cx + 0.5, self.world.cy + CUBE_HALF + 6.0, self.world.cz + 0.5)
        return (x + 0.5, top + 1.5, z + 0.5)

    def stabilize_player_spawn(self) -> None:
        # Resolve any tiny initial overlap with terrain so movement works immediately.
        for _ in range(40):
            pos = self.player_pos()
            if not self.collides_body(pos):
                break
            up = self.outward_up(pos)
            self.set_player_pos(v_add(pos, v_scale(up, 0.08)))

        pos = self.player_pos()
        self.player.on_ground = self.collides_body(v_add(pos, v_scale(self.gravity_dir(pos), 0.10)))

    def render_world(self) -> None:
        px, pz = self.player.x, self.player.z
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

        self.render_mines()
        self.render_break_particles()
        self.render_explosion_particles()

    def render_mines(self) -> None:
        if not self.mines:
            return

        was_blend = glIsEnabled(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

        for mine in self.mines.values():
            pos = mine.get("pos")
            normal = mine.get("normal")
            support = mine.get("support")
            timer = float(mine.get("timer", 0.0))
            if not isinstance(pos, tuple) or not isinstance(normal, tuple) or not isinstance(support, tuple):
                continue

            n = v_norm((float(normal[0]), float(normal[1]), float(normal[2])))
            if v_len(n) < 1e-6:
                continue

            # Build tangent basis from mine normal.
            ref = (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0)
            u = v_norm(v_cross(ref, n))
            v = v_norm(v_cross(n, u))

            r = 0.22
            # Position: sit flush on the support face with a tiny outward offset.
            support_center = (support[0] + 0.5, support[1] + 0.5, support[2] + 0.5)
            center = v_add(support_center, v_scale(n, 0.5 + r * 0.72))

            # Closed low-poly sphere body (round mine, fully opaque).
            stacks = 7
            slices = 14
            glColor3f(0.015, 0.012, 0.01)
            for i in range(stacks):
                t0 = math.pi * i / stacks
                t1 = math.pi * (i + 1) / stacks
                glBegin(GL_TRIANGLE_STRIP)
                for j in range(slices + 1):
                    p = 2.0 * math.pi * j / slices
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

            # Flashing red indicator on top.
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
            glEnable(GL_LIGHTING)

        glEnable(GL_CULL_FACE)
        if was_blend:
            glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)

    def render_break_particles(self) -> None:
        if not self.break_particles:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for p in self.break_particles:
            if p["d"] > 0.5:
                continue
            alpha = clamp(p["life"] / p["ttl"], 0.0, 1.0)
            glColor4f(p["r"], p["g"], p["b"], alpha)
            glVertex3f(p["x"], p["y"], p["z"])
        glEnd()

        glPointSize(7.0)
        glBegin(GL_POINTS)
        for p in self.break_particles:
            if p["d"] <= 0.5:
                continue
            alpha = clamp(p["life"] / p["ttl"], 0.0, 1.0)
            glColor4f(p["r"], p["g"], p["b"], alpha * 0.9)
            glVertex3f(p["x"], p["y"], p["z"])
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

    def render_explosion_particles(self) -> None:
        if not self.explosion_particles:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glPointSize(16.0)
        glBegin(GL_POINTS)
        for p in self.explosion_particles:
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
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

    def update_lantern_light(self) -> None:
        ex, ey, ez = self.eye_pos()
        glLightfv(GL_LIGHT0, GL_POSITION, (ex, ey, ez, 1.0))

    def render_crosshair(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)
        cx = self.width // 2
        cy = self.height // 2
        size = 8.0 + self.break_pulse * 5.0
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()

        # Mining progress line under reticle.
        if self.mining_target is not None:
            target_block = self.world.block_at(*self.mining_target)
            if target_block != AIR:
                req = self.mine_time_for(target_block)
                if req > 0.0:
                    progress = clamp(self.mining_progress / req, 0.0, 1.0)
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

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def apply_camera(self) -> None:
        glLoadIdentity()
        eye = self.eye_pos()
        look = self.look_dir()
        up = self.outward_up(self.player_pos())
        target = v_add(eye, look)
        gluLookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2], up[0], up[1], up[2])

    def shutdown(self) -> None:
        for mesh in self.chunk_meshes.values():
            mesh.delete()
        glDeleteTextures(1, [self.texture_atlas])
        pygame.quit()

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            self.process_input(dt)
            self.update_mines(dt)
            self.update_break_feedback(dt)
            self.update_explosion_effects(dt)
            self.update_dirty_meshes(per_frame=5)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.apply_camera()
            self.update_lantern_light()
            self.render_world()
            self.render_crosshair()
            pygame.display.flip()

        self.shutdown()


if __name__ == "__main__":
    try:
        Game().run()
    except Exception as exc:
        pygame.quit()
        print(f"Fatal error: {exc}")
        sys.exit(1)
