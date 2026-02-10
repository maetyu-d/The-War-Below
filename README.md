# Not Minecraft (OpenGL)

A first-person voxel mining game built with Python + OpenGL.

## Features
- Blasted stone is scorched into charred rock instead of disappearing.
- Explosion visuals now propagate along the same directional blast pattern as mine damage.
- Mine explosions use a multi-phase effect with dense blast particles, bright sparks, and a shock ring.
- Mines require solid support (ground/wall); unsupported mines are removed and never float.
- Mines render as round black landmines that sit on surfaces/attach to walls and flash red while armed.
- Timed mine detonation now includes a visible explosion burst effect.
- Mines explode after 10s and clear: center, left/right, forward/back, and 3 blocks up/down (relative to placement up).
- Dirt/soil break effects: denser muddy particle bursts with heavier settling motion.
- Polished block breaking: hold-to-mine progress bar, impact particles, and reticle punch on break.
- Guaranteed soil tunnel paths through the cube from face to face, with occasional rock rubble inside.
- Player lantern (point light) that illuminates nearby surfaces, especially in tunnels.
- Procedural **cube world** (a giant block/moon) with terrain on all 6 faces.
- Face-aware gravity with edge/corner blending for smooth transitions between faces.
- Seamless edge traversal between faces with camera reorientation via local up vector.
- First-person mining (`LMB`) and building (`RMB`).
- Procedurally generated low-fi muddy texture atlas.
- Chunk meshing + GPU VBO rendering.

## Controls
- `W/A/S/D`: move
- `Mouse`: look around
- `Space`: jump
- `Left Click` (hold): mine target block with resistance
- `Right Click`: place selected block
- `1`/`2`/`3`: select dirt/stone/wood
- `E`: place timed mine (10s fuse)
- `Esc`: quit

## Run
```bash
cd "/Users/md/Downloads/Not Minecraft"
python3 -m pip install -r requirements.txt
python3 main.py
```
