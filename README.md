# The War Below (OpenGL)

A split-screen, two-player voxel trench warfare mining game built with Python + OpenGL (tested on macOS 13.7.8).

## Features
- 2-player split-screen with both players having the same abilities.
- Player 1 supports keyboard/mouse and optional gamepad.
- Player 2 is gamepad-only.
- Players spawn on opposite sides of the cube-world.
- Procedural cube world with trench-like muddy terrain and cross-face traversal gravity.
- Mining/building, timed mines, chain reactions, and Bomberman-style directional blasts.
- Toxic hazard faces, x-ray hazard mines, and split-screen HUD/radar indicators.
- Dynamic lantern lighting and particle-heavy explosions.

## Controls
### Player 1 (left screen)
Keyboard/Mouse:
- `W/A/S/D`: move
- `Mouse`: look
- `Space`: jump
- `Left Click` (hold): mine
- `Right Click`: place block
- `E`: place timed mine
- `1`/`2`/`3`: select block

Gamepad (optional):
- Left stick: move
- Right stick: look
- `A` (button 0): jump
- `RB` (button 5 hold): mine
- `X` (button 2): place block
- `Y` (button 3): place timed mine
- `B`/`LB` (buttons 1/4): cycle block selection

### Player 2 (right screen, gamepad required)
- Left stick: move
- Right stick: look
- `A` (button 0): jump
- `RB` (button 5 hold): mine
- `X` (button 2): place block
- `Y` (button 3): place timed mine
- `B`/`LB` (buttons 1/4): cycle block selection

### Global
- `Esc`: quit

## Run
```bash
cd "/Users/md/Downloads/Not Minecraft"
python3 -m pip install -r requirements.txt
python3 main.py
```
