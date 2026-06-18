"""
Tetris task with EEG recording via LSL.

Each EEG sample is annotated with the current Tetris game state:

    lsl_timestamp; ch0; ...; chN-1; current_piece; next1; next2; next3; next4; next5

A separate markers CSV logs game events (line clears, level ups, game over).

Pieces: I O T S Z J L (7-bag randomizer, NES-style gravity progression)
Controls: ← → move │ ↑ / Z rotate │ ↓ soft drop │ Space hard drop │ Esc quit

Usage
-----
    python scripts/tetris_task.py --duration 300
    python scripts/tetris_task.py --duration 600 --prefix subjectX --out-dir ./data/subjectX
    python scripts/tetris_task.py --no-eeg-check --duration 60   # test without PiEEG
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from datetime import datetime

import pygame
from pylsl import StreamInlet, StreamInfo, StreamOutlet, local_clock, resolve_byprop

# ── Board geometry ────────────────────────────────────────────────────────────
BOARD_COLS = 10
BOARD_ROWS = 20
CELL       = 30          # px per cell

BOARD_LEFT = 20
BOARD_TOP  = 30
PANEL_LEFT = BOARD_LEFT + BOARD_COLS * CELL + 15
WIN_W      = PANEL_LEFT + 190
WIN_H      = BOARD_TOP  + BOARD_ROWS * CELL + 22

# ── Colors ────────────────────────────────────────────────────────────────────
BG     = (  0,   0,   0)
GRID_C = ( 35,  35,  35)
TEXT_C = (220, 220, 220)
DIM_C  = (100, 100, 100)

PIECE_COLORS = {
    'I': (  0, 220, 220),
    'O': (220, 220,   0),
    'T': (160,   0, 220),
    'S': (  0, 200,   0),
    'Z': (220,   0,   0),
    'J': (  0,  80, 220),
    'L': (220, 140,   0),
}
PIECE_NAMES = list(PIECE_COLORS)

# ── Piece shapes ──────────────────────────────────────────────────────────────
# 4 rotation states, each a list of (row, col) offsets from spawn origin.
# Bounding boxes: I/O → 4×4;  T S Z J L → 3×3.
SHAPES = {
    'I': [
        [(1, 0), (1, 1), (1, 2), (1, 3)],   # ....XXXX....
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ],
    'O': [                                   # all rotations identical
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
    ],
    'T': [
        [(0, 1), (1, 0), (1, 1), (1, 2)],   # .X. / XXX / ...
        [(0, 1), (1, 1), (1, 2), (2, 1)],   # .X. / .XX / .X.
        [(1, 0), (1, 1), (1, 2), (2, 1)],   # ... / XXX / .X.
        [(0, 1), (1, 0), (1, 1), (2, 1)],   # .X. / XX. / .X.
    ],
    'S': [                                   # two unique states
        [(0, 1), (0, 2), (1, 0), (1, 1)],   # .XX / XX.
        [(0, 1), (1, 1), (1, 2), (2, 2)],   # .X. / .XX / ..X
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
    ],
    'Z': [                                   # two unique states
        [(0, 0), (0, 1), (1, 1), (1, 2)],   # XX. / .XX
        [(0, 2), (1, 1), (1, 2), (2, 1)],   # ..X / .XX / .X.
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
    ],
    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],   # X.. / XXX / ...
        [(0, 1), (0, 2), (1, 1), (2, 1)],   # .XX / .X. / .X.
        [(1, 0), (1, 1), (1, 2), (2, 2)],   # ... / XXX / ..X
        [(0, 1), (1, 1), (2, 0), (2, 1)],   # .X. / .X. / XX.
    ],
    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],   # ..X / XXX / ...
        [(0, 1), (1, 1), (2, 1), (2, 2)],   # .X. / .X. / .XX
        [(1, 0), (1, 1), (1, 2), (2, 0)],   # ... / XXX / X..
        [(0, 0), (0, 1), (1, 1), (2, 1)],   # XX. / .X. / .X.
    ],
}

# ── NES-style gravity (seconds per automatic drop) ────────────────────────────
GRAVITY_TABLE = [
    0.800, 0.717, 0.633, 0.550, 0.467, 0.383,
    0.300, 0.217, 0.133, 0.100, 0.083, 0.083,
    0.083, 0.067, 0.067, 0.067, 0.050, 0.050,
    0.050, 0.033,
]

# ── LSL marker codes ──────────────────────────────────────────────────────────
MKR_START     =  1
MKR_LINE1     = 11   # single
MKR_LINE2     = 12   # double
MKR_LINE3     = 13   # triple
MKR_TETRIS    = 14   # Tetris (4 lines)
MKR_LEVELUP   = 20
MKR_GAMEOVER  = 30
MKR_END       = 99


# ── Game ──────────────────────────────────────────────────────────────────────

class _Bag:
    def __init__(self):
        self._bag: list[str] = []

    def draw(self) -> str:
        if not self._bag:
            self._bag = list(PIECE_NAMES)
            random.shuffle(self._bag)
        return self._bag.pop()


class _Piece:
    def __init__(self, name: str, row: int, col: int, rot: int = 0):
        self.name = name
        self.row  = row
        self.col  = col
        self.rot  = rot

    def cells(self) -> list[tuple[int, int]]:
        return [(self.row + dr, self.col + dc)
                for dr, dc in SHAPES[self.name][self.rot]]

    def moved(self, dr: int, dc: int) -> '_Piece':
        return _Piece(self.name, self.row + dr, self.col + dc, self.rot)

    def rotated(self, delta: int) -> '_Piece':
        return _Piece(self.name, self.row, self.col, (self.rot + delta) % 4)


SPAWN_ROW = 0
SPAWN_COL = 3


class TetrisGame:
    def __init__(self):
        self.board: list[list[str | None]] = [
            [None] * BOARD_COLS for _ in range(BOARD_ROWS)
        ]
        self._bag       = _Bag()
        # Keep ≥6 in queue so we always have 5 "nexts"
        self._queue: deque[str] = deque(self._bag.draw() for _ in range(6))

        self.score     = 0
        self.lines     = 0
        self.level     = 0
        self.game_over = False

        self.lines_cleared = 0   # set after each lock; caller reads & resets
        self.current       = self._spawn()
        self.last_drop     = time.perf_counter()

    # ── public interface ──────────────────────────────────────────────────────

    def move(self, dc: int) -> bool:
        moved = self.current.moved(0, dc)
        if self._valid(moved):
            self.current = moved
            return True
        return False

    def rotate(self, delta: int = 1) -> bool:
        rotated = self.current.rotated(delta)
        for kick in (0, -1, 1, -2, 2):
            kicked = rotated.moved(0, kick)
            if self._valid(kicked):
                self.current = kicked
                return True
        return False

    def soft_drop(self) -> bool:
        dropped = self.current.moved(1, 0)
        if self._valid(dropped):
            self.current  = dropped
            self.last_drop = time.perf_counter()
            self.score   += 1
            return True
        self._lock()
        return False

    def hard_drop(self):
        n = 0
        while True:
            dropped = self.current.moved(1, 0)
            if self._valid(dropped):
                self.current = dropped
                n += 1
            else:
                break
        self.score += n * 2
        self._lock()

    def gravity_tick(self):
        dropped = self.current.moved(1, 0)
        if self._valid(dropped):
            self.current = dropped
        else:
            self._lock()

    @property
    def gravity_interval(self) -> float:
        return GRAVITY_TABLE[min(self.level, len(GRAVITY_TABLE) - 1)]

    @property
    def next5(self) -> list[str]:
        return list(self._queue)[:5]

    @property
    def current_name(self) -> str:
        return 'none' if self.game_over else self.current.name

    def ghost_row(self) -> int:
        p = self.current
        while self._valid(p.moved(1, 0)):
            p = p.moved(1, 0)
        return p.row

    # ── internals ─────────────────────────────────────────────────────────────

    def _valid(self, p: _Piece) -> bool:
        for r, c in p.cells():
            if c < 0 or c >= BOARD_COLS or r >= BOARD_ROWS:
                return False
            if r >= 0 and self.board[r][c] is not None:
                return False
        return True

    def _spawn(self) -> _Piece:
        name = self._queue.popleft()
        self._queue.append(self._bag.draw())
        p = _Piece(name, SPAWN_ROW, SPAWN_COL)
        if not self._valid(p):
            self.game_over = True
        return p

    def _lock(self):
        for r, c in self.current.cells():
            if 0 <= r < BOARD_ROWS:
                self.board[r][c] = self.current.name

        full = [r for r in range(BOARD_ROWS)
                if all(cell is not None for cell in self.board[r])]
        for r in full:
            del self.board[r]
            self.board.insert(0, [None] * BOARD_COLS)

        n = len(full)
        self.lines_cleared  = n
        self.lines         += n
        self.score         += [0, 100, 300, 500, 800][n] * (self.level + 1)
        self.level          = min(self.lines // 10, len(GRAVITY_TABLE) - 1)

        self.current   = self._spawn()
        self.last_drop = time.perf_counter()


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_cell(surf, color: tuple, row: int, col: int):
    x  = BOARD_LEFT + col * CELL + 1
    y  = BOARD_TOP  + row * CELL + 1
    r, g, b = color
    rect = pygame.Rect(x, y, CELL - 2, CELL - 2)
    pygame.draw.rect(surf, color, rect)
    hi = (min(r + 60, 255), min(g + 60, 255), min(b + 60, 255))
    pygame.draw.line(surf, hi, (x, y), (x + CELL - 3, y))
    pygame.draw.line(surf, hi, (x, y), (x, y + CELL - 3))


def _draw_mini(surf, name: str, cx: int, cy: int, sz: int = 13):
    cells = SHAPES[name][0]
    rows  = [r for r, _ in cells]
    cols  = [c for _, c in cells]
    ro    = (min(rows) + max(rows)) / 2
    co    = (min(cols) + max(cols)) / 2
    color = PIECE_COLORS[name]
    for r, c in cells:
        x = cx + int((c - co) * sz) - sz // 2
        y = cy + int((r - ro) * sz) - sz // 2
        pygame.draw.rect(surf, color, (x, y, sz - 1, sz - 1))


def draw_scene(surf, game: TetrisGame, font_ui, font_sm):
    # Board border
    pygame.draw.rect(surf, (60, 60, 60),
                     (BOARD_LEFT - 2, BOARD_TOP - 2,
                      BOARD_COLS * CELL + 4, BOARD_ROWS * CELL + 4), 2)

    # Grid lines
    for r in range(BOARD_ROWS + 1):
        y = BOARD_TOP + r * CELL
        pygame.draw.line(surf, GRID_C,
                         (BOARD_LEFT, y),
                         (BOARD_LEFT + BOARD_COLS * CELL, y))
    for c in range(BOARD_COLS + 1):
        x = BOARD_LEFT + c * CELL
        pygame.draw.line(surf, GRID_C,
                         (x, BOARD_TOP),
                         (x, BOARD_TOP + BOARD_ROWS * CELL))

    # Locked cells
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            name = game.board[r][c]
            if name:
                _draw_cell(surf, PIECE_COLORS[name], r, c)

    if not game.game_over:
        # Ghost
        ghost_r = game.ghost_row()
        ghost_surf = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
        color = PIECE_COLORS[game.current.name]
        ghost_surf.fill((*color, 55))
        for dr, dc in SHAPES[game.current.name][game.current.rot]:
            r = ghost_r + dr
            c = game.current.col + dc
            if 0 <= r < BOARD_ROWS:
                x = BOARD_LEFT + c * CELL + 1
                y = BOARD_TOP  + r * CELL + 1
                surf.blit(ghost_surf, (x, y))
                pygame.draw.rect(surf, color, (x, y, CELL - 2, CELL - 2), 1)

        # Active piece
        for dr, dc in SHAPES[game.current.name][game.current.rot]:
            r = game.current.row + dr
            c = game.current.col + dc
            if 0 <= r < BOARD_ROWS:
                _draw_cell(surf, PIECE_COLORS[game.current.name], r, c)

    # ── Side panel ────────────────────────────────────────────────────────────
    px, py = PANEL_LEFT, BOARD_TOP

    def pt(text, y, color=TEXT_C, font=None):
        surf.blit((font or font_ui).render(text, True, color), (px, y))

    pt("SCORE",       py,        DIM_C, font_sm)
    pt(str(game.score), py + 16)
    pt("LINES",       py + 52,   DIM_C, font_sm)
    pt(str(game.lines), py + 68)
    pt("LEVEL",       py + 104,  DIM_C, font_sm)
    pt(str(game.level), py + 120)
    pt("NEXT",        py + 160,  DIM_C, font_sm)

    for i, name in enumerate(game.next5):
        cy = py + 195 + i * 52
        _draw_mini(surf, name, px + 48, cy, sz=14)
        label = font_sm.render(name, True, PIECE_COLORS[name])
        surf.blit(label, (px + 88, cy - 7))


# ── LSL helpers ───────────────────────────────────────────────────────────────

def _connect_inlet(stream_name: str, timeout: float = 10.0):
    print(f"[LSL] Searching for '{stream_name}' …", flush=True)
    streams = resolve_byprop("name", stream_name, timeout=timeout)
    if not streams:
        return None
    inlet = StreamInlet(streams[0], max_buflen=1)
    info  = streams[0]
    print(f"[LSL] Connected  ({info.channel_count()} ch, {info.nominal_srate():.0f} Hz)")
    return inlet


def _make_outlet() -> StreamOutlet:
    info = StreamInfo("Tetris_Markers", "Markers", 1, 0, "int32", "tetris_001")
    return StreamOutlet(info)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tetris task with LSL EEG recording")
    p.add_argument("--duration",     type=float, default=300.0,
                   help="Recording duration in seconds (default: 300)")
    p.add_argument("--stream",       default="PiEEG")
    p.add_argument("--no-eeg-check", action="store_true",
                   help="Skip EEG stream check (test without PiEEG)")
    p.add_argument("--out-dir",      default=".",
                   help="Output directory for CSV files")
    p.add_argument("--prefix",       default="tetris",
                   help="Filename prefix (e.g. subject ID)")
    p.add_argument("--seed",         type=int, default=None)
    return p.parse_args()


def run(args):
    if args.seed is not None:
        random.seed(args.seed)

    # ── LSL ──────────────────────────────────────────────────────────────────
    eeg_inlet = None
    if not args.no_eeg_check:
        eeg_inlet = _connect_inlet(args.stream)
        if eeg_inlet is None:
            print(f"[LSL] WARNING: '{args.stream}' not found.")
            ans = input("Continue without EEG? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit(0)
    else:
        print("[LSL] EEG check skipped.")

    outlet = _make_outlet()
    print("[LSL] Tetris_Markers outlet open.\n")

    # ── CSV ──────────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    eeg_path = os.path.join(args.out_dir, f"{args.prefix}_eeg_{stamp}.csv")
    mkr_path = os.path.join(args.out_dir, f"{args.prefix}_markers_{stamp}.csv")

    eeg_tc = 0.0
    eeg_fh = eeg_writer = None

    if eeg_inlet is not None:
        n_ch   = eeg_inlet.info().channel_count()
        eeg_fh = open(eeg_path, "w", newline="")
        eeg_writer = csv.writer(eeg_fh, delimiter=";")
        eeg_writer.writerow(
            ["lsl_timestamp"] +
            [f"ch{i}" for i in range(n_ch)] +
            ["current_piece", "next1", "next2", "next3", "next4", "next5"]
        )
        eeg_tc = eeg_inlet.time_correction(timeout=5.0)
        print(f"[LSL] EEG time correction: {eeg_tc:+.3f} s")
        print(f"[CSV] EEG     → {eeg_path}")

    mkr_fh = open(mkr_path, "w", newline="")
    mkr_writer = csv.writer(mkr_fh, delimiter=";")
    mkr_writer.writerow(["lsl_timestamp", "marker_code", "marker_label",
                         "score", "lines", "level"])
    print(f"[CSV] Markers → {mkr_path}\n")

    def write_marker(code: int, label: str, game: TetrisGame):
        ts = local_clock() + eeg_tc
        outlet.push_sample([code])
        mkr_writer.writerow([f"{ts:.6f}", code, label,
                             game.score, game.lines, game.level])

    def drain_eeg(game: TetrisGame):
        if eeg_inlet is None or eeg_writer is None:
            return
        next5 = game.next5
        cur   = game.current_name
        while True:
            sample, ts = eeg_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            row = (
                [f"{ts + eeg_tc:.6f}"] +
                [f"{v:.6f}" for v in sample] +
                [cur] + next5
            )
            eeg_writer.writerow(row)

    # ── PyGame ───────────────────────────────────────────────────────────────
    pygame.init()
    pygame.key.set_repeat(160, 50)   # DAS: 160 ms delay, 50 ms repeat
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Tetris — EEG Recording")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Courier New", 26, bold=True)
    font_ui    = pygame.font.SysFont("Courier New", 20, bold=True)
    font_sm    = pygame.font.SysFont("Courier New", 14)

    # ── Instruction screen ────────────────────────────────────────────────────
    instruct = [
        "TETRIS  —  EEG",
        "",
        f"Duration : {args.duration:.0f} s",
        "",
        "← →   move",
        "↑ / Z  rotate",
        "↓      soft drop",
        "Space  hard drop",
        "Esc    quit",
        "",
        "SPACE to start",
    ]
    screen.fill(BG)
    for i, line in enumerate(instruct):
        f   = font_title if i == 0 else font_ui
        col = (0, 220, 220) if i == 0 else TEXT_C
        r   = f.render(line, True, col)
        screen.blit(r, (WIN_W // 2 - r.get_width() // 2, 60 + i * 40))
    if eeg_inlet is None:
        warn = font_sm.render("WARNING: EEG not connected — game state only",
                              True, (220, 180, 50))
        screen.blit(warn, (WIN_W // 2 - warn.get_width() // 2, WIN_H - 25))
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:
                    waiting = False
        clock.tick(30)

    # ── Game loop ─────────────────────────────────────────────────────────────
    game        = TetrisGame()
    last_level  = game.level
    session_end = time.perf_counter() + args.duration

    write_marker(MKR_START, "start", game)

    while time.perf_counter() < session_end and not game.game_over:
        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                game.game_over = True
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    game.game_over = True
                elif ev.key == pygame.K_LEFT:
                    game.move(-1)
                elif ev.key == pygame.K_RIGHT:
                    game.move(1)
                elif ev.key in (pygame.K_UP, pygame.K_z):
                    game.rotate(1)
                elif ev.key == pygame.K_DOWN:
                    game.soft_drop()
                elif ev.key == pygame.K_SPACE:
                    game.hard_drop()

        # Gravity
        now = time.perf_counter()
        if now - game.last_drop >= game.gravity_interval:
            game.gravity_tick()
            game.last_drop = now

        # Post-action events
        if game.lines_cleared > 0:
            codes  = (MKR_LINE1, MKR_LINE2, MKR_LINE3, MKR_TETRIS)
            labels = ("single", "double", "triple", "tetris")
            idx    = min(game.lines_cleared, 4) - 1
            write_marker(codes[idx], f"line_clear_{labels[idx]}", game)
            game.lines_cleared = 0

        if game.level != last_level:
            write_marker(MKR_LEVELUP, f"level_up_{game.level}", game)
            last_level = game.level

        # Drain EEG — annotated with current game state
        drain_eeg(game)

        # Draw
        screen.fill(BG)
        draw_scene(screen, game, font_ui, font_sm)

        # Progress bar below board
        elapsed = args.duration - (session_end - time.perf_counter())
        frac    = min(elapsed / args.duration, 1.0)
        bx = BOARD_LEFT
        by = BOARD_TOP + BOARD_ROWS * CELL + 6
        bw = BOARD_COLS * CELL
        pygame.draw.rect(screen, (40, 40, 40), (bx, by, bw, 6))
        pygame.draw.rect(screen, (0, 180, 220), (bx, by, int(bw * frac), 6))

        pygame.display.flip()
        clock.tick(60)

    # ── Teardown ──────────────────────────────────────────────────────────────
    if game.game_over:
        write_marker(MKR_GAMEOVER, "game_over", game)
    write_marker(MKR_END, "end", game)

    if eeg_fh:
        eeg_fh.flush(); eeg_fh.close()
    mkr_fh.flush(); mkr_fh.close()

    print(f"\n[CSV] EEG     → {eeg_path}" if eeg_fh else "\n[no EEG recorded]")
    print(f"[CSV] Markers → {mkr_path}")
    print(f"Score={game.score}  Lines={game.lines}  Level={game.level}")

    # ── End screen ────────────────────────────────────────────────────────────
    end_lines = [
        "GAME OVER" if game.game_over else "TIME UP",
        "",
        f"Score : {game.score:,}",
        f"Lines : {game.lines}",
        f"Level : {game.level}",
        "",
        "Esc to exit",
    ]
    screen.fill(BG)
    for i, line in enumerate(end_lines):
        f   = font_title if i == 0 else font_ui
        col = (220, 50, 50) if (i == 0 and game.game_over) else TEXT_C
        r   = f.render(line, True, col)
        screen.blit(r, (WIN_W // 2 - r.get_width() // 2, 120 + i * 42))
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                waiting = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                waiting = False
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run(parse_args())
