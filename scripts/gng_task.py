"""
Go/No-Go motor intention task with PyGame + LSL marker streaming.

Paradigm
--------
  ITI (fixation cross)
    └─ [optional] S1 warning cue  ──── S1–S2 interval (ISI) ──── S2 Go/NoGo stimulus
         └─ subject prepares & executes movement within response_window
            └─ feedback (optional) ──── next ITI

Connects to the PiEEG LSL inlet so the experiment won't start unless EEG
is streaming.  Markers are sent on a 'GNG_Markers' LSL outlet — both streams
share the LSL clock, so timestamps align sample-accurately.

Button presses are read from the PiEEG hardware button channel (--button-channel <index>, default 17).

LSL marker codes
----------------
  5  = S1 warning cue onset
  10 = Go stimulus (S2) onset
  20 = No-Go stimulus (S2) onset
  30 = Button / movement detected
  11 = Hit        (Go + moved in time)
  12 = Miss       (Go + no movement)
  21 = Correct rejection  (No-Go + no movement)
  22 = False alarm        (No-Go + moved)
  99 = Experiment end

Usage
-----
  python scripts/gng_task.py
  python scripts/gng_task.py --n_trials 100 --go_ratio 0.70 \\
      --iti_min 1.5 --iti_max 3.0 \\
      --response_window 2.0 \\
      --s1 --isi 1.5
  python scripts/gng_task.py --button-channel 8
  python scripts/gng_task.py --no-eeg-check   # test without PiEEG
"""

import argparse
import random
import sys
import time

import pygame
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    stream="PiEEG",
    n_trials=80,
    go_ratio=0.75,
    iti_min=1.5,           # longer ITI suits motor tasks
    iti_max=3.0,
    response_window=2.0,   # time after S2 onset during which movement is accepted
    # S1 warning cue
    use_s1=False,
    isi=1.5,               # S1 → S2 interval (s); only used when --s1 is set
    s1_duration=0.2,       # how long S1 is displayed (s)
    # feedback
    feedback=True,
    feedback_duration=0.5,
    # display
    fullscreen=False,
    win_width=800,
    win_height=600,
    # hardware button
    button_channel=17,
    button_threshold=0.5,
)

# LSL marker codes
MARKER_S1 = 5
MARKER_GO_ONSET = 10
MARKER_NOGO_ONSET = 20
MARKER_MOVEMENT = 30
MARKER_HIT = 11
MARKER_MISS = 12
MARKER_CR = 21
MARKER_FA = 22
MARKER_END = 99

# Colors
BG = (30, 30, 30)
FIX_C = (200, 200, 200)
S1_C = (180, 180, 60)       # yellow warning cue
GO_C = (50, 200, 80)        # green  → move
NOGO_C = (200, 50, 50)      # red    → withhold
TEXT_C = (220, 220, 220)
DIM_C = (120, 120, 120)
HIT_C = (50, 200, 80)
MISS_C = (200, 100, 50)
FA_C = (200, 50, 50)
CR_C = (100, 150, 220)
WARN_C = (220, 180, 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Go/No-Go motor intention task with LSL markers"
    )
    p.add_argument("--stream", default=DEFAULTS["stream"])
    p.add_argument("--n_trials", type=int, default=DEFAULTS["n_trials"])
    p.add_argument("--go_ratio", type=float, default=DEFAULTS["go_ratio"],
                   help="Fraction of trials that are Go (0–1)")
    p.add_argument("--iti_min", type=float, default=DEFAULTS["iti_min"],
                   help="Min ITI in seconds")
    p.add_argument("--iti_max", type=float, default=DEFAULTS["iti_max"],
                   help="Max ITI in seconds")
    p.add_argument("--response_window", type=float, default=DEFAULTS["response_window"],
                   help="Time after S2 onset within which movement is accepted (s)")
    # S1 warning cue
    p.add_argument("--s1", action="store_true", default=DEFAULTS["use_s1"],
                   help="Show an S1 warning cue before the Go/NoGo stimulus")
    p.add_argument("--isi", type=float, default=DEFAULTS["isi"],
                   help="S1–S2 interval in seconds (only used with --s1)")
    p.add_argument("--s1_duration", type=float, default=DEFAULTS["s1_duration"],
                   help="S1 display duration in seconds")
    # feedback
    p.add_argument("--feedback", action=argparse.BooleanOptionalAction,
                   default=DEFAULTS["feedback"])
    p.add_argument("--feedback_duration", type=float, default=DEFAULTS["feedback_duration"])
    # display
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--win_width", type=int, default=DEFAULTS["win_width"])
    p.add_argument("--win_height", type=int, default=DEFAULTS["win_height"])
    # misc
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--button-channel", type=int, default=DEFAULTS["button_channel"],
                   dest="button_channel",
                   help="PiEEG channel index (0-based) for HW button (default: 17); -1 = no button")
    p.add_argument("--button-threshold", type=float, default=DEFAULTS["button_threshold"],
                   dest="button_threshold")
    p.add_argument("--no-eeg-check", action="store_true",
                   help="Skip PiEEG stream check (keyboard-only test mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LSL helpers
# ---------------------------------------------------------------------------

def connect_eeg_inlet(stream_name: str, timeout: float = 10.0) -> StreamInlet | None:
    print(f"[LSL] Searching for '{stream_name}' …", flush=True)
    streams = resolve_byprop("name", stream_name, timeout=timeout)
    if not streams:
        return None
    inlet = StreamInlet(streams[0], max_buflen=1)
    info = streams[0]
    print(f"[LSL] Connected  ({info.channel_count()} ch, {info.nominal_srate():.0f} Hz)",
          flush=True)
    return inlet


def make_marker_outlet(n_trials: int) -> StreamOutlet:
    info = StreamInfo(
        name="GNG_Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="int32",
        source_id="gng_task_001",
    )
    info.desc().append_child_value("n_trials", str(n_trials))
    return StreamOutlet(info)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_fixation(surface, cx, cy, size=18, color=FIX_C):
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 2)
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 2)


def draw_circle(surface, color, cx, cy, radius=90, outline=None):
    pygame.draw.circle(surface, color, (cx, cy), radius)
    if outline:
        pygame.draw.circle(surface, outline, (cx, cy), radius, 3)


def draw_text(surface, text, font, cx, cy, color=TEXT_C):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=(cx, cy))
    surface.blit(rendered, rect)


def sleep_precise(duration: float):
    deadline = time.perf_counter() + duration
    coarse = duration - 0.003
    if coarse > 0:
        time.sleep(coarse)
    while time.perf_counter() < deadline:
        pass


# ---------------------------------------------------------------------------
# Trial sequence
# ---------------------------------------------------------------------------

def generate_trials(n_trials: int, go_ratio: float, seed=None) -> list[str]:
    rng = random.Random(seed)
    n_go = round(n_trials * go_ratio)
    seq = ["go"] * n_go + ["nogo"] * (n_trials - n_go)
    rng.shuffle(seq)
    return seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(args):
    # ── EEG inlet ────────────────────────────────────────────────────────
    eeg_inlet = None
    if not args.no_eeg_check:
        eeg_inlet = connect_eeg_inlet(args.stream)
        if eeg_inlet is None:
            print(f"[LSL] WARNING: '{args.stream}' not found.")
            ans = input("Continue without EEG? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit(0)
    else:
        print("[LSL] EEG check skipped.")

    use_hw_button = (args.button_channel >= 0) and (eeg_inlet is not None)
    if args.button_channel >= 0 and not use_hw_button:
        print("[LSL] WARNING: --button-channel set but no EEG inlet — button responses disabled.")
    if use_hw_button:
        n_ch = eeg_inlet.info().channel_count()
        if args.button_channel >= n_ch:
            print(f"[LSL] ERROR: --button-channel {args.button_channel} is out of range "
                  f"(stream has {n_ch} channels, indices 0–{n_ch - 1}).")
            sys.exit(1)
        print(f"[LSL] HW button on channel {args.button_channel} "
              f"(threshold {args.button_threshold}, stream has {n_ch} ch)")

    # ── Marker outlet ─────────────────────────────────────────────────────
    outlet = make_marker_outlet(args.n_trials)
    print("[LSL] GNG_Markers outlet open.\n")

    # ── PyGame ───────────────────────────────────────────────────────────
    pygame.init()
    flags = pygame.FULLSCREEN if args.fullscreen else 0
    screen = pygame.display.set_mode((args.win_width, args.win_height), flags)
    pygame.display.set_caption("Go / No-Go — motor intention")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Arial", 56, bold=True)
    font_fb    = pygame.font.SysFont("Arial", 36)
    font_ui    = pygame.font.SysFont("Arial", 22)
    font_small = pygame.font.SysFont("Arial", 18)

    W, H = args.win_width, args.win_height
    cx, cy = W // 2, H // 2

    btn_label = "HW button"

    # ── Instructions ─────────────────────────────────────────────────────
    s1_note = (f"A yellow circle (S1) will appear {args.isi:.1f} s before "
               "the imperative stimulus." if args.s1 else "")
    instruct_lines = [
        "Go / No-Go — Motor Intention Task",
        "",
        f"GREEN circle  →  press {btn_label}",
        "RED circle    →  do NOT press anything",
        "",
        s1_note,
        f"You have {args.response_window:.1f} s to move after the stimulus appears.",
        "",
        "Press SPACE to begin",
    ]
    screen.fill(BG)
    for i, line in enumerate(instruct_lines):
        f = font_title if i == 0 else font_ui
        draw_text(screen, line, f, cx, 80 + i * 40, TEXT_C)
    if eeg_inlet is None:
        draw_text(screen, "WARNING: EEG not connected", font_ui, cx, H - 40, WARN_C)
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

    # ── Helpers used inside the trial loop ───────────────────────────────
    prev_hw = False

    def pump_events() -> None:
        """Drain the pygame event queue. Quits immediately on QUIT or ESC."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                outlet.push_sample([MARKER_END])
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                outlet.push_sample([MARKER_END])
                pygame.quit(); sys.exit()

    def poll_hw_edge() -> bool:
        nonlocal prev_hw
        if not use_hw_button:
            return False
        fired = False
        while True:
            sample, _ = eeg_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            current = sample[args.button_channel] > args.button_threshold
            if current and not prev_hw:
                fired = True
            prev_hw = current
        return fired

    def drain_hw():
        nonlocal prev_hw
        if not use_hw_button:
            return
        while True:
            sample, _ = eeg_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            prev_hw = sample[args.button_channel] > args.button_threshold

    def wait_period(duration: float, draw_fn=None):
        """Block for duration seconds, draining events."""
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            pump_events()
            drain_hw()
            if draw_fn:
                draw_fn()
            pygame.display.flip()
            clock.tick(120)

    # ── Trial loop ────────────────────────────────────────────────────────
    trials = generate_trials(args.n_trials, args.go_ratio, args.seed)
    results = []
    rng = random.Random(args.seed)

    for t_idx, trial_type in enumerate(trials):
        trial_num = t_idx + 1
        counter_str = f"{trial_num} / {args.n_trials}"

        # ── ITI: fixation cross ──────────────────────────────────────────
        iti = rng.uniform(args.iti_min, args.iti_max)
        prev_hw = False  # reset edge detector
        drain_hw()

        def draw_iti():
            screen.fill(BG)
            draw_fixation(screen, cx, cy)
            draw_text(screen, counter_str, font_small, W - 60, 20, DIM_C)

        wait_period(iti, draw_iti)

        # ── S1 warning cue (optional) ────────────────────────────────────
        if args.s1:
            # S1 on
            outlet.push_sample([MARKER_S1])

            def draw_s1():
                screen.fill(BG)
                draw_circle(screen, S1_C, cx, cy, radius=90)
                draw_text(screen, counter_str, font_small, W - 60, 20, DIM_C)

            wait_period(args.s1_duration, draw_s1)

            # S1 off, blank until S2
            remaining_isi = args.isi - args.s1_duration
            if remaining_isi > 0:
                wait_period(remaining_isi, draw_iti)

        # ── S2 stimulus onset ─────────────────────────────────────────────
        marker_code = MARKER_GO_ONSET if trial_type == "go" else MARKER_NOGO_ONSET
        stim_color  = GO_C if trial_type == "go" else NOGO_C

        screen.fill(BG)
        draw_circle(screen, stim_color, cx, cy)
        draw_text(screen, counter_str, font_small, W - 60, 20, DIM_C)
        pygame.display.flip()

        outlet.push_sample([marker_code])    # ← LSL marker, precise timestamp
        stim_onset = time.perf_counter()

        # ── Response window: stimulus stays on; wait for movement ─────────
        pressed = False
        press_rt = None
        deadline = stim_onset + args.response_window

        while time.perf_counter() < deadline:
            pump_events()

            if not pressed and poll_hw_edge():
                press_rt = time.perf_counter() - stim_onset
                pressed = True
                outlet.push_sample([MARKER_MOVEMENT])

            if pressed:
                break

            # keep stimulus on screen
            screen.fill(BG)
            draw_circle(screen, stim_color, cx, cy)
            draw_text(screen, counter_str, font_small, W - 60, 20, DIM_C)
            # show elapsed time bar at bottom
            elapsed_frac = (time.perf_counter() - stim_onset) / args.response_window
            bar_w = int((W - 80) * min(elapsed_frac, 1.0))
            pygame.draw.rect(screen, (60, 60, 60), (40, H - 20, W - 80, 8))
            pygame.draw.rect(screen, stim_color,   (40, H - 20, bar_w,  8))
            pygame.display.flip()
            clock.tick(120)

        # ── Outcome marker ────────────────────────────────────────────────
        if trial_type == "go":
            outcome_code  = MARKER_HIT if pressed else MARKER_MISS
            outcome_label = "MOVE" if pressed else "MISS"
            outcome_color = HIT_C if pressed else MISS_C
        else:
            outcome_code  = MARKER_FA if pressed else MARKER_CR
            outcome_label = "FALSE ALARM" if pressed else "WITHHELD"
            outcome_color = FA_C if pressed else CR_C

        outlet.push_sample([outcome_code])

        results.append(dict(
            trial=trial_num,
            type=trial_type,
            pressed=pressed,
            rt=round(press_rt * 1000, 1) if press_rt else None,
            outcome=outcome_label,
        ))

        # ── Feedback ──────────────────────────────────────────────────────
        if args.feedback:
            screen.fill(BG)
            rt_str = f"  {press_rt * 1000:.0f} ms" if press_rt else ""
            draw_text(screen, outcome_label + rt_str, font_fb, cx, cy, outcome_color)
            pygame.display.flip()
            sleep_precise(args.feedback_duration)

    # ── End ───────────────────────────────────────────────────────────────
    outlet.push_sample([MARKER_END])

    hits   = sum(1 for r in results if r["outcome"] == "MOVE")
    misses = sum(1 for r in results if r["outcome"] == "MISS")
    fas    = sum(1 for r in results if r["outcome"] == "FALSE ALARM")
    crs    = sum(1 for r in results if r["outcome"] == "WITHHELD")
    rts    = [r["rt"] for r in results if r["rt"] is not None and r["type"] == "go"]
    mean_rt = sum(rts) / len(rts) if rts else 0.0

    print("\n=== Results ===")
    for r in results:
        rt_str = f"{r['rt']} ms" if r["rt"] else "—"
        print(f"  [{r['trial']:3d}] {r['type']:<5s}  {r['outcome']:<12s}  RT: {rt_str}")
    print(f"\nHits={hits}  Misses={misses}  FAs={fas}  CRs={crs}  "
          f"Mean RT={mean_rt:.0f} ms")

    screen.fill(BG)
    for i, line in enumerate([
        "Task complete!",
        f"Hits: {hits}   Misses: {misses}   False Alarms: {fas}   Withheld: {crs}",
        f"Mean movement RT: {mean_rt:.0f} ms",
        "",
        "Press ESC to exit",
    ]):
        f = font_title if i == 0 else font_ui
        draw_text(screen, line, f, cx, cy - 80 + i * 44, TEXT_C)
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
    run_task(parse_args())