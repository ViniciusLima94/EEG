#!/usr/bin/env python3
"""
Offline evaluation: replay a trained model on a recorded CSV trial.

Loads the trial with the same preprocessing used during training (bandpass,
CAR, decimate, trial z-score), runs sliding-window inference with the saved
W matrix, and reports per-press detection results.

Usage:
    python scripts/evaluate_trial.py -s 9 -t spt -n 4
    python scripts/evaluate_trial.py -s 0 -t gng -n 2 --model notebooks/models/lgb_demo.pkl
    python scripts/evaluate_trial.py -s 9 -t spt -n 4 --plot
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import load_trial, make_horizon_labels, extract_features  # type: ignore[import]
from src.postproc import event_detection_metrics, persistence_gate           # type: ignore[import]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Offline model evaluation on a CSV trial")
    ap.add_argument("-s", "--subject", type=int, required=True,  help="Subject index")
    ap.add_argument("-t", "--tache",              required=True,  help="Task name (e.g. spt, gng)")
    ap.add_argument("-n", "--trial",  type=int,   required=True,  help="Trial number")
    ap.add_argument("-ch", "--channels",  type=str,   required=False, default=None, help="Trial number")
    ap.add_argument("--model",  default="notebooks/models/lgb_demo.pkl",
                    help="Path to trained model pkl")
    ap.add_argument("--data",   default="data",
                    help="Data directory (relative to repo root or absolute)")
    ap.add_argument("--plot",   action="store_true", help="Show score plot")
    ap.add_argument("--despike", type=float, default=0.0,
                    help="MAD sigma threshold for spike removal before filtering (0=off, try 5.0)")
    ap.add_argument("--skip", type=float, default=0.0,
                    help="Seconds to discard from trial start after filtering (handles settling/transients)")
    args = ap.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    pkl = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    with open(pkl, "rb") as fh:
        m = pickle.load(fh)

    model      = m["model"]
    W          = m["W"].astype(np.float64)
    USE_EA     = bool(m.get("USE_EA", True))
    T_OPT      = int(m["T_OPT"])
    FS         = int(m["FS"])
    DECIM      = int(m["DECIM"])
    FS_EFF     = int(m["FS_EFF"])
    LP         = float(m["LP"])
    HP         = float(m["HP"])
    HORIZON    = int(m["HORIZON"])
    SMOOTH_WIN = int(m["SMOOTH_WIN"])
    thresh     = float(m["thresh"])
    REFRACTORY = int(m.get("REFRACTORY", int(0.5 * FS_EFF)))
    PERSIST_K  = int(m.get("PERSIST_K",  1))
    MF_TEMPLATE = m.get("mf_template", None)   # (T_OPT, N_CH) or None

    # ── Load trial ─────────────────────────────────────────────────────────────

    if args.channels is None:
        if args.tache == "way":
            CHANNELS = "Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10".split(",")
        else:
            CHANNELS = [f"ch{ch}" for ch in range(1, 17)]
    else:
        CHANNELS = args.channels.split(",")

    data_root = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    data_path = data_root / f"subject{args.subject}"
    df, cols = load_trial(
        subject=args.subject,
        tache=args.tache,
        trial=args.trial,
        data_path=str(data_path),
        eeg_cols=None,          # no validation — allow any channel layout
        fs=FS,
        lp=LP,
        hp=HP,
        apply_filter=True,
        car=True,
        trial_zscore=True,
        clip_percentile=1.0,
        decimate=DECIM,
        despike_sigma=args.despike,
        skip_seconds=args.skip,
        channels=CHANNELS
    )

    X       = df[cols].values.astype(np.float64)         # (N, C)
    y_mom   = df["button"].values.astype(np.int64)        # (N,) momentary press labels
    n_press = int(y_mom.sum())
    N, C    = X.shape

    print(f"\n── Subject {args.subject} | tache={args.tache} | trial={args.trial} ──────")
    print(f"   Model : {pkl.name}  (thresh={thresh:.4f}  USE_EA={USE_EA})")
    print(f"   FS_EFF={FS_EFF}  T_OPT={T_OPT} ({T_OPT/FS_EFF*1000:.0f} ms)  "
          f"HORIZON={HORIZON} ({HORIZON/FS_EFF*1000:.0f} ms)  refractory={REFRACTORY/FS_EFF*1000:.0f} ms")
    print(f"   {N} samples ({N/FS_EFF:.1f}s)  {C} channels  {n_press} presses\n")

    if n_press == 0:
        print("[warn] No button presses found in this trial.")

    y_h = make_horizon_labels(y_mom, HORIZON)

    # ── Build windows (no per-window z-score — matches training) ──────────────
    # load_trial already applies trial-level z-score; adding a second per-window
    # z-score forces mean≈0 / std≈1 on every window, zeroing out the mean,
    # std, and RMS temporal features that encode the MRCP amplitude ramp.
    wins, win_y = [], []
    for i in range(T_OPT, N):
        wins.append(X[i - T_OPT : i].astype(np.float32))
        win_y.append(int(y_h[i]))

    X_win = np.array(wins)     # (M, T, C)

    # ── Euclidean Alignment (only if model was trained with it) ───────────────
    if USE_EA:
        X_ea = np.nan_to_num(X_win @ W.T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        X_ea = X_win.astype(np.float32)

    # ── Feature extraction ─────────────────────────────────────────────────────
    print("Extracting features…", end="", flush=True)
    if MF_TEMPLATE is not None:
        t_norm = (MF_TEMPLATE - MF_TEMPLATE.mean(0)) / (MF_TEMPLATE.std(0) + 1e-8)
        def _extract(w):
            base = extract_features(w, FS_EFF)
            mu = w.mean(0); sd = w.std(0) + 1e-8
            mf = ((w - mu) / sd * t_norm).mean(0).astype(np.float32)
            return np.concatenate([base, mf])
    else:
        _extract = lambda w: extract_features(w, FS_EFF)
    feats = np.array([_extract(w) for w in X_ea])
    print(f" {len(feats)} windows  ({feats.shape[1]} features)", flush=True)

    # ── Inference ──────────────────────────────────────────────────────────────
    raw_scores = model.predict_proba(feats)[:, 1]  # type: ignore[index]

    # Align scores back to original timeline (first T_OPT samples have no score)
    scores_full = np.full(N, 0.0)
    scores_full[T_OPT:] = raw_scores

    # Smooth (causal rolling mean, same as online)
    scores_smooth = np.convolve(scores_full, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")

    # Persistence gate: rolling minimum over PERSIST_K samples (matches training eval)
    scores_gated = persistence_gate(scores_smooth, PERSIST_K)

    # ── Event detection metrics ────────────────────────────────────────────────
    metrics = event_detection_metrics(
        probs=scores_gated,
        y_momentary=y_mom,
        threshold=thresh,
        horizon=HORIZON,
        fs_eff=FS_EFF,
        refractory=REFRACTORY,
    )

    # ── Per-press table ────────────────────────────────────────────────────────
    press_onsets = np.where(y_mom > 0)[0]

    print(f"\n{'Press':>5}  {'Onset (s)':>10}  {'Status':>8}  {'Latency (ms)':>14}")
    print("─" * 46)
    for i, onset in enumerate(press_onsets):
        win_start = max(0, onset - HORIZON)
        win_probs = scores_gated[win_start:onset]
        onset_s   = onset / FS_EFF
        if len(win_probs) > 0 and win_probs.max() >= thresh:
            first_det     = win_start + int(np.argmax(win_probs >= thresh))
            latency_ms    = (onset - first_det) / FS_EFF * 1000.0
            print(f"  {i+1:>3}  {onset_s:>10.3f}  {'✓ det':>8}  {latency_ms:>12.0f}")
        else:
            print(f"  {i+1:>3}  {onset_s:>10.3f}  {'MISSED':>8}  {'—':>14}")
    print("─" * 46)

    # False alarm onsets for display
    preds = (scores_gated >= thresh).astype(np.int8)
    protected = np.zeros(N, dtype=bool)
    for onset in press_onsets:
        protected[max(0, onset - HORIZON) : onset + 1] = True
    fa_onsets_s = [
        i / FS_EFF
        for i in range(1, N)
        if preds[i] == 1 and preds[i - 1] == 0 and not protected[i]
    ]

    evts    = metrics["n_events"]
    det     = metrics["n_detected"]
    fa_min  = metrics["fa_per_minute"]
    lat_ms  = metrics["mean_latency_ms"]
    print(f"\n  EvtSens = {det}/{evts} ({det/max(evts,1):.0%})   "
          f"FA/min = {fa_min:.1f}   "
          f"Mean latency = {lat_ms:.0f} ms")
    if fa_onsets_s:
        print(f"  FA times (s): {[f'{t:.1f}' for t in fa_onsets_s[:10]]}"
              + ("…" if len(fa_onsets_s) > 10 else ""))
    print()

    # ── Optional plot ──────────────────────────────────────────────────────────
    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        DARK, CARD, EDGE         = "#0a0e17", "#111827", "#1f2937"
        TEAL, CORAL, GOLD, WHITE = "#00e5cc", "#ff4f5e", "#ffc947", "#f0f4ff"

        t_axis = np.arange(N) / FS_EFF

        fig, (ax_p, ax_d) = plt.subplots(
            2, 1, figsize=(16, 6), facecolor=DARK,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            sharex=True,
        )

        # ── Top: probability curve ─────────────────────────────────────────────
        ax_p.set_facecolor(CARD)
        for sp in ax_p.spines.values():
            sp.set_color(EDGE)
        ax_p.tick_params(colors=WHITE, labelsize=8, labelbottom=False)
        ax_p.set_xlim(t_axis[0], t_axis[-1])
        ax_p.set_ylim(-0.05, 1.10)
        ax_p.set_ylabel("P(press)", color=WHITE, fontsize=9)

        for onset in press_onsets:
            pt = onset / FS_EFF
            ax_p.axvspan(max(0.0, pt - HORIZON / FS_EFF), pt,
                         alpha=0.12, color=GOLD, zorder=1)
            ax_p.axvline(pt, color=WHITE, lw=0.8, alpha=0.5, zorder=4)
        for t_fa in fa_onsets_s:
            ax_p.axvline(t_fa, color=CORAL, lw=0.8, alpha=0.5, zorder=3)
        ax_p.axhline(thresh, color=CORAL, lw=1.4, ls="--", alpha=0.9, zorder=3)
        ax_p.plot(t_axis, scores_gated, color=TEAL, lw=1.4, zorder=5)

        leg_handles = [
            mpatches.Patch(color=TEAL,  label="P(press)"),
            Line2D([0], [0], color=CORAL, ls="--", lw=1.4,
                   label=f"Threshold ({thresh:.3f})"),
            mpatches.Patch(color=GOLD, alpha=0.35,
                   label=f"Horizon (−{HORIZON/FS_EFF*1000:.0f} ms)"),
            Line2D([0], [0], color=WHITE, lw=0.9, alpha=0.6,
                   label="Button press"),
            Line2D([0], [0], color=CORAL, lw=0.9, alpha=0.6,
                   label="False alarm"),
        ]
        ax_p.legend(handles=leg_handles, loc="upper left",
                    facecolor=CARD, labelcolor=WHITE, edgecolor=EDGE,
                    fontsize=7, ncol=5)

        # ── Bottom: binary detections ──────────────────────────────────────────
        ax_d.set_facecolor(CARD)
        for sp in ax_d.spines.values():
            sp.set_color(EDGE)
        ax_d.tick_params(colors=WHITE, labelsize=8)
        ax_d.set_xlim(t_axis[0], t_axis[-1])
        ax_d.set_ylim(-0.15, 1.35)
        ax_d.set_yticks([0, 1])
        ax_d.set_yticklabels(["0", "1"], color=WHITE, fontsize=7)
        ax_d.set_ylabel("Detection", color=WHITE, fontsize=9)
        ax_d.set_xlabel("Time (s)", color=WHITE, fontsize=9)

        # Binary signal as filled step plot
        ax_d.fill_between(t_axis, preds, step="post",
                          color=TEAL, alpha=0.35, zorder=2)
        ax_d.step(t_axis, preds, color=TEAL, lw=0.8,
                  where="post", zorder=3)

        # Detected presses: teal triangle up at press onset
        # Missed presses:   coral triangle down at press onset
        for onset in press_onsets:
            pt = onset / FS_EFF
            win_probs = scores_gated[max(0, onset - HORIZON) : onset]
            detected  = len(win_probs) > 0 and win_probs.max() >= thresh
            if detected:
                ax_d.plot(pt, 1.2, marker="^", color=TEAL,
                          ms=7, zorder=5, clip_on=False)
            else:
                ax_d.plot(pt, 1.2, marker="v", color=CORAL,
                          ms=7, zorder=5, clip_on=False)

        # False alarms: coral X below the axis
        for t_fa in fa_onsets_s:
            ax_d.plot(t_fa, -0.1, marker="x", color=CORAL,
                      ms=6, mew=1.5, zorder=5, clip_on=False)

        det_handles = [
            Line2D([0], [0], color=TEAL, lw=2, alpha=0.5,  label="Active detection"),
            Line2D([0], [0], marker="^", color=TEAL,  ls="none", ms=7, label="Press detected"),
            Line2D([0], [0], marker="v", color=CORAL, ls="none", ms=7, label="Press missed"),
            Line2D([0], [0], marker="x", color=CORAL, ls="none", ms=7,
                   mew=1.5, label="False alarm"),
        ]
        ax_d.legend(handles=det_handles, loc="upper left",
                    facecolor=CARD, labelcolor=WHITE, edgecolor=EDGE,
                    fontsize=7, ncol=4)

        fig.suptitle(
            f"Subject {args.subject} · {args.tache} · trial {args.trial}   "
            f"EvtSens {det}/{evts} ({det/max(evts,1):.0%})   "
            f"FA/min {fa_min:.1f}   Lat {lat_ms:.0f} ms",
            color=WHITE, fontsize=10, fontweight="bold",
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
