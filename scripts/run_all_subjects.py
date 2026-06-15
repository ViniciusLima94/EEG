"""
Multi-subject LOTO evaluation — runs the full LightGBM pipeline for every subject
and reports a summary table.

Pipeline matches eeg_lgb_focused.ipynb:
  bandpass 0.5-8 Hz → CAR → trial z-score → decimate ×4 → HORIZON=1000ms labels
  → Optuna (N_OPTUNA trials, joint T + HP search, AUC objective)
  → LOTO with per-fold EA + MF features
  → persistence gate (PERSIST_K)
  → threshold sweep for event-level operating curve
"""
import warnings, sys, os, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import optuna
from scipy.signal import welch
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import lightgbm as lgb

from src.preprocessing import (
    load_trial, make_horizon_labels, build_windows, euclidean_align,
)
from src.postproc import event_detection_metrics, persistence_gate

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECTS    = list(range(10))
TACHE       = "spt"
ALL_TRIALS  = list(range(10))
TEST_TRIALS = list(range(10))
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "data")

FS, DECIM   = 250, 4
FS_EFF      = FS // DECIM          # 62 Hz
LP, HP      = 0.5, 8.0
HORIZON     = int(1.0 * FS_EFF)    # 1000 ms look-ahead
T_MAX       = int(2.0 * FS_EFF)    # 2 s upper bound
T_MIN       = int(0.2 * FS_EFF)    # 200 ms lower bound
SMOOTH_WIN  = max(1, int(0.1 * FS_EFF))
REFRACTORY  = int(0.5 * FS_EFF)
PERSIST_K   = 8
MIN_SPEC    = 0.90
TRAIN_CAP   = 50_000
N_OPTUNA    = 5
SEED        = 42

# ── Feature functions ─────────────────────────────────────────────────────────
def temporal_features(X3d):
    N, T, C = X3d.shape
    X64 = X3d.astype(np.float64)
    t_c = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    t_var = (t_c ** 2).sum() + 1e-12
    mean_  = X64.mean(axis=1)
    std_   = X64.std(axis=1)
    slope_ = (X64 * t_c[None, :, None]).sum(axis=1) / t_var
    min_   = X64.min(axis=1)
    argm_  = X64.argmin(axis=1) / T
    rms_   = np.sqrt((X64 ** 2).mean(axis=1))
    return np.stack([mean_, std_, slope_, min_, argm_, rms_], axis=2).reshape(N, C * 6).astype(np.float32)

def freq_features(X3d, fs, batch=2000):
    N, T, C = X3d.shape
    nperseg = min(T, 64)
    out = np.zeros((N, C * 3), dtype=np.float32)
    for s in range(0, N, batch):
        xb = X3d[s:s+batch].transpose(0, 2, 1)
        f, p = welch(xb, fs=fs, nperseg=nperseg, axis=-1)
        df = f[1] - f[0]
        dp = p[:, :, (f >= 0.5) & (f <= 4.0)].sum(axis=-1) * df
        tp = p[:, :, (f >= 4.0) & (f <= 8.0)].sum(axis=-1) * df
        ma = f > 0
        pn = p[:, :, ma] / (p[:, :, ma].sum(-1, keepdims=True) + 1e-12)
        se = -(pn * np.log(pn + 1e-12)).sum(axis=-1)
        out[s:s+batch, 0::3] = dp
        out[s:s+batch, 1::3] = tp
        out[s:s+batch, 2::3] = se
    return out

def hjorth_features(X3d):
    d1 = np.diff(X3d, axis=1)
    d2 = np.diff(d1,  axis=1)
    v0 = X3d.var(axis=1) + 1e-12
    v1 = d1.var(axis=1) + 1e-12
    v2 = d2.var(axis=1) + 1e-12
    mob = np.sqrt(v1 / v0)
    cmp = np.sqrt(v2 / v1) / (mob + 1e-12)
    return np.stack([v0, mob, cmp], axis=2).reshape(len(X3d), -1).astype(np.float32)

def mf_features(X3d, template):
    t_norm = (template - template.mean(axis=0)) / (template.std(axis=0) + 1e-8)
    mu = X3d.mean(axis=1, keepdims=True)
    sd = X3d.std(axis=1, keepdims=True) + 1e-8
    return ((X3d - mu) / sd * t_norm[None]).mean(axis=1).astype(np.float32)

def features_for_T(T, X_3d):
    X_t = X_3d[:, -T:, :]
    return np.concatenate([temporal_features(X_t), freq_features(X_t, FS_EFF),
                           hjorth_features(X_t)], axis=1).astype(np.float32)

# ── Utilities ─────────────────────────────────────────────────────────────────
def subsample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), min(n, len(y)), replace=False))
    return X[idx], y[idx]

def smooth(scores):
    if SMOOTH_WIN <= 1:
        return scores.copy()
    return pd.Series(scores).rolling(SMOOTH_WIN, min_periods=1).mean().values.astype(np.float32)

def thresh_at_spec(fpr, tpr, thresholds):
    valid = (1 - fpr) >= MIN_SPEC
    if valid.any():
        return float(thresholds[np.argmax(tpr[valid])])
    return float(thresholds[np.argmax(tpr - fpr)])

def make_model(params):
    return lgb.LGBMClassifier(
        n_estimators=params["n_est"], learning_rate=params["lr"],
        max_depth=params["max_d"], num_leaves=params["n_leaves"],
        min_child_samples=params["min_ch"], subsample=params["sub"],
        colsample_bytree=params["col"], reg_lambda=params["lam"],
        class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1,
    )

def best_at_sens(target, sw):
    cands = [r for r in sw if r["sens"] >= target]
    return max(cands, key=lambda r: r["thresh"]) if cands else None

# ── Per-subject pipeline ──────────────────────────────────────────────────────
def run_subject(subject):
    t_start = time.time()
    data_path = os.path.join(DATA_ROOT, f"subject{subject}")
    print(f"\n{'='*60}")
    print(f"  Subject {subject}")
    print(f"{'='*60}")

    # Load
    raw_data, EEG_COLS = {}, None
    for t in ALL_TRIALS:
        df, cols = load_trial(subject, TACHE, t, data_path, eeg_cols=EEG_COLS,
                              lp=LP, hp=HP, decimate=DECIM, car=True, trial_zscore=True)
        if EEG_COLS is None:
            EEG_COLS = cols
        raw_data[t] = (
            df[EEG_COLS].values.astype(np.float32),
            make_horizon_labels(df["button"].values.astype(int), HORIZON),
            df["button"].values.astype(np.int8),
        )
    N_CH = len(EEG_COLS)

    # Build windows at T_MAX
    _X, _y, _ymom, _tid = [], [], [], []
    for t in ALL_TRIALS:
        X_t, y_t, y_mom_t = raw_data[t]
        Xw, yw = build_windows(X_t, y_t, T_MAX, per_window_norm=False)
        y_mom_w = y_mom_t[T_MAX:][:len(yw)]
        n = min(len(yw), len(y_mom_w))
        _X.append(Xw[:n].reshape(-1, T_MAX, N_CH))
        _y.append(yw[:n])
        _ymom.append(y_mom_w[:n])
        _tid.append(np.full(n, t, dtype=np.int32))
    X_max   = np.concatenate(_X)
    y_all   = np.concatenate(_y)
    y_mom   = np.concatenate(_ymom)
    tid_all = np.concatenate(_tid)
    n_pos = int(y_all.sum())
    print(f"  {len(y_all):,} windows  {n_pos:,} pos ({n_pos/len(y_all)*100:.1f}%)")

    # Optuna
    def objective(trial):
        T = trial.suggest_int("T", T_MIN, T_MAX)
        params = dict(
            n_est    = trial.suggest_int("n_est",    100, 600),
            lr       = trial.suggest_float("lr",     0.01, 0.3, log=True),
            max_d    = trial.suggest_int("max_d",    3, 9),
            n_leaves = trial.suggest_int("n_leaves", 15, 127),
            min_ch   = trial.suggest_int("min_ch",   10, 120),
            sub      = trial.suggest_float("sub",    0.5, 1.0),
            col      = trial.suggest_float("col",    0.5, 1.0),
            lam      = trial.suggest_float("lam",    1e-3, 10.0, log=True),
        )
        fold_aucs = []
        for fold, t_fold in enumerate(TEST_TRIALS[::2]):
            te = tid_all == t_fold
            X_tr_3d = X_max[~te, -T:, :]
            X_al, W  = euclidean_align(X_tr_3d)
            X_f_tr   = features_for_T(T, X_al)
            X_te_3d  = X_max[te, -T:, :]
            X_al_te  = (X_te_3d @ W.T).astype(np.float32)
            X_f_te   = features_for_T(T, X_al_te)
            X_sub, y_sub = subsample(X_f_tr, y_all[~te], TRAIN_CAP, seed=trial.number + fold)
            m = make_model(params)
            m.fit(X_sub, y_sub)
            sc = m.predict_proba(X_f_te)[:, 1]
            fold_aucs.append(roc_auc_score(y_all[te], sc))
            trial.report(float(np.mean(fold_aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    BEST  = study.best_params
    T_OPT = BEST["T"]
    print(f"  Optuna best AUC={study.best_value:.4f}  T_OPT={T_OPT} ({T_OPT/FS_EFF*1000:.0f} ms)")

    # Full LOTO
    scores  = np.zeros(len(y_all), dtype=np.float32)
    fold_results = []

    for test_trial in TEST_TRIALS:
        te = tid_all == test_trial
        tr = ~te
        X_tr_3d = X_max[tr, -T_OPT:, :]
        X_te_3d = X_max[te, -T_OPT:, :]
        X_tr_al, W = euclidean_align(X_tr_3d)
        X_te_al    = (X_te_3d @ W.T).astype(np.float32)

        # MF features
        press_idx = np.where(y_mom[tr] > 0)[0]
        template  = X_tr_al[press_idx].mean(axis=0) if len(press_idx) > 0 else np.zeros((T_OPT, N_CH), dtype=np.float32)
        X_feat_tr = np.concatenate([features_for_T(T_OPT, X_tr_al), mf_features(X_tr_al, template)], axis=1)
        X_feat_te = np.concatenate([features_for_T(T_OPT, X_te_al), mf_features(X_te_al, template)], axis=1)

        X_sub, y_sub = subsample(X_feat_tr, y_all[tr], TRAIN_CAP, seed=SEED)
        model = make_model(BEST)
        model.fit(X_sub, y_sub)

        sc_tr = model.predict_proba(X_feat_tr)[:, 1]
        sc_te = model.predict_proba(X_feat_te)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_all[tr], sc_tr)
        thresh = thresh_at_spec(fpr_tr, tpr_tr, thr_tr)

        scores[te] = sc_te.astype(np.float32)
        auc = roc_auc_score(y_all[te], sc_te)
        ap  = average_precision_score(y_all[te], sc_te)
        fold_results.append(dict(trial=test_trial, auc=auc, ap=ap, thresh=thresh))

    # Smooth per trial
    scores_s = np.zeros_like(scores)
    for t in TEST_TRIALS:
        m = tid_all == t
        scores_s[m] = smooth(scores[m])

    thresh_final = float(np.mean([r["thresh"] for r in fold_results]))
    test_mask = np.isin(tid_all, TEST_TRIALS)

    # Gated scores
    scores_gated = np.zeros_like(scores_s)
    for t in TEST_TRIALS:
        m = tid_all == t
        scores_gated[m] = persistence_gate(scores_s[m], PERSIST_K)

    evt = event_detection_metrics(scores_s[test_mask], y_mom[test_mask],
                                  threshold=thresh_final, horizon=HORIZON,
                                  fs_eff=FS_EFF, refractory=REFRACTORY)
    evt_g = event_detection_metrics(scores_gated[test_mask], y_mom[test_mask],
                                    threshold=thresh_final, horizon=HORIZON,
                                    fs_eff=FS_EFF, refractory=REFRACTORY)

    # Threshold sweep (full dataset)
    scores_test  = scores_s[test_mask]
    scores_g_test = scores_gated[test_mask]
    ymom_test    = y_mom[test_mask]
    thresholds   = np.linspace(float(np.percentile(scores_test, 0.5)),
                               float(np.percentile(scores_test, 99.5)), 300)
    sweep_g = []
    for thr in thresholds:
        e = event_detection_metrics(scores_g_test, ymom_test, threshold=thr,
                                    horizon=HORIZON, fs_eff=FS_EFF, refractory=REFRACTORY)
        sweep_g.append(dict(thresh=thr, sens=e["event_sensitivity"], fa=e["fa_per_minute"],
                            lat=e.get("mean_latency_ms", 0.0)))

    op80 = best_at_sens(0.80, sweep_g)
    op70 = best_at_sens(0.70, sweep_g)

    auc_mean = float(np.mean([r["auc"] for r in fold_results]))
    ap_mean  = float(np.mean([r["ap"]  for r in fold_results]))
    elapsed  = time.time() - t_start

    print(f"  AUC={auc_mean:.4f}  AP={ap_mean:.4f}  thresh={thresh_final:.3f}")
    print(f"  No gate : EvtSens={evt['event_sensitivity']:.3f}  FA/min={evt['fa_per_minute']:.1f}  Lat={evt.get('mean_latency_ms',0):.0f}ms")
    print(f"  + gate  : EvtSens={evt_g['event_sensitivity']:.3f}  FA/min={evt_g['fa_per_minute']:.1f}  Lat={evt_g.get('mean_latency_ms',0):.0f}ms")
    if op80:
        print(f"  Sweep @≥80% sens: FA/min={op80['fa']:.1f}  Lat={op80['lat']:.0f}ms")
    print(f"  Time: {elapsed:.0f}s")

    return dict(
        subject=subject,
        auc=auc_mean, ap=ap_mean,
        T_OPT=T_OPT,
        n_events=evt["n_events"],
        sens_ng=evt["event_sensitivity"],    fa_ng=evt["fa_per_minute"],    lat_ng=evt.get("mean_latency_ms", 0),
        sens_g=evt_g["event_sensitivity"],   fa_g=evt_g["fa_per_minute"],   lat_g=evt_g.get("mean_latency_ms", 0),
        fa_at_80=op80["fa"] if op80 else float("nan"),
        lat_at_80=op80["lat"] if op80 else float("nan"),
        fa_at_70=op70["fa"] if op70 else float("nan"),
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    out_csv = os.path.join(os.path.dirname(__file__), "all_subjects_results.csv")

    for subj in SUBJECTS:
        try:
            row = run_subject(subj)
            results.append(row)
            pd.DataFrame(results).to_csv(out_csv, index=False)  # save after each subject
        except Exception as e:
            print(f"  ERROR subject {subj}: {e}")
            results.append(dict(subject=subj, auc=float("nan")))

    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("SUMMARY — All subjects")
    print(f"{'='*80}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMean:  AUC={df['auc'].mean():.3f}  "
          f"sens_g={df['sens_g'].mean():.3f}  "
          f"fa_g={df['fa_g'].mean():.1f}  "
          f"fa_at_80={df['fa_at_80'].mean():.1f}")
    print(f"Saved → {out_csv}")
