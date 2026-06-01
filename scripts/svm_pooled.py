"""
Pooled SVM RBF + CAR — all subjects.
Trains on trials 0-7 pooled across subjects (K=10000 balanced windows/subject),
tests on each subject's held-out trials 8-9.
SVM RBF is O(n²) so the final model is capped at SVM_FINAL_MAX samples drawn
from the full pool; Optuna search is also capped per trial for speed.
"""
import warnings, sys, glob
warnings.filterwarnings('ignore')

import numpy as np
import optuna
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')
from src.preprocessing import build_windows, load_trial, make_horizon_labels
from src.postproc import event_detection_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
TACHE           = 'spt'
FS, DECIM       = 250, 4
FS_EFF          = FS // DECIM
LP, HP          = 0.5, 8.0
HORIZON         = int(1.0 * FS_EFF)    # 1000 ms
TARGET          = 'button'
T_MAX           = int(1.5 * FS_EFF)    # 1500 ms max window
K_PER_SUBJECT    = 10_000              # balanced windows sampled per subject
OPTUNA_TRAIN_MAX = 15_000             # cap per Optuna trial (speed)
SVM_FINAL_MAX    = 30_000             # cap for final model (RBF kernel is O(n²))
N_OPTUNA         = 20
T_MIN            = int(0.5 * FS_EFF)  # 500 ms floor
MIN_SPEC        = 0.80
SMOOTH_WIN      = int(0.5 * FS_EFF)    # 500 ms smoothing
SUBJECTS        = list(range(10))

# ── Helpers ────────────────────────────────────────────────────────────────────
def available_trials(subj):
    files = glob.glob(f'../data/subject{subj}/subject_{subj}_tache_spt_trial_*.csv')
    return sorted(int(f.split('trial_')[1].replace('.csv', '')) for f in files)

def subsample_balanced(X, y, n_total, seed=42):
    rng = np.random.default_rng(seed)
    n_per = n_total // 2
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    pi = rng.choice(pos, min(n_per, len(pos)), replace=False)
    ni = rng.choice(neg, min(n_per, len(neg)), replace=False)
    idx = np.sort(np.concatenate([pi, ni]))
    return X[idx], y[idx]

def smooth_scores(scores, window):
    import pandas as pd
    return pd.Series(scores).rolling(window, min_periods=1).mean().values

def threshold_at_spec(fpr, tpr, thresholds, min_spec):
    spec = 1 - fpr
    mask = spec >= min_spec
    if not mask.any():
        return thresholds[np.argmax(spec)]
    return thresholds[np.where(mask)[0][np.argmax(tpr[mask])]]

def sens_at_spec(fpr, tpr, min_spec):
    mask = (1 - fpr) >= min_spec
    if mask.any():
        return float(tpr[mask][np.argmax(tpr[mask])])
    return float(tpr[np.argmax(tpr - fpr)])

# ── Phase 1: load & pool training data ─────────────────────────────────────────
print("Loading training data from all subjects...")
N_CH = None
pool_X   = []   # (T_MAX, N_CH) windows
pool_y   = []
test_data = {}  # subj -> dict

for subj in SUBJECTS:
    data_path = f'../data/subject{subj}/'
    all_avail   = available_trials(subj)
    TRAIN_TRIALS = [t for t in all_avail if t < 8]
    TEST_TRIALS  = [t for t in all_avail if t >= 8]

    if len(TRAIN_TRIALS) < 2:
        print(f'  Subject {subj}: skipped (< 2 train trials)')
        continue

    EEG_COLS = None
    raw = {}
    for t in TRAIN_TRIALS + TEST_TRIALS:
        df, cols = load_trial(subj, TACHE, t, data_path,
                              eeg_cols=EEG_COLS, lp=LP, hp=HP,
                              decimate=DECIM, car=True)
        if EEG_COLS is None:
            EEG_COLS = cols
        if N_CH is None:
            N_CH = len(EEG_COLS)
        X_t  = df[EEG_COLS].values.astype(np.float32)
        y_t  = make_horizon_labels(df[TARGET].values.astype(int), HORIZON)
        y_mom = df[TARGET].values.astype(np.int8)
        raw[t] = (X_t, y_t, y_mom)

    # Collect windows at T_MAX from all train trials
    Xw_parts, yw_parts = [], []
    for t in TRAIN_TRIALS:
        X_t, y_t, _ = raw[t]
        Xw, yw = build_windows(X_t, y_t, T_MAX, per_window_norm=False)
        Xw_parts.append(Xw.reshape(-1, T_MAX, N_CH))
        yw_parts.append(yw)
    Xw_all = np.concatenate(Xw_parts)
    yw_all  = np.concatenate(yw_parts)

    # Balanced subsample of K windows for this subject
    Xw_s, yw_s = subsample_balanced(Xw_all, yw_all, K_PER_SUBJECT, seed=subj * 100 + 7)
    pool_X.append(Xw_s)
    pool_y.append(yw_s)
    n_pos = yw_s.sum(); n_neg = (yw_s == 0).sum()
    print(f'  Subject {subj}: {len(TRAIN_TRIALS)} train, {len(TEST_TRIALS)} test '
          f'→ sampled {len(yw_s)} windows ({n_pos}+/{n_neg}-)')

    # Build test arrays (stored at T_MAX; truncated to T_best later)
    if TEST_TRIALS:
        Xte_p, yte_p, ymom_p, tid_p = [], [], [], []
        for t in TEST_TRIALS:
            X_t, y_t, y_mom_t = raw[t]
            Xw, yw = build_windows(X_t, y_t, T_MAX, per_window_norm=False)
            Xte_p.append(Xw.reshape(-1, T_MAX, N_CH))
            yte_p.append(yw)
            ymom_p.append(y_mom_t[T_MAX:T_MAX + len(yw)])
            tid_p.append(np.full(len(yw), t, dtype=np.int32))
        test_data[subj] = dict(
            X=np.concatenate(Xte_p),
            y=np.concatenate(yte_p),
            y_mom=np.concatenate(ymom_p).astype(np.int8),
            trial_ids=np.concatenate(tid_p),
            test_trials=TEST_TRIALS,
        )

X_pool = np.concatenate(pool_X)   # (N, T_MAX, N_CH)
y_pool = np.concatenate(pool_y)
print(f'\nPooled: {len(y_pool)} windows, '
      f'{y_pool.sum()} pos ({100*y_pool.mean():.1f}%), '
      f'{(y_pool==0).sum()} neg')

# 80/20 stratified split for Optuna validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_pool, y_pool, test_size=0.20, random_state=42, stratify=y_pool)
print(f'Optuna split: {len(y_tr)} train / {len(y_val)} val')

def flat(X3d, T):
    return X3d[:, :T, :].reshape(len(X3d), -1).astype(np.float64)

# ── Phase 2: Optuna hyperparameter search ──────────────────────────────────────
def objective(trial):
    T     = trial.suggest_int('T', T_MIN, T_MAX, log=True)
    C     = trial.suggest_float('C', 0.01, 1000.0, log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1.0, log=True)

    Xtr_s, ytr_s = subsample_balanced(X_tr, y_tr, OPTUNA_TRAIN_MAX,
                                       seed=trial.number)
    pipe = Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=C, kernel='rbf', gamma=gamma,
                                 probability=False, class_weight='balanced',
                                 random_state=42))])
    pipe.fit(flat(Xtr_s, T), ytr_s)
    sc = pipe.decision_function(flat(X_val, T))
    return float(roc_auc_score(y_val, sc))

print(f'\nOptuna search ({N_OPTUNA} trials)...')
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=True)

T_best = study.best_params['T']
C_best = study.best_params['C']
g_best = study.best_params['gamma']
print(f'Optuna best: AUC={study.best_value:.4f}  '
      f'T={T_best} ({T_best/FS_EFF*1000:.0f} ms)  '
      f'C={C_best:.3f}  gamma={g_best:.2e}')

# ── Phase 3: train final model (balanced subsample from full pool) ─────────────
X_fin, y_fin = subsample_balanced(X_pool, y_pool, SVM_FINAL_MAX, seed=0)
print(f'\nTraining final model on {len(y_fin)}/{len(y_pool)} pooled windows '
      f'(capped at {SVM_FINAL_MAX} for RBF tractability)...')
pipe_final = Pipeline([('sc', StandardScaler()),
                       ('svc', SVC(C=C_best, kernel='rbf', gamma=g_best,
                                   probability=False, class_weight='balanced',
                                   random_state=42, cache_size=2000))])
pipe_final.fit(flat(X_fin, T_best), y_fin)
print('Done.')

# Validation AUC on held-out 20%
val_sc = pipe_final.decision_function(flat(X_val, T_best))
val_auc = roc_auc_score(y_val, val_sc)
print(f'Val AUC (pooled model, 20% holdout): {val_auc:.4f}')

# ── Phase 4: per-subject test evaluation ───────────────────────────────────────
print('\n' + '=' * 90)
print('  POOLED MODEL — PER-SUBJECT TEST RESULTS (trials 8-9)')
print('=' * 90)

results = []
for subj in sorted(test_data.keys()):
    d = test_data[subj]
    X_te   = flat(d['X'], T_best)
    sc     = pipe_final.decision_function(X_te)

    # Smooth per trial
    sc_s = np.zeros_like(sc)
    for t in d['test_trials']:
        m = d['trial_ids'] == t
        if m.any():
            sc_s[m] = smooth_scores(sc[m], SMOOTH_WIN)

    auc  = float(roc_auc_score(d['y'], sc_s))
    fpr, tpr, thr = roc_curve(d['y'], sc_s)
    thresh = threshold_at_spec(fpr, tpr, thr, MIN_SPEC)
    sens   = sens_at_spec(fpr, tpr, MIN_SPEC)

    evt = event_detection_metrics(
        sc_s, d['y_mom'][:len(sc_s)],
        threshold=thresh, horizon=HORIZON, fs_eff=FS_EFF)

    print(f'  Subject {subj:2d}: AUC={auc:.4f}  Sens={sens:.3f}  '
          f'EvtSens={evt["event_sensitivity"]:.3f}  FA/min={evt["fa_per_minute"]:.1f}')
    results.append(dict(
        subject=subj,
        test_auc=round(auc, 4),
        test_sens=round(sens, 3),
        test_evt_sens=round(evt['event_sensitivity'], 3),
        test_fa_min=round(evt['fa_per_minute'], 1),
    ))

print('-' * 90)
print(f'  Mean    : AUC={np.mean([r["test_auc"] for r in results]):.4f}  '
      f'Sens={np.mean([r["test_sens"] for r in results]):.3f}  '
      f'EvtSens={np.mean([r["test_evt_sens"] for r in results]):.3f}')
print('=' * 90)

print(f'\nConfig: K={K_PER_SUBJECT}/subj, pool={len(y_pool)}, final_fit={len(y_fin)}, '
      f'T={T_best}({T_best/FS_EFF*1000:.0f}ms), C={C_best:.3f}, gamma={g_best:.2e}')
