"""
LightGBM / XGBoost LOTO evaluation — standalone runner.
Pass --model lgb|xgb, --min-spec, --horizon-ms, --subject, --n-optuna.

Optuna jointly searches window length T (200 ms – 2 s) and model hyperparameters.
"""
import warnings, sys, argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.signal import welch
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

sys.path.insert(0, '..')
from src.preprocessing import build_windows, load_trial, make_horizon_labels
from src.postproc import event_detection_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min-spec',   type=float, default=0.90)
parser.add_argument('--n-optuna',   type=int,   default=50)
parser.add_argument('--subject',    type=int,   default=9)
parser.add_argument('--horizon-ms', type=int,   default=300)
parser.add_argument('--model',         type=str,   default='lgb', choices=['lgb', 'xgb'])
parser.add_argument('--refractory-ms', type=int,   default=500)
args = parser.parse_args()

SUBJECT    = args.subject
MIN_SPEC   = args.min_spec
N_OPTUNA   = args.n_optuna
MODEL      = args.model

# ── Config ────────────────────────────────────────────────────────────────────
TACHE       = 'spt'
ALL_TRIALS  = list(range(10))
TEST_TRIALS = list(range(1, 10))   # trial 0 is calibration — always in train
DATA_PATH   = f'../data/subject{SUBJECT}/'
FS, DECIM   = 250, 4
FS_EFF      = FS // DECIM           # 62 Hz
LP, HP      = 0.5, 8.0
HORIZON     = int(args.horizon_ms / 1000.0 * FS_EFF)
TARGET      = 'button'
T_MAX       = int(2.0 * FS_EFF)    # upper bound for T search (2 s)
T_MIN       = int(0.2 * FS_EFF)    # lower bound for T search (200 ms)
SMOOTH_WIN  = max(1, int(0.1 * FS_EFF))
REFRACTORY  = int(args.refractory_ms / 1000.0 * FS_EFF)
TRAIN_CAP   = 20_000
SEED        = 42

print(f'Subject={SUBJECT}  Model={MODEL.upper()}  MIN_SPEC={MIN_SPEC}  N_OPTUNA={N_OPTUNA}')
print(f'FS_EFF={FS_EFF} Hz  T_MAX={T_MAX} ({T_MAX/FS_EFF:.1f}s)  HORIZON={HORIZON} ({HORIZON/FS_EFF*1000:.0f}ms)')
print(f'T search: {T_MIN} ({T_MIN/FS_EFF*1000:.0f}ms) … {T_MAX} ({T_MAX/FS_EFF*1000:.0f}ms)')
print(f'Refractory: {REFRACTORY} samp ({args.refractory_ms}ms)')

# ── Load ──────────────────────────────────────────────────────────────────────
print('\nLoading trials …')
raw_data, EEG_COLS = {}, None
for t in ALL_TRIALS:
    df, cols = load_trial(SUBJECT, TACHE, t, DATA_PATH,
                          eeg_cols=EEG_COLS, lp=LP, hp=HP,
                          decimate=DECIM, car=True, trial_zscore=True)
    if EEG_COLS is None:
        EEG_COLS = cols
    raw_data[t] = (
        df[EEG_COLS].values.astype(np.float32),
        make_horizon_labels(df[TARGET].values.astype(int), HORIZON),
        df[TARGET].values.astype(np.int8),
    )
N_CH = len(EEG_COLS)

# ── Features ──────────────────────────────────────────────────────────────────
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
    return np.stack([mean_, std_, slope_, min_, argm_, rms_],
                    axis=2).reshape(N, C * 6).astype(np.float32)

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

def features_for_T(T):
    """Slice X_max to last T samples and compute all features. Returns (N, C*9)."""
    X_t = X_max[:, -T:, :]
    return np.concatenate([temporal_features(X_t), freq_features(X_t, FS_EFF)],
                          axis=1).astype(np.float32)

# ── Build windows at T_MAX (features sliced per Optuna trial) ─────────────────
print(f'Building windows at T_MAX={T_MAX} ({T_MAX/FS_EFF:.1f}s) …')
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

X_max   = np.concatenate(_X)   # (M, T_MAX, C) — sliced per Optuna trial
y_all   = np.concatenate(_y)
y_mom   = np.concatenate(_ymom)
tid_all = np.concatenate(_tid)
del _X, _y, _ymom, _tid
print(f'  {len(y_all):,} windows  {int(y_all.sum()):,} pos ({y_all.mean()*100:.1f}%)')
print(f'  X_max shape: {X_max.shape}')

# ── Utilities ─────────────────────────────────────────────────────────────────
def subsample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), min(n, len(y)), replace=False))
    return X[idx], y[idx]

def smooth(scores, window=SMOOTH_WIN):
    if window <= 1:
        return scores.copy()
    return pd.Series(scores).rolling(window, min_periods=1).mean().values.astype(np.float32)

def thresh_at_spec(fpr, tpr, thresholds, min_spec=MIN_SPEC):
    valid = (1 - fpr) >= min_spec
    if valid.any():
        return float(thresholds[np.argmax(tpr[valid])])
    return float(thresholds[np.argmax(tpr - fpr)])

# ── Model factory ─────────────────────────────────────────────────────────────
def make_model(params):
    if MODEL == 'lgb':
        return lgb.LGBMClassifier(
            n_estimators      = params['n_est'],
            learning_rate     = params['lr'],
            max_depth         = params['max_d'],
            num_leaves        = params['n_leaves'],
            min_child_samples = params['min_ch'],
            subsample         = params['sub'],
            colsample_bytree  = params['col'],
            reg_lambda        = params['lam'],
            class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=1,
        )
    else:
        scale = float((y_all == 0).sum()) / max(float((y_all == 1).sum()), 1)
        return xgb.XGBClassifier(
            n_estimators      = params['n_est'],
            learning_rate     = params['lr'],
            max_depth         = params['max_d'],
            subsample         = params['sub'],
            colsample_bytree  = params['col'],
            reg_lambda        = params['lam'],
            scale_pos_weight  = scale,
            random_state=SEED, verbosity=0, n_jobs=1,
            eval_metric='logloss', use_label_encoder=False,
        )

# ── Optuna (joint T + hyperparameter search) ──────────────────────────────────
def objective(trial):
    T = trial.suggest_int('T', T_MIN, T_MAX)
    X_f = features_for_T(T)

    params = dict(
        n_est    = trial.suggest_int('n_est',    100, 600),
        lr       = trial.suggest_float('lr',     0.01, 0.3, log=True),
        max_d    = trial.suggest_int('max_d',    3, 9),
        n_leaves = trial.suggest_int('n_leaves', 15, 127),
        min_ch   = trial.suggest_int('min_ch',   10, 120),
        sub      = trial.suggest_float('sub',    0.5, 1.0),
        col      = trial.suggest_float('col',    0.5, 1.0),
        lam      = trial.suggest_float('lam',    1e-3, 10.0, log=True),
    )
    fold_aucs = []
    for fold, t_fold in enumerate(TEST_TRIALS[::2]):
        te = tid_all == t_fold
        X_tr, y_tr = subsample(X_f[~te], y_all[~te], TRAIN_CAP, seed=trial.number + fold)
        m = make_model(params)
        m.fit(X_tr, y_tr)
        sc = m.predict_proba(X_f[te])[:, 1]
        fold_aucs.append(roc_auc_score(y_all[te], sc))
        trial.report(float(np.mean(fold_aucs)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_aucs))

print(f'\nOptuna search ({N_OPTUNA} trials, joint T + HP) …')
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=True)
BEST  = study.best_params
T_OPT = BEST['T']
print(f'Best #{study.best_trial.number}  AUC={study.best_value:.4f}  '
      f'T={T_OPT} ({T_OPT/FS_EFF*1000:.0f}ms)')

print(f'\nComputing features with T_OPT={T_OPT} …')
X_feat = features_for_T(T_OPT)
print(f'  X_feat shape: {X_feat.shape}')

# ── Full LOTO ─────────────────────────────────────────────────────────────────
print(f'\nLOTO evaluation (MIN_SPEC={MIN_SPEC}) …')
scores       = np.zeros(len(y_all), dtype=np.float32)
fold_results = []

for test_trial in TEST_TRIALS:
    te = tid_all == test_trial
    tr = ~te
    X_tr, y_tr = X_feat[tr], y_all[tr]
    X_te, y_te = X_feat[te], y_all[te]

    X_sub, y_sub = subsample(X_tr, y_tr, TRAIN_CAP, seed=SEED)
    model = make_model(BEST)
    model.fit(X_sub, y_sub)

    sc_tr = model.predict_proba(X_tr)[:, 1]
    sc_te = model.predict_proba(X_te)[:, 1]

    fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, sc_tr)
    thresh = thresh_at_spec(fpr_tr, tpr_tr, thr_tr, MIN_SPEC)

    scores[te] = sc_te.astype(np.float32)
    auc = roc_auc_score(y_te, sc_te)
    ap  = average_precision_score(y_te, sc_te)
    fold_results.append(dict(trial=test_trial, auc=auc, ap=ap, thresh=thresh))
    print(f'  fold {test_trial:2d}  AUC={auc:.4f}  AP={ap:.4f}  thresh={thresh:.4f}')

# Smooth per trial
scores_s = np.zeros_like(scores)
for t in TEST_TRIALS:
    m = tid_all == t
    scores_s[m] = smooth(scores[m])

thresh_final = float(np.mean([r['thresh'] for r in fold_results]))

test_mask = np.isin(tid_all, TEST_TRIALS)
evt = event_detection_metrics(scores_s[test_mask], y_mom[test_mask],
                              threshold=thresh_final, horizon=HORIZON, fs_eff=FS_EFF,
                              refractory=REFRACTORY)

auc_mean = float(np.mean([r['auc'] for r in fold_results]))
ap_mean  = float(np.mean([r['ap']  for r in fold_results]))

print(f'\n{"="*57}')
print(f'  {MODEL.upper()} LOTO  Subject={SUBJECT}  HORIZON={args.horizon_ms}ms  MIN_SPEC={MIN_SPEC}')
print(f'{"="*57}')
print(f'  T_OPT      : {T_OPT} samp ({T_OPT/FS_EFF*1000:.0f} ms)')
print(f'  Refractory : {REFRACTORY} samp ({args.refractory_ms} ms)')
print(f'  LOTO AUC  : {auc_mean:.4f}')
print(f'  LOTO AP   : {ap_mean:.4f}')
print(f'  Threshold : {thresh_final:.4f}')
print(f"  EvtSens   : {evt['event_sensitivity']:.3f}")
print(f"  FA/min    : {evt['fa_per_minute']:.1f}")
print(f"  Latency   : {evt.get('mean_latency_ms', 0):.1f} ms")
print(f"  Detected  : {evt['n_detected']} / {evt['n_events']}")
print(f'{"="*57}')
