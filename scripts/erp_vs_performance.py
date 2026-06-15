"""
For each subject: find the channel with the biggest MRCP effect
(most negative mean in the -500 to 0 ms pre-press window), extract
its peak amplitude, then scatter against per-subject test AUC and
test event sensitivity.
"""
import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.preprocessing import load_trial

# ── Config ────────────────────────────────────────────────────────────────────
TACHE   = 'spt'
FS      = 250
LP, HP  = 0.5, 8.0
PRE_MS  = 1500
POST_MS = 500
PRE     = int(PRE_MS  / 1000 * FS)
POST    = int(POST_MS / 1000 * FS)
SUBJECTS = list(range(10))
TIME_AX  = np.linspace(-PRE_MS, POST_MS, PRE + POST)

# Per-subject test results from Section 13 (SVM RBF per-subject, trials 8-9)
# Subject 7 has no test trials; subject 8: AUC=0.84 but EvtSens=0 (threshold issue)
TEST_AUC = {0: 0.5337, 1: 0.6585, 2: 0.3682, 3: 0.5626, 4: 0.7099,
            5: 0.5042, 6: 0.5703, 8: 0.8383, 9: 0.6253}
TEST_EVT = {0: 0.709,  1: 0.638,  2: 0.125,  3: 0.375,  4: 0.574,
            5: 0.286,  6: 0.133,  8: 0.000,  9: 0.769}

def available_trials(subj):
    files = glob.glob(f'../data/subject{subj}/subject_{subj}_tache_{TACHE}_trial_*.csv')
    return sorted(int(f.split('trial_')[1].replace('.csv','')) for f in files)

def press_onsets(button_arr):
    arr = np.asarray(button_arr)
    return np.where(np.diff(arr, prepend=arr[0]) > 0)[0]

def extract_epochs(X, onsets, pre, post):
    epochs = []
    for onset in onsets:
        s, e = onset - pre, onset + post
        if s >= 0 and e <= len(X):
            epochs.append(X[s:e])
    return np.stack(epochs) if epochs else np.empty((0, pre + post, X.shape[1]))

# ── Compute ERPs ──────────────────────────────────────────────────────────────
subject_erps = {}
EEG_COLS = None

for subj in SUBJECTS:
    data_path = f'../data/subject{subj}/'
    trials = available_trials(subj)
    if not trials:
        continue
    all_epochs = []
    for t in trials:
        df, cols = load_trial(subj, TACHE, t, data_path,
                              eeg_cols=EEG_COLS, fs=FS, lp=LP, hp=HP,
                              decimate=1, car=True)
        if EEG_COLS is None:
            EEG_COLS = cols
        X = df[EEG_COLS].values.astype(np.float32)
        epochs = extract_epochs(X, press_onsets(df['button'].values), PRE, POST)
        all_epochs.append(epochs)
    all_epochs = np.concatenate(all_epochs)
    baseline = all_epochs[:, :PRE, :].mean(axis=1, keepdims=True)
    all_epochs -= baseline
    subject_erps[subj] = all_epochs.mean(axis=0)  # (T, C), in V
    print(f'Subject {subj}: {len(all_epochs)} epochs')

N_CH = len(EEG_COLS)

# ── Extract peak per subject ───────────────────────────────────────────────────
# Look in the pre-press window only (-PRE_MS to 0 ms)
pre_mask = TIME_AX < 0

erp_peaks   = {}  # subj -> (peak_raw, best_channel_name)
# Raw CSV values are ADC counts (~100K), not Volts.
# After bandpass + CAR the signal is centred at 0 in ADC units.
# Use the late pre-press window (-500 to 0 ms) — avoids filter edge
# artefacts at the epoch start and targets the MRCP negative ramp.
late_mask = (TIME_AX >= -500) & (TIME_AX < 0)

for subj, erp in subject_erps.items():
    late_erp = erp[late_mask, :]        # (T_late, C), raw filtered ADC units
    ch_means = late_erp.mean(axis=0)    # mean over window per channel
    best_ch  = int(np.argmin(ch_means)) # most negative channel
    peak     = float(ch_means[best_ch]) # mean amplitude on that channel
    erp_peaks[subj] = (peak, EEG_COLS[best_ch])
    print(f'  Subject {subj}: best channel = {EEG_COLS[best_ch]}, peak = {peak:.1f} ADC')

# Subjects with both ERP and test data, excluding artefact-dominated subjects
EXCLUDE = {6, 8}
common = sorted(set(erp_peaks) & set(TEST_AUC) - EXCLUDE)
peaks  = np.array([erp_peaks[s][0] for s in common])  # negative = stronger MRCP
auc    = np.array([TEST_AUC[s]     for s in common])
evt    = np.array([TEST_EVT[s]     for s in common])
labels = [f'S{s}\n({erp_peaks[s][1]})' for s in common]

# ── Scatter plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y, ylabel, title in [
    (axes[0], auc, 'Test AUC (per-subject SVM)',       'MRCP Peak vs Test AUC'),
    (axes[1], evt, 'Test Event Sensitivity',            'MRCP Peak vs Test Event Sensitivity'),
]:
    # scatter
    sc = ax.scatter(peaks, y, s=90, zorder=3,
                    c=y, cmap='RdYlGn', vmin=0.3, vmax=0.85, edgecolors='k', lw=0.7)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (peaks[i], y[i]),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    # regression line
    slope, intercept, r, p, _ = stats.linregress(peaks, y)
    x_line = np.linspace(peaks.min() - 5, peaks.max() + 5, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.2,
            label=f'r={r:.2f}, p={p:.3f}')

    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('MRCP pre-press amplitude (most negative channel, −500–0 ms, raw ADC units)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.colorbar(sc, ax=axes[1], label='Performance value')
fig.suptitle('ERP signal strength vs classifier performance (per-subject SVM RBF, trials 8–9, excl. subj 6 & 8)',
             fontsize=11)
plt.tight_layout()
plt.savefig('report/erp_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: report/erp_vs_performance.png')
print(f'\nPearson r (peak vs AUC): {stats.pearsonr(peaks, auc)[0]:.3f}  '
      f'p={stats.pearsonr(peaks, auc)[1]:.3f}')
print(f'Pearson r (peak vs EvtSens): {stats.pearsonr(peaks, evt)[0]:.3f}  '
      f'p={stats.pearsonr(peaks, evt)[1]:.3f}')
