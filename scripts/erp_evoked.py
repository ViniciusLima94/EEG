import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

print(f'Epoch: -{PRE_MS} ms to +{POST_MS} ms  ({PRE+POST} samples at {FS} Hz)')

def available_trials(subj):
    files = glob.glob(f'../data/subject{subj}/subject_{subj}_tache_{TACHE}_trial_*.csv')
    return sorted(int(f.split('trial_')[1].replace('.csv','')) for f in files)

def press_onsets(button_arr):
    arr = np.asarray(button_arr)
    diff = np.diff(arr, prepend=arr[0])
    return np.where(diff > 0)[0]

def extract_epochs(X, onsets, pre, post):
    epochs = []
    for onset in onsets:
        start, end = onset - pre, onset + post
        if start >= 0 and end <= len(X):
            epochs.append(X[start:end])
    return np.stack(epochs) if epochs else np.empty((0, pre + post, X.shape[1]))

# ── Load and epoch ────────────────────────────────────────────────────────────
subject_erps = {}
subject_n    = {}
EEG_COLS     = None

for subj in SUBJECTS:
    data_path = f'../data/subject{subj}/'
    trials = available_trials(subj)
    if not trials:
        print(f'Subject {subj}: no trials found, skipping')
        continue

    all_epochs = []
    for t in trials:
        df, cols = load_trial(subj, TACHE, t, data_path,
                              eeg_cols=EEG_COLS, fs=FS, lp=LP, hp=HP,
                              decimate=1, car=True)
        if EEG_COLS is None:
            EEG_COLS = cols
        X      = df[EEG_COLS].values.astype(np.float32)
        onsets = press_onsets(df['button'].values)
        epochs = extract_epochs(X, onsets, PRE, POST)
        all_epochs.append(epochs)

    all_epochs = np.concatenate(all_epochs, axis=0)
    baseline   = all_epochs[:, :PRE, :].mean(axis=1, keepdims=True)
    all_epochs = all_epochs - baseline

    subject_erps[subj] = all_epochs.mean(axis=0)
    subject_n[subj]    = len(all_epochs)
    print(f'Subject {subj}: {len(trials)} trials, {len(all_epochs)} epochs')

N_CH = len(EEG_COLS)
print(f'\n{len(subject_erps)} subjects loaded, {N_CH} channels')

# ── Plot 1: per-subject grid ──────────────────────────────────────────────────
n_subj = len(subject_erps)
ncols  = 2
nrows  = (n_subj + 1) // ncols
cmap   = plt.cm.tab20(np.linspace(0, 1, N_CH))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), sharey=False)
axes = axes.flatten()

for ax_idx, (subj, erp) in enumerate(sorted(subject_erps.items())):
    ax = axes[ax_idx]
    for ch in range(N_CH):
        ax.plot(TIME_AX, erp[:, ch] * 1e6, alpha=0.55, lw=0.8, color=cmap[ch])
    ax.plot(TIME_AX, erp.mean(axis=1) * 1e6, color='black', lw=2, label='Mean')
    ax.axvline(0, color='red', lw=1.2, ls='--', label='Press')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'Subject {subj}  (n={subject_n[subj]} epochs)', fontsize=11)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.invert_yaxis()
    if ax_idx == 0:
        ax.legend(loc='lower left', fontsize=8)

for ax in axes[n_subj:]:
    ax.set_visible(False)

handles = [Line2D([0],[0], color=cmap[i], lw=1.5, label=EEG_COLS[i]) for i in range(N_CH)]
fig.legend(handles=handles, loc='lower right', ncol=4, fontsize=7,
           title='Channels', bbox_to_anchor=(0.98, 0.01))
fig.suptitle('MRCP — Evoked Potential Aligned to Button Press (0.5–8 Hz + CAR)',
             fontsize=13)
plt.tight_layout()
plt.savefig('report/erp_per_subject.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: report/erp_per_subject.png')

# ── Plot 2: grand average (exclude subject 6 — EMG artifact) ─────────────────
ARTIFACT_SUBJ = {6}
clean_erps = {s: e for s, e in subject_erps.items() if s not in ARTIFACT_SUBJ}
grand_avg = np.mean(list(clean_erps.values()), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
for ch in range(N_CH):
    ax.plot(TIME_AX, grand_avg[:, ch] * 1e6, alpha=0.6, lw=0.9,
            color=cmap[ch], label=EEG_COLS[ch])
ax.plot(TIME_AX, grand_avg.mean(axis=1) * 1e6, color='black', lw=2.5, label='Mean')
ax.axvline(0, color='red', lw=1.5, ls='--', label='Press')
ax.axhline(0, color='gray', lw=0.5)
ax.invert_yaxis()
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (µV)')
ax.set_title(f'Grand Average — All Channels (subjects {sorted(clean_erps.keys())})')
ax.legend(loc='lower left', fontsize=7, ncol=2)

ax = axes[1]
mean_ch = grand_avg.mean(axis=1) * 1e6
std_ch  = grand_avg.std(axis=1) * 1e6
ax.plot(TIME_AX, mean_ch, color='steelblue', lw=2)
ax.fill_between(TIME_AX, mean_ch - std_ch, mean_ch + std_ch,
                alpha=0.3, color='steelblue', label='±1 SD across channels')
ax.axvline(0, color='red', lw=1.5, ls='--', label='Press')
ax.axhline(0, color='gray', lw=0.5)
ax.invert_yaxis()
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (µV)')
ax.set_title('Grand Average — Mean ± SD across channels (excl. subj 6)')
ax.legend(fontsize=9)

fig.suptitle('MRCP Grand Average (0.5–8 Hz + CAR, all subjects)', fontsize=12)
plt.tight_layout()
plt.savefig('report/erp_grand_average.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: report/erp_grand_average.png')

# ── Plot 3: heatmap ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
data_uv = grand_avg.T * 1e6
vmax = np.percentile(np.abs(data_uv), 95)
im = ax.imshow(data_uv, aspect='auto', origin='upper',
               extent=[TIME_AX[0], TIME_AX[-1], N_CH - 0.5, -0.5],
               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.axvline(0, color='black', lw=1.5, ls='--')
ax.set_yticks(range(N_CH))
ax.set_yticklabels(EEG_COLS, fontsize=8)
ax.set_xlabel('Time (ms)')
ax.set_title('Grand Average MRCP — Channel × Time Heatmap (µV, negative = blue, excl. subj 6)')
plt.colorbar(im, ax=ax, label='µV')
plt.tight_layout()
plt.savefig('report/erp_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: report/erp_heatmap.png')

# ── Summary stats ─────────────────────────────────────────────────────────────
print('\nPer-subject MRCP summary:')
print(f'{"Subject":>8}  {"Epochs":>7}  {"Trough µV":>10}  {"Trough ms":>10}  {"Post-peak µV":>13}  {"Post-peak ms":>13}')
print('-' * 75)
for subj in sorted(subject_erps):
    erp    = subject_erps[subj]
    mean_t = erp.mean(axis=1) * 1e6
    trough_idx = np.argmin(mean_t)
    trough_t   = TIME_AX[trough_idx]
    trough_v   = mean_t[trough_idx]
    post_mask  = TIME_AX >= 0
    peak_idx   = np.argmax(mean_t[post_mask])
    peak_t     = TIME_AX[post_mask][peak_idx]
    peak_v     = mean_t[post_mask][peak_idx]
    print(f'{subj:>8}  {subject_n[subj]:>7}  {trough_v:>10.3f}  {trough_t:>10.1f}  '
          f'{peak_v:>13.3f}  {peak_t:>13.1f}')

grand_mean_t = grand_avg.mean(axis=1) * 1e6
trough_idx = np.argmin(grand_mean_t)
print(f'\nGrand average trough (excl. subj 6): {grand_mean_t[trough_idx]:.3f} µV at {TIME_AX[trough_idx]:.1f} ms')
