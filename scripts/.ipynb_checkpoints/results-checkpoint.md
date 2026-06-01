# EEG MRCP Classification — Results

**MRCP button-press detection, band-pass 0.5–8 Hz, HORIZON = 1000 ms**
**Two subjects: Subject 9 (spt task, 10 trials, FS=250 Hz) · Subject 0 (spt task, 10 trials, FS=250 Hz)**

---

## Best Model

The answer depends on the evaluation regime:

### Single-trial (train and test on the same recording session)
**Winner: Random Forest (`eeg_causal_rf`) — LOTO AUC 0.8324, Test AUC 0.8002**

Within a single trial, the RF learns session-specific MRCP amplitude and timing and generalises well to the final portion of the same recording. This is the performance ceiling: a model calibrated in real time on the current session. Not deployable without an online calibration phase.

### Cross-trial, sample-level AUC (the main generalization benchmark)
**Winner: RLDA + CAR (`eeg_svm_multitrial`) — LOTO AUC 0.6445 ± 0.047 over 10 trials**

Regularised LDA with Common Average Reference is the best model for generalising across recording sessions within the same subject. It matches the SVM RBF in AUC (0.6445 vs 0.6394) but with lower false-alarm rate (32.9 vs 34.4 FA/min) and faster inference. This matches the prediction of Lotte et al. 2007 that RLDA and SVM perform equivalently for MRCP-class BCI problems.

### Cross-trial, event-level detection (practical BCI metric)
**Winner: SVM RBF + CAR (`eeg_svm_multitrial`) — 74.5% event sensitivity, 677 ms mean latency**

If the goal is to detect as many individual button presses as possible before they happen, SVM RBF detects 74.5% of events (vs 60.8% for RLDA) at a mean latency of 677 ms before the press, with 34.4 false alarms per minute. This is the most practically relevant metric for a real-time BCI.

### Summary table

| Regime | Best model | Key metric |
|---|---|---|
| Single-trial (calibrated) | RF `eeg_causal_rf` | Test AUC **0.80** |
| Cross-trial, sample AUC | RLDA + CAR | LOTO AUC **0.6445**, FA/min 32.9 |
| Cross-trial, event detection | SVM RBF + CAR | Event sensitivity **74.5%**, latency 677 ms |

**Cross-trial generalization ceiling: ~0.62–0.64 LOTO AUC** — consistent across all model families (RF, LightGBM, SVM, Conv1D) on 10-trial LOTO. Closing the 0.20 gap to single-trial performance is the central open problem; domain shift between trials is the bottleneck, not model capacity.

---

## 1. CAR vs No-CAR Comparison

### 1a. LGBM Multitrial (LOTO CV, all 10 trials, Subject 9)

| Metric | Without CAR | With CAR | Δ |
|---|---|---|---|
| LOTO AUC | 0.5654 ± 0.056 | **0.6182 ± 0.049** | +0.053 |
| LOTO AP | 0.6091 ± 0.127 | **0.6559 ± 0.108** | +0.047 |
| Smooth AUC (500 ms window) | 0.5165 | **0.5634** | +0.047 |
| Sensitivity (spec ≥ 0.80) | 18.6% | **26.8%** | +8.2 pp |
| Specificity | 80.0% | 80.0% | — |
| Best T | 710 ms | 984 ms | longer window |

### 1b. SVM Multitrial (LOTO CV, all 10 trials, Subject 9)

| Metric | SVM no CAR | SVM + CAR | Δ | RLDA no CAR | RLDA + CAR | Δ |
|---|---|---|---|---|---|---|
| LOTO AUC | 0.5902 ± 0.079 | **0.6394 ± 0.046** | +0.049 | 0.5996 ± 0.070 | **0.6445 ± 0.047** | +0.045 |
| LOTO AP | 0.6272 ± 0.114 | **0.6735 ± 0.114** | +0.046 | 0.6278 ± 0.126 | **0.6567 ± 0.121** | +0.029 |
| Smooth AUC | 0.5594 | **0.6130** | +0.054 | 0.5520 | **0.6000** | +0.048 |
| Event sensitivity | 62.4% | **74.5%** | +12.1 pp | 51.3% | **60.8%** | +9.5 pp |
| FA / min | 32.7 | 34.4 | +1.7 | 32.1 | 32.9 | +0.8 |
| Mean latency (ms) | 738.8 | **676.9** | −62 ms | 767.9 | **751.1** | −17 ms |
| Best T | 952 ms | 1484 ms | — | 887 ms | 887 ms | — |

---

## 2. Full Model Comparison

| Notebook | Model | Subject | Approach | Look-back T | LOTO AUC | Test AUC | Filter |
|---|---|---|---|---|---|---|---|
| `eeg_causal_rf` | Random Forest | 9 | In-trial CV (trial 0) | 952 ms | 0.8324 ± 0.060 | 0.8002 | broadband; single-trial |
| `eeg_svm_single_trial` | SVM linear + PCA | 9 | Single trial (trial 0) | ~500 ms | — | 0.7775 (0.6293 smooth) | broadband; single-trial |
| `eeg_rf_balanced` | Balanced RF | 9 | LOTO (2 trials) | 1116 ms | **0.7343 ± 0.054** | — | 0.1–60 Hz; 2 folds only |
| `eeg_svm_multitrial` RLDA | RLDA + CAR | 9 | LOTO (10 trials) | 887 ms | 0.6445 ± 0.047 | — | 0.5–8 Hz |
| `eeg_svm_multitrial` SVM | SVM RBF + CAR | 9 | LOTO (10 trials) | 1484 ms | 0.6394 ± 0.046 | — | 0.5–8 Hz |
| `eeg_lgbm_multitrial` | LightGBM + CAR | 9 | LOTO (10 trials) | 984 ms | 0.6182 ± 0.049 | — | 0.5–8 Hz |
| `eeg_causal_rf_multitrial` | Causal RF + CAR | 9 | LOTO 8 / Test 2 | 512 ms | 0.6203 ± 0.105 | 0.5197 | 0.5–8 Hz |
| `eeg_conv1d_causal_multitrial` | Conv1D causal | 0 | LOTO 8 / Test 2 | 128 ms | **0.6131 ± 0.045** | 0.5708 | 0.1–60 Hz (broadband baseline) |
| `eeg_conv1d_causal_multitrial` +MRCP | Conv1D + CAR | 0 | LOTO 8 / Test 2 | 308 ms | 0.5790 ± 0.045 | 0.5808 | 0.5–8 Hz + CAR; LOTO −0.034 vs broadband |
| `eeg_rf_pooled_balanced` | Balanced RF + CAR | 9 | Train 0–7 / Test 8–9 | 968 ms | — | 0.6151 | 0.5–8 Hz |
| `eeg_conv1d_fixed_multitrial` | Conv1D causal + CAR | 0 | LOTO 8 / Test 2 | 176 ms | 0.5547 ± 0.042 | 0.5215 | 0.5–8 Hz |

---

## 3. LOTO Per-Trial AUC Breakdown

### LGBM + CAR (Subject 9, 10-trial LOTO)

| Trial | AUC | AP | Press % |
|---|---|---|---|
| 0 | 0.5991 | 0.4015 | 28.7% |
| 1 | 0.6477 | 0.5864 | 43.7% |
| 2 | 0.5762 | 0.6457 | 56.2% |
| 3 | 0.6510 | 0.6078 | 48.4% |
| 4 | 0.5285 | 0.6260 | 61.1% |
| 5 | 0.5680 | 0.6746 | 60.4% |
| 6 | 0.6657 | 0.7411 | 63.1% |
| 7 | 0.5983 | 0.7174 | 63.7% |
| 8 | 0.6805 | 0.8128 | 67.6% |
| 9 | 0.6666 | 0.7460 | 60.4% |
| **Mean** | **0.6182 ± 0.049** | **0.6559 ± 0.108** | — |

### SVM RBF + CAR and RLDA + CAR (Subject 9, 10-trial LOTO)

| Trial | SVM AUC | SVM AP | RLDA AUC | RLDA AP |
|---|---|---|---|---|
| 0 | 0.7027 | 0.4086 | 0.6526 | 0.3599 |
| 1 | 0.6443 | 0.5385 | 0.6303 | 0.5291 |
| 2 | 0.5804 | 0.6397 | 0.5972 | 0.6479 |
| 3 | 0.4765 | 0.4996 | 0.4458 | 0.4913 |
| 4 | 0.4458 | 0.5863 | 0.5125 | 0.6290 |
| 5 | 0.5282 | 0.6421 | 0.5520 | 0.6635 |
| 6 | 0.6069 | 0.7206 | 0.6456 | 0.7331 |
| 7 | 0.6078 | 0.7340 | 0.6222 | 0.7097 |
| 8 | 0.6686 | 0.8046 | 0.6709 | 0.8058 |
| 9 | 0.6520 | 0.7276 | 0.7063 | 0.7325 |
| **Mean** | **0.6394 ± 0.046** | **0.6735 ± 0.114** | **0.6445 ± 0.047** | **0.6567 ± 0.121** |

### Causal RF + CAR (Subject 9, LOTO 8-trial train)

| Trial held | AUC |
|---|---|
| 0 | 0.8014 |
| 1 | 0.7293 |
| 2 | 0.7237 |
| 3 | 0.5141 |
| 4 | 0.5725 |
| 5 | 0.5320 |
| 6 | 0.5626 |
| 7 | 0.5271 |
| **Mean** | **0.6203 ± 0.105** |

### Conv1D Fixed + CAR (Subject 0, LOTO 8-trial train)

| Trial held | AUC | Epochs |
|---|---|---|
| 0 | 0.4984 | 20 |
| 1 | 0.5360 | 15 |
| 2 | 0.5921 | 11 |
| 3 | 0.5357 | 12 |
| 4 | 0.5159 | 11 |
| 5 | 0.5429 | 42 |
| 6 | 0.5838 | 11 |
| 7 | 0.6325 | 16 |
| **Mean** | **0.5547 ± 0.042** | — |

### Conv1D Causal (Subject 0, LOTO 8-trial train)

| Trial held | AUC | Epochs |
|---|---|---|
| 0 | 0.5578 | 35 |
| 1 | 0.5577 | 23 |
| 2 | 0.5650 | 21 |
| 3 | 0.6246 | 12 |
| 4 | 0.6163 | 12 |
| 5 | 0.6727 | 15 |
| 6 | 0.6708 | 21 |
| 7 | 0.6403 | 15 |
| **Mean** | **0.6131 ± 0.045** | — |

---

## 4. Event-Level Detection (SVM multitrial, Subject 9, with CAR)

| Model | Event Sensitivity | FA / min | Mean Latency | Events | Detected | False Alarms |
|---|---|---|---|---|---|---|
| SVM RBF | **74.5%** | 34.4 | 677 ms | 263 | 196 | 232 |
| RLDA | 60.8% | 32.9 | 751 ms | 263 | 160 | 223 |

### Metric definitions

**Sample-level sensitivity (Sens):** At each time step a label y=1 is assigned if a press will occur within the next 1000 ms (the horizon window). A threshold is chosen so specificity ≥ 0.80 on that split. Sensitivity = fraction of all y=1 samples correctly flagged. A single press generates ~62 consecutive positive samples, so the model gets partial credit for catching any of them.

**Event-level sensitivity (EvtSens):** For each discrete button-press event, the model fires a detection if its smoothed score exceeds the threshold at least once in the 1000 ms before that press. Sensitivity = fraction of press events where at least one alert was issued before the movement occurred. Each unique press counts once regardless of how many samples exceeded the threshold. This is the operationally meaningful metric for a BCI.

**Key implication:** A model can have low sample-level sensitivity but high event sensitivity (e.g. it fires one alert per press reliably) or vice versa. Subject 8 in the cross-subject run is the inverse extreme: test AUC = 0.84 (good ranking) but event sensitivity = 0.00 — the LOTO-calibrated threshold does not transfer to test trials, so no sample ever exceeds it and zero events are detected.

---

## 5. Best Optuna Configurations

| Notebook | Model | T (ms) | Key Hyperparameters |
|---|---|---|---|
| `eeg_lgbm_multitrial` | LightGBM + CAR | 984 ms (61 samples) | num_leaves=22, lr=0.073, subsample=0.859, colsample=0.889, class_weight=None |
| `eeg_svm_multitrial` | SVM RBF + CAR | 1484 ms (92 samples) | C=2.39, γ=7.3×10⁻⁴ |
| `eeg_svm_multitrial` | RLDA + CAR | 887 ms (55 samples) | shrinkage=auto (Ledoit-Wolf) |
| `eeg_rf_pooled_balanced` | Balanced RF + CAR | 968 ms (60 samples at FS_EFF=62) | n_estimators=284, max_depth=None, class_weight=balanced |
| `eeg_rf_balanced` | Balanced RF | 1116 ms (279 samples at FS=250) | n_estimators=172, max_depth=5, min_samples_split=13 |
| `eeg_causal_rf_multitrial` | Causal RF + CAR | 512 ms (128 samples at FS=250) | n_estimators=50, max_depth=5, min_samples_split=3, max_features=log2, class_weight=balanced |
| `eeg_conv1d_fixed_multitrial` | Conv1D + CAR | 176 ms (44 samples, fixed) | 5 layers × 60 filters, kernel=3, dropout=0.3, adamw lr=0.001 |
| `eeg_conv1d_causal_multitrial` | Conv1D causal | 128 ms (32 samples at FS=250) | 4 layers × 18 filters, kernel=11, elu, dropout=0.447, adamw lr=3.5e-4 |

---

## 6. Key Observations

1. **CAR gives a consistent +0.05 AUC lift** across all LOTO models (LGBM, SVM, RLDA).
2. **Single-trial ceiling is ~0.80–0.83**: in-trial CV reaches 0.83 (causal RF) and 0.78 (SVM), confirming the MRCP signal is detectable within a trial.
3. **Cross-trial generalization ceiling is ~0.62–0.64**: all multi-trial models on 10-trial LOTO converge to this range regardless of model family (RF, LightGBM, SVM RBF, RLDA, Conv1D). Domain shift between trials is the bottleneck, not model capacity.
4. **LOTO → held-out test gap is large**: causal RF drops 0.62 → 0.52, Conv1D fixed drops 0.55 → 0.52. Generalization to truly unseen trials (8-9) is worse than leave-one-out suggests.
5. **Tuned Conv1D ≈ tuned RF** at LOTO (0.61 vs 0.62), and both beat the fixed-architecture Conv1D (0.55), showing that architecture search matters more than model family.
6. **Short windows underperform**: Conv1D fixed (176 ms) → 0.55 LOTO vs longer-window RF/SVM (887–1484 ms) → 0.62–0.64. The MRCP ramp begins 1–1.5 s before movement; 176 ms captures only the terminal phase.
7. **RLDA ≈ SVM RBF** in LOTO AUC (0.6445 vs 0.6394), matching Lotte et al. 2007. RLDA has lower FA rate; SVM RBF has higher event sensitivity (+12 pp event-level).
8. **False alarm rate remains high** (~33–34 FA/min = ~1 per 2 s) for the best multi-trial models; feature engineering (Hjorth, band power, derivatives) is the next lever.
9. **MRCP filter (0.5–8 Hz) helps classical models but hurts Conv1D**: applying the filter to LGBM/SVM/RF gave +0.05 AUC; applying it to Conv1D dropped LOTO by −0.034 (0.6131 → 0.5790). Conv1D uses broadband features (likely beta ERD 15–30 Hz) that the filter removes; classical models lack the architecture to suppress out-of-band artifacts, so the filter does it for them.

---

## 7. Next Steps (Priority Order)

| Priority | Change | Expected Gain |
|---|---|---|
| 1 | ~~Apply 0.5–8 Hz filter to Conv1D~~ ✗ done — hurt LOTO −0.034 | Keep broadband for Conv1D; filter helps only classical models |
| 2 | Extend Conv1D window T to 512–1000 ms (currently 128–308 ms) | Full MRCP ramp is 1–1.5 s; short T is the main remaining gap |
| 3 | Add beta band (15–30 Hz) power feature to Conv1D | Explicit beta ERD feature may be more robust than raw broadband |
| 4 | Temporal derivative features for RF/LGBM (`add_derivative=True`) | Capture MRCP negative slope explicitly |
| 5 | Band power features (delta 0.5–4 Hz, theta 4–8 Hz) | Reduce dimensionality while adding frequency signal |
| 6 | Matched filter / template correlation | Literature: ~80% accuracy, 12 ms latency |
| 7 | Event-level metric as primary optimization target | FA/min + sensitivity instead of sample-level AUC |
| 8 | Per-trial amplitude normalization before pooling | Reduce cross-trial domain shift |

---

## 8. RF Pooled Balanced (CAR + T floor 500 ms, Subject 9)

| Metric | Value |
|---|---|
| Optuna best AUC (inner 3-fold GroupKFold CV) | 0.6397 |
| Best T | 60 samples (968 ms at FS_EFF=62 Hz) |
| Test AUC (trials 8–9) | **0.6151** |
| Test Accuracy | 0.5951 (threshold=0.518) |
| CV AUC on balanced train (5-fold) | 0.9884 ± 0.0019 — strong overfit on train |
| n_estimators | 284 |
| max_depth | None (fully grown) |
| class_weight | balanced |
| Top channels | ch1 (0.094), ch16 (0.092), ch2 (0.076) |

---

## 9. RF Balanced — LOTO (Subject 9, FS=250 Hz)

| Metric | Value |
|---|---|
| Band-pass | 0.1–60 Hz (FS=250 Hz, no decimation) |
| Look-back T | 279 samples (1116 ms) |
| LOTO folds | 2 (trials 0 and 1) |
| LOTO AUC | **0.7343 ± 0.0536** |
| Smooth AUC (500 ms window) | 0.6490 |
| Accuracy @ optimal threshold | 0.7417 |
| Threshold (Youden's J) | 0.414 |
| n_estimators | 172 |
| max_depth | 5 |
| min_samples_split | 13 |

| Trial | AUC |
|---|---|
| 0 | 0.7879 |
| 1 | 0.6808 |

Note: Only 2 trials in this configuration. High LOTO AUC likely reflects easier 2-trial generalization vs 10-trial LOTO.

---

## 10. Conv1D Fixed Multitrial (Subject 0, 0.5–8 Hz + CAR)

| Metric | Value |
|---|---|
| Window T | 44 samples (176 ms) |
| Train / Test trials | 0–7 / 8–9 |
| Architecture | 5 layers × 60 filters (grow), kernel=3, 494k params |
| LOTO AUC | 0.5547 ± 0.0416 |
| LOTO Accuracy | 0.6597 |
| Test AUC | 0.5215 |
| Test trial 8 AUC | 0.5795 |
| Threshold | 0.541 |

---

## 11. Causal RF Multitrial (Subject 9, 0.5–8 Hz + CAR)

| Metric | Value |
|---|---|
| Window T | 128 samples (512 ms at FS=250 Hz) |
| Train / Test trials | 0–7 / 8–9 |
| Optuna best AUC | 0.6287 |
| LOTO AUC | 0.6203 ± 0.1053 |
| Test AUC | 0.5197 |
| Best params | n_estimators=50, max_depth=5, max_features=log2, class_weight=balanced |

Note: High variance (range 0.51–0.80 across folds); trials 0–2 ≥0.72 AUC, trials 3–7 near chance. Strong cross-trial domain shift.

---

## 12. Conv1D Causal Multitrial — Filter Experiment (Subject 0)

### 12a. Broadband (0.1–60 Hz) — best result

| Metric | Value |
|---|---|
| Window T | 32 samples (128 ms) |
| Optuna trials | 3, T range 16–256 |
| Optuna best AUC | 0.5995 |
| LOTO AUC | **0.6131 ± 0.0450** |
| Test AUC | 0.5708 |
| Test trial 8 / 9 | 0.5647 / 0.5847 |
| Threshold | 0.533 |
| Architecture | 4 layers × 18 filters, kernel=11, elu, BN, dropout=0.447, 54k params |
| Optimizer | adamw lr=3.5e-4, wd=3.1e-3, batch=256 |

### 12b. MRCP filter (0.5–8 Hz + CAR) — worse

| Metric | Value |
|---|---|
| Window T | 77 samples (308 ms) |
| Optuna trials | 5, T range 64–256 |
| Optuna best AUC | 0.5879 |
| LOTO AUC | 0.5790 ± 0.0445 (−0.034 vs broadband) |
| Test AUC | 0.5808 (+0.010 vs broadband) |
| Test trial 8 / 9 | 0.6106 / 0.5457 |
| Threshold | 0.475 |
| Architecture | 5 layers × 9 filters (grow), kernel=3, relu, dropout=0.365, 54k params |
| Optimizer | adam lr=1.3e-4, wd=8.6e-5, batch=128 |

**Per-trial LOTO AUC (MRCP run):**

| Trial held | AUC | Epochs |
|---|---|---|
| 0 | 0.4980 | 28 |
| 1 | 0.5273 | 14 |
| 2 | 0.6356 | 18 |
| 3 | 0.5605 | 27 |
| 4 | 0.6226 | 11 |
| 5 | 0.6069 | 14 |
| 6 | 0.5992 | 12 |
| 7 | 0.5818 | 19 |

**Why the MRCP filter hurt Conv1D:** The Conv1D leverages information above 8 Hz (likely beta ERD/ERS at 15–30 Hz) that predicts movement. Stripping this to 0.5–8 Hz removes that signal while leaving only the slow MRCP ramp — which at T=308 ms is still too short to see the full 1–1.5 s build-up. Classical models (LGBM, SVM, RF) benefit from the MRCP filter because they have no mechanism to suppress EMG/HF artifacts; the Conv1D handles this implicitly via learned filters and per-window normalisation.

---

## 13. Cross-Subject Evaluation — SVM RBF + CAR (All 10 Subjects)

**Setup:** Per-subject Optuna search (N=8 trials, GroupKFold on TRAIN_TRIALS), LOTO CV on trials 0–7, test on trials 8–9. Band-pass 0.5–8 Hz + CAR, HORIZON=1000 ms, DECIM=4 (FS_EFF=62 Hz). Sensitivity reported at specificity ≥ 0.80. Subject 7 has no trials 8–9 (LOTO only).

| Subject | Train/Test | T (ms) | LOTO AUC | LOTO Sens | LOTO EvtSens | Test AUC | Test Sens | Test EvtSens |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 / 2 | 97 | 0.5310 ± 0.039 | 0.225 | 0.591 | 0.5337 | 0.275 | 0.709 |
| 1 | 7 / 2 | 403 | 0.5377 ± 0.061 | 0.257 | 0.440 | 0.6585 | 0.318 | 0.638 |
| 2 | 8 / 2 | 65 | 0.5461 ± 0.078 | 0.237 | 0.520 | 0.3682 | 0.007 | 0.125 |
| 3 | 8 / 2 | 871 | **0.6014 ± 0.104** | **0.391** | 0.524 | 0.5626 | 0.195 | 0.375 |
| 4 | 8 / 2 | 581 | 0.5305 ± 0.050 | 0.489 | 0.494 | **0.7099** | **0.598** | 0.574 |
| 5 | 8 / 2 | 65 | 0.5024 ± 0.056 | 0.175 | 0.366 | 0.5042 | 0.125 | 0.286 |
| 6 | 8 / 2 | 581 | 0.5531 ± 0.150 | 0.267 | 0.164 | 0.5703 | 0.333 | 0.133 |
| 7 | 7 / — | 871 | 0.5188 ± 0.055 | 0.229 | 0.320 | — | — | — |
| 8 | 8 / 2 | 97 | 0.4861 ± 0.212 | 0.109 | 0.581 | 0.8383 † | 0.740 | 0.000 † |
| 9 | 8 / 2 | 403 | 0.5335 ± 0.108 | 0.257 | 0.370 | 0.6253 | 0.377 | 0.769 |
| **Mean** | | | **0.5341 ± 0.029** | **0.264** | **0.437** | **0.5968** | **0.330** | **0.401** |

† Subject 8: test AUC=0.84 but event sensitivity=0.00 — the LOTO threshold does not transfer to test trials (all test samples predicted positive; AUC reflects ranking only).

### Per-subject LOTO AUC breakdown

| Subject | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.576 | 0.539 | 0.474 | 0.500 | 0.484 | 0.584 | 0.559 | 0.532 | 0.531 |
| 1 | 0.440 | 0.516 | — | 0.543 | 0.541 | 0.609 | 0.486 | 0.629 | 0.538 |
| 2 | 0.468 | 0.470 | 0.602 | 0.535 | 0.720 | 0.545 | 0.544 | 0.485 | 0.546 |
| 3 | 0.552 | 0.758 | 0.597 | 0.519 | 0.679 | 0.611 | 0.692 | 0.403 | 0.601 |
| 4 | 0.485 | 0.624 | 0.574 | 0.521 | 0.546 | 0.481 | 0.548 | 0.466 | 0.531 |
| 5 | 0.509 | 0.534 | 0.516 | 0.539 | 0.549 | 0.363 | 0.522 | 0.488 | 0.502 |
| 6 | 0.551 | 0.568 | 0.500 | 0.351 | 0.720 | 0.738 | 0.315 | 0.682 | 0.553 |
| 7 | 0.588 | 0.487 | 0.586 | 0.566 | 0.485 | 0.445 | 0.474 | — | 0.519 |
| 8 | 0.546 | 0.520 | 0.827 | 0.062 | 0.674 | 0.485 | 0.435 | 0.340 | 0.486 |
| 9 | 0.522 | 0.545 | 0.789 | 0.471 | 0.464 | 0.395 | 0.533 | 0.549 | 0.534 |

### Key observations

1. **Cross-subject LOTO AUC mean: 0.53** — substantially below the single-subject reference (Subject 9 full run: 0.6445). The model does not generalise well across subjects without per-subject recalibration.
2. **Optuna found short T for subjects 0, 2, 5, 8** (65–97 ms vs the optimal ~900–1500 ms for Subject 9). With only 8 Optuna trials the search is too sparse to reliably find the correct window length; these subjects are likely underperforming due to poor hyperparameters rather than weak MRCP signal.
3. **Subject 4 is the strongest cross-trial generaliser**: LOTO 0.53 but test AUC 0.71, test sensitivity 0.60. Its trials 8–9 are unusually consistent with training.
4. **Subject 3 has the best LOTO AUC (0.60)** — the only subject approaching Subject 9's original result.
5. **Subject 2 test AUC = 0.37** (worse than chance) — model predictions are anti-correlated with true labels on test trials, suggesting a sign flip or strong session-level artefact.
6. **High cross-subject variance** in both LOTO (0.49–0.60) and test (0.37–0.84) AUC confirms that the cross-trial generalisation bottleneck is even harder when crossing subjects than when crossing trials within a subject.
7. **Event sensitivity mean: 0.44 LOTO, 0.40 test** — well below Subject 9's original 74.5%. Threshold calibrated from short-T LOTO does not reliably transfer to test trials in most subjects.

### What would improve cross-subject results

- **More Optuna trials per subject (≥20)** to find the correct window length — the biggest single fix
- **Per-subject threshold calibration** (a few labelled test samples to recalibrate threshold before deployment)
- **Domain adaptation** (e.g. per-trial z-score before pooling, or adversarial feature alignment)

---

## 14. Pooled Cross-Subject Model — SVM RBF + CAR

**Setup:** Single SVM trained on pooled trials 0–7 from all 10 subjects. Band-pass 0.5–8 Hz + CAR, HORIZON=1000 ms, DECIM=4 (FS_EFF=62 Hz). Optuna (12 trials) on random 80/20 stratified split of pool; each trial capped at OPTUNA_TRAIN_MAX for speed. Final model capped at SVM_FINAL_MAX drawn from the full pool (SVM RBF is O(n²) — training on the full pool is infeasible). Tested on each subject's held-out trials 8–9. Subject 7 has no trials 8–9 (contributes to training only).

Three runs were performed, varying pool size and T search floor:

| Run | K/subj | Pool | Optuna cap | Final fit | N_opt | T floor | Best T | Opt val AUC |
|---|---|---|---|---|---|---|---|---|
| K=2K | 2 000 | 20 000 | 8 000 | 20 000 | 12 | free | 226 ms | 0.679 |
| K=10K | 10 000 | 92 155 | 15 000 | 30 000 | 12 | free | 226 ms | 0.721 |
| K=10K T≥500 | 10 000 | 92 155 | 15 000 | 30 000 | 20 | **500 ms** | **903 ms** | **0.767** |

*(In-distribution val AUC inflated — final model sees same subject pool as validation.)*

### Per-subject test results

| Subject | K=2K AUC | K=2K EvtS | K=10K AUC | K=10K EvtS | K=10K T≥500 AUC | K=10K T≥500 EvtS | Per-subj EvtS |
|---|---|---|---|---|---|---|---|
| 0 | 0.495 | 0.455 | 0.495 | 0.491 | 0.528 | 0.527 | 0.709 |
| 1 | 0.551 | 0.457 | 0.575 | 0.574 | **0.613** | 0.479 | 0.638 |
| 2 | 0.581 | **0.875** | 0.338 | 0.562 | 0.541 | 0.750 | 0.125 |
| 3 | 0.645 | 0.875 | **0.808** | **0.875** | 0.696 | **0.875** | 0.375 |
| 4 | **0.677** | **0.672** | 0.504 | 0.328 | 0.526 | 0.344 | 0.574 |
| 5 | 0.537 | 0.476 | 0.505 | 0.524 | **0.617** | **0.714** | 0.286 |
| 6 | 0.553 | 0.417 | 0.488 | 0.083 | 0.528 | 0.167 | 0.133 |
| 8 | 0.259 ‡ | 0.000 | 0.344 ‡ | 0.000 | 0.391 ‡ | 0.000 | 0.000 |
| 9 | 0.557 | 0.615 | 0.513 | 0.492 | 0.477 | 0.446 | 0.769 |
| **Mean** | **0.540** | **0.538** | **0.508** | **0.437** | **0.546** | **0.478** | **0.401** |

† Subject 8 per-subject: AUC=0.84 but EvtSens=0.00 — threshold calibration failure.  
‡ Subject 8 pooled: AUC < 0.5 in all runs — predictions anti-correlated on test trials.

### Comparison summary

| Metric | Per-subj SVM | K=2K (226ms) | K=10K (226ms) | K=10K T≥500ms (903ms) |
|---|---|---|---|---|
| Mean test AUC | 0.597 | 0.540 | 0.508 | **0.546** |
| Mean test Sens | 0.330 | 0.249 | 0.226 | 0.259 |
| Mean test EvtSens | 0.401 | **0.538** | 0.437 | 0.478 |

### Key observations

1. **T_min=500ms substantially improves AUC** (0.508→0.546) by forcing Optuna to find a window long enough to capture the readiness potential ramp. The K=10K T≥500ms run achieves the best mean test AUC of any pooled model.
2. **T=226ms was a Optuna artifact** — with free T search, log-uniform sampling over a wide range causes short windows to be over-visited. Short T + high γ overfits the validation set without capturing the MRCP slow ramp.
3. **Event sensitivity trade-off persists** — K=2K (T=226ms) still leads on EvtSens (0.538), while K=10K T≥500ms leads on AUC (0.546). The short-window model fires alerts more liberally (lower implicit threshold on features), catching more events at the cost of false alarms. Subjects 5 and 3 improve markedly with longer T.
4. **Pooled vs per-subject**: pooled models consistently exceed the per-subject baseline on EvtSens (+7–14 pp depending on run), confirming that cross-subject training generalises the MRCP shape better than per-subject fitting on 8 short trials.
5. **Subject 8 is unclassifiable** across all settings — consistent with ERP analysis showing late, weak MRCP (−8.4 µV trough at −17 ms vs −951 ms for subject 2).

### What the comparison tells us

The T floor is the most important hyperparameter for pooled MRCP models: T ≥ 500 ms is required to capture the readiness potential. With the T floor fixed, more data (K=10K vs K=2K) gives better AUC (the model has seen more MRCP morphologies) but slightly lower event sensitivity (threshold calibration is harder across diverse subjects). The recommended configuration for a production pooled model is **K=10K, T_min=500ms, T_max=1500ms, N_opt≥30**, plus per-subject threshold calibration from a short (2–3 trial) calibration session.

---

## 15. MRCP Evoked Potential Analysis

**Setup:** EEG epochs aligned to button press (t=0), window −1500 ms to +500 ms. Filter: 0.5–8 Hz (MRCP band) + CAR, FS=250 Hz (no decimation). Baseline correction: subtract mean of pre-stimulus window per epoch. All trials pooled per subject. Grand average and heatmap exclude Subject 6 (EMG artefact dominates).

Plots: `report/erp_per_subject.png` · `report/erp_grand_average.png` · `report/erp_heatmap.png`

### Per-subject epoch counts and MRCP statistics

Mean across all channels, baseline-corrected. Trough = most negative pre-/peri-press sample; post-peak = most positive sample after t=0.

| Subject | Epochs | Trough (µV) | Trough (ms) | Post-peak (µV) | Signal quality |
|---|---|---|---|---|---|
| 0 | 267 | −0.33 | +207 | +0.30 | **No MRCP** — trough is post-press; channels flat throughout |
| 1 | 375 | −6.20 | −1312 | +4.78 | Weak-moderate — very early onset (~−1300 ms), low amplitude |
| 2 | 193 | −38.15 | −951 | +36.24 | **Strong** — clear negative ramp ~950 ms pre-press, sharp rebound |
| 3 | 174 | −31.47 | −690 | +13.35 | **Strong** — MRCP onset ~700 ms pre-press, clean morphology |
| 4 | 240 | −5.05 | −574 | +3.71 | Moderate — visible ramp ~600 ms pre-press |
| 5 | 157 | −1.79 | +207 | +1.55 | **No MRCP** — trough post-press, marginal pre-press activity |
| 6 | 239 | −167.85 | +35 | +137.33 | **EMG artefact** — ±168 µV spike at press onset, no pre-press ramp; excluded from grand average |
| 7 | 28 | −0.92 | −1252 | +0.46 | Unreliable — only 28 epochs (7 short trials) |
| 8 | 42 | −8.37 | −17 | +7.14 | **No pre-press ramp** — deflection begins only 17 ms before press; 42 epochs |
| 9 | 278 | −22.93 | −1404 | +12.40 | **Strongest** — earliest onset (~−1400 ms), largest latency, smoothest buildup |

Grand average trough (excl. Subject 6): **−4.4 µV at −951 ms**. The negative deflection is visible in nearly all channels from ~−500 ms onward, with a canonical post-movement positivity after t=0. The heatmap confirms the negativity is distributed across most channels (not focal), consistent with the broad MRCP scalp distribution.

### Per-subject visual description

- **Subjects 2, 3, 9**: Classic MRCP morphology — smooth negative ramp building over hundreds of milliseconds, peaking just before press, sharp positive rebound afterward. Subject 9 has the widest buildup window (~1400 ms) and the most channel consistency.
- **Subject 6**: Massive artifact spike at t=0; completely masks any underlying slow ramp. The classifier's moderate AUC for this subject likely exploits the amplitude discontinuity at press rather than a genuine pre-press buildup.
- **Subject 8**: Brief, sharp deflection centred on press onset — looks more like a sensory response to the press than a preparatory motor potential. No usable anticipatory information.
- **Subjects 0, 5**: Virtually flat across all channels and time. No ERP visible even with 157–267 epochs. Either the MRCP for these subjects is sub-µV (very low signal-to-noise) or the task elicits no genuine voluntary motor preparation.
- **Subject 7**: Too few epochs (28) to draw conclusions; the average is dominated by trial noise.

### Connection to classifier performance

| Finding | Subjects | Classifier consequence |
|---|---|---|
| Strong MRCP, onset 700–1400 ms | 2, 3, 9 | Best classified; per-subject SVM and pooled model both work |
| Moderate MRCP, onset ~600 ms | 4 | Moderate AUC; benefits from per-subject T tuning |
| Weak/absent MRCP | 0, 5, 7 | Near-chance AUC in all models; no pre-press signal to detect |
| EMG artefact at press | 6 | Model detects muscle burst, not MRCP; EvtSens appears decent but may be trivial |
| No pre-press ramp | 8 | EvtSens=0.00 universally — nothing to detect before movement onset |

The ERP analysis predicts classifier performance better than any single model metric: subjects with visible MRCPs (2, 3, 9) are classifiable; subjects without (0, 5, 8) are not, regardless of model complexity or training data size.

### ERP amplitude vs classifier performance (scatter analysis)

Plot: `report/erp_vs_performance.png`

For each subject, the most negative channel mean in the −500–0 ms pre-press window (raw ADC units after 0.5–8 Hz + CAR filtering) is used as the MRCP signal strength proxy. Tested against per-subject SVM test AUC and test event sensitivity.

**Per-subject MRCP peaks (best channel, −500–0 ms window):**

| Subject | Best channel | Peak (ADC) | Test AUC | Test EvtSens |
|---|---|---|---|---|
| 0 | ch14 | −0.3 | 0.534 | 0.709 |
| 1 | ch6 | −10.9 | 0.659 | 0.638 |
| 2 | ch15 | −36.0 | 0.368 | 0.125 |
| 3 | ch4 | −206.4 | 0.563 | 0.375 |
| 4 | ch8 | −10.3 | 0.710 | 0.574 |
| 5 | ch1 | −4.1 | 0.504 | 0.286 |
| 6 ‡ | ch16 | −912.6 | 0.570 | 0.133 |
| 7 | ch16 | −66.1 | — | — |
| 8 ‡ | ch8 | −896.6 | 0.838 | 0.000 |
| 9 | ch1 | −253.7 | 0.625 | 0.769 |

‡ Subjects 6 and 8 excluded from correlation analysis (artifact-dominated peaks).

**Pearson correlations (subjects 0–5, 9; n=7):**

| | r | p |
|---|---|---|
| Peak vs AUC | −0.13 | 0.783 |
| Peak vs EvtSens | −0.21 | 0.654 |

**Key finding: MRCP amplitude is not a useful predictor of cross-trial classification performance.**

- The correlations are flat and non-significant after excluding artefact subjects.
- The two strongest MRCPs (S9: −254, S3: −206) span the full range of EvtSens (0.77 vs 0.38) — amplitude alone does not determine classifiability.
- **S2** has a clear MRCP (−36 ADC) but below-chance test AUC (0.37): its trials 8–9 are a different distribution from 0–7 (session-level non-stationarity), not a weak signal.
- **S0** has virtually no pre-press signal (−0.3 ADC) yet EvtSens=0.71 — the model fires liberally (FA/min=35.7), catching presses by high false-alarm rate rather than genuine detection.

**What actually predicts performance: cross-trial MRCP stationarity** — whether the MRCP morphology and amplitude remain consistent from session to session. This is invisible in a grand-average ERP but is the real bottleneck for BCI deployment.

---

*Last updated: 2026-05-31. ERP amplitude vs performance scatter analysis complete.*
