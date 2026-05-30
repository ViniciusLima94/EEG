# EEG MRCP Classification — Results

**Subject 9, task `spt`, 10 trials, band-pass 0.5–8 Hz, HORIZON = 1000 ms, FS_EFF = 62 Hz**

---

## 1. CAR vs No-CAR Comparison

### 1a. LGBM Multitrial (LOTO CV, all 10 trials)

| Metric | Without CAR | With CAR | Δ |
|---|---|---|---|
| LOTO AUC | 0.5654 ± 0.056 | **0.6182 ± 0.049** | +0.053 |
| LOTO AP | 0.6091 ± 0.127 | **0.6559 ± 0.108** | +0.047 |
| Smooth AUC (500 ms window) | 0.5165 | **0.5634** | +0.047 |
| Sensitivity (spec ≥ 0.80) | 18.6% | **26.8%** | +8.2 pp |
| Specificity | 80.0% | 80.0% | — |
| Best T | 710 ms | 984 ms | longer window |

### 1b. SVM Multitrial (LOTO CV, all 10 trials)

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

## 2. Full Model Comparison (all notebooks, with CAR where applied)

| Notebook | Model | Approach | Look-back T | LOTO / Test AUC | Sensitivity | Specificity | Notes |
|---|---|---|---|---|---|---|---|
| `eeg_causal_rf` | Random Forest | In-trial CV (trial 0) | 952 ms | 0.83 CV / 0.61 cross-trial | — | — | No CAR; eye-state task |
| `eeg_svm_single_trial` | SVM (linear + PCA) | Single trial (trial 0) | ~500 ms | 0.78 test / 0.63 smooth | 36.5% | 80.7% | No CAR |
| `eeg_lgbm_multitrial` | LightGBM + CAR | LOTO (10 trials) | 984 ms | 0.62 ± 0.049 | 26.8% | 80.0% | CAR applied |
| `eeg_svm_multitrial` | RLDA + CAR | LOTO (10 trials) | 887 ms | **0.64 ± 0.047** | 60.8% | — | CAR applied; best LOTO |
| `eeg_svm_multitrial` | SVM RBF + CAR | LOTO (10 trials) | 1484 ms | **0.64 ± 0.046** | **74.5%** | — | CAR applied; best event sens. |
| `eeg_rf_pooled_balanced` | Balanced RF + CAR | Train 0–7 / Test 8–9 | TBD | *(pending re-run)* | — | — | T floor = 500 ms fix in progress |

---

## 3. LOTO Per-Trial AUC Breakdown

### LGBM + CAR (LOTO)

| Trial | AUC | AP | Press % |
|---|---|---|---|
| 0 | — | — | 28.4% |
| 1 | — | — | 43.1% |
| 2 | — | — | 55.3% |
| 3 | — | — | 47.7% |
| 4 | — | — | 60.2% |
| 5 | — | — | 59.5% |
| 6 | — | — | 62.2% |
| 7 | — | — | 62.8% |
| 8 | — | — | 66.5% |
| 9 | 0.6666 | 0.7460 | 59.5% |
| **Mean** | **0.6182 ± 0.049** | **0.6559 ± 0.108** | — |

### SVM + CAR (LOTO)

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

---

## 4. Event-Level Detection (SVM multitrial, with CAR)

| Model | Event Sensitivity | FA / min | Mean Latency | Events | Detected | False Alarms |
|---|---|---|---|---|---|---|
| SVM RBF | **74.5%** | 34.4 | 677 ms | 263 | 196 | 232 |
| RLDA | 60.8% | 32.9 | 751 ms | 263 | 160 | 223 |

---

## 5. Best Optuna Configurations

| Model | T (ms) | Key Hyperparameters |
|---|---|---|
| LGBM + CAR | 984 ms (61 samples) | num_leaves=22, lr=0.073, subsample=0.859 |
| SVM RBF + CAR | 1484 ms (92 samples) | C=2.39, γ=7.3×10⁻⁴ |
| RLDA + CAR | 887 ms (55 samples) | shrinkage=auto (Ledoit-Wolf) |

---

## 6. Key Observations

1. **CAR gives a consistent +0.05 AUC lift** across all LOTO models (LGBM, SVM, RLDA).
2. **Event sensitivity improved +9–12 pp** with CAR; SVM detects 74.5% of press events vs 62.4% before.
3. **Models selected longer windows with CAR** (984–1484 ms vs 710–952 ms), suggesting CAR removes session-level artifacts that were confusing the short-window features.
4. **Cross-trial generalization is the central bottleneck** — LOTO AUC 0.62–0.64 vs within-trial 0.83.
5. **False alarm rate remains high** (~33–34 FA/min = ~1 every 2 s); further improvement needs Hjorth/band-power features or Laplacian spatial filtering.
6. **RLDA ≈ SVM** in LOTO AUC (0.6445 vs 0.6394), matching Lotte et al. 2007 prediction. RLDA has lower FA rate; SVM has higher event sensitivity.

---

## 7. Next Steps (Priority Order)

| Priority | Change | Expected Gain |
|---|---|---|
| 1 | RF pooled re-run with T floor = 500 ms (in progress) | Prevent degenerate 97 ms window |
| 2 | Temporal derivative features (`add_derivative=True`) | Explicitly encode MRCP ramp slope |
| 3 | Band power features (delta 0.5–4 Hz, theta 4–8 Hz) | Add frequency-domain signal |
| 4 | Matched filter / template correlation | Literature: ~80% accuracy, 12 ms latency |
| 5 | Soft / graduated horizon labels | Smoother decision boundary |
| 6 | Event-level metric as primary optimization target | FA/min + sensitivity instead of sample AUC |

---

*Last updated: 2026-05-30. RF pooled result pending.*
