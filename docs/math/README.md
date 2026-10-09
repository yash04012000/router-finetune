# Maths notes

Why we compute things the way we do, so future-you doesn't have to re-derive it. One page per
topic, added as the project reaches it. Each page has: the idea in plain words, the formula, a
worked example with real numbers, and a pointer to the code.

| Page | Covers | Code | Status |
|---|---|---|---|
| [metrics.md](metrics.md) | Accuracy, precision, recall, F1, macro F1, confusion matrix, percentile latency, bootstrap confidence intervals | `src/router/metrics.py` | Written |
| [cost.md](cost.md) | Cost per 1,000 decisions for API, self-hosted and hybrid setups | `src/router/cost.py` | Written |
| [data-splits.md](data-splits.md) | Train/val/test, leakage, stratified splits, class imbalance, how test-set size sets precision | `src/router/dataset_build.py`, `splits.py` | Written |
| [tfidf-logreg.md](tfidf-logreg.md) | TF-IDF, softmax, cross-entropy, regularisation (`C`), class weights | `src/router/baselines/tfidf.py` | Written |
| calibration.md | Softmax with temperature, negative log-likelihood, expected calibration error | `src/router/calibrate.py` | Step 3.9 |
| memory.md | Bytes per parameter, optimizer state, mixed precision, what fills 8 GB | `scripts/profile_training.py` | Step 3.6 |
| lora.md | Why low-rank adapters work, parameter count, the `alpha / r` scale, what QLoRA changes | `src/router/lora_model.py` | PRD 4 |
| fine-tuning.md | Logits and softmax (3.3), cross-entropy and gradient descent (3.4), schedules, weight decay, early stopping (3.7) | training scripts | Steps 3.3, 3.4, 3.7 |
| hybrid.md | Confidence thresholds, risk-coverage, accuracy/cost trade-off | `src/router/hybrid.py` | PRD 6 |

Later topics (SFT, preference tuning, etc.) get their own page here too.
