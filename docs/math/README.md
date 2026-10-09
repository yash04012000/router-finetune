# Maths notes

Why we compute things the way we do, so future-you doesn't have to re-derive it. One page per
topic, added as the project reaches it. Each page has: the idea in plain words, the formula, a
worked example with real numbers, and a pointer to the code.

| Page | Covers | Code | Status |
|---|---|---|---|
| [metrics.md](metrics.md) | Accuracy, precision, recall, F1, macro F1, confusion matrix, percentile latency, bootstrap confidence intervals | `src/router/metrics.py` | Written |
| [cost.md](cost.md) | Cost per 1,000 decisions for API, self-hosted and hybrid setups | `src/router/cost.py` | Written |
| [data-splits.md](data-splits.md) | Train/val/test, leakage, stratified splits, class imbalance, how test-set size sets precision | `src/router/dataset_build.py`, `splits.py` | Written |
| calibration.md | Softmax, temperature scaling, negative log-likelihood, expected calibration error | `src/router/calibrate.py` | PRD 3 |
| lora.md | Why low-rank adapters work, parameter count, the `alpha / r` scale, what QLoRA changes | `src/router/lora_model.py` | PRD 4 |
| fine-tuning.md | Cross-entropy loss, gradient descent, learning rate, warm-up, epochs, overfitting | training scripts | PRD 3 / 4 |
| hybrid.md | Confidence thresholds, risk-coverage, accuracy/cost trade-off | `src/router/hybrid.py` | PRD 6 |

Later topics (SFT, preference tuning, etc.) get their own page here too.
