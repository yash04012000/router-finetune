# 3.9 — Calibration: can we trust the confidence?

Status: Not started · Depends on: 3.8 · Next: PRD 4 (small LLM with LoRA)

## The one idea

When a model says "90% sure", it should be right about 90% of the time. Modern neural networks are usually **over-confident**: they say 99%
and are right 94%. The hybrid router (PRD 6) sends low-confidence messages to the large model, so it only works if the
confidence means something. **Temperature scaling** fixes most of the over-confidence with a single number.

## Goal

Measure how well DistilBERT's confidence matches reality, fix it with one parameter fit on val, and show the before and after.

## Concepts you will learn

- **Reliability diagram**: bucket predictions by confidence (0-10%, ..., 90-100%) and compare average confidence with actual accuracy in each bucket.
- **Expected calibration error (ECE)**: the average gap between confidence and accuracy, weighted by bucket size.
- **Negative log-likelihood (NLL)**: the loss used to fit the temperature.
- **Temperature scaling**: divide logits by `T` before softmax. `T > 1` softens, `T < 1` sharpens. It never changes which class wins,
  so accuracy is untouched; only the confidence changes.
- Why we fit `T` on **val** and report on **test**.

## What we build

```
src/router/calibrate.py        fit_temperature(logits, labels), apply, ece(); simple grid / 1-D search, plain NumPy
docs/math/calibration.md       softmax with temperature, NLL, ECE, a worked 3-example case checked against the code
src/router/baselines/distilbert.py   saves the fitted T next to the model; predictions use calibrated confidence
Playground / Scoreboard        a reliability diagram per model (bars: confidence vs accuracy), ECE column
tests/test_calibrate.py        calibrated synthetic logits give T near 1; over-confident ones give T > 1 and lower ECE
```

TF-IDF's probabilities get the same treatment as an extra row, since the calibration code is model-agnostic.

## Hands-on

1. Draw the reliability diagram of the uncalibrated model on val. Notice bars below the diagonal (over-confident).
2. Fit `T` on val; see ECE drop and accuracy stay identical.
3. Check on test: does the improvement hold on data not used for fitting?
4. Rerun predictions with calibrated confidence; compare the Scoreboard before and after.

## Done when

- `calibrate.py` is tested, `T` is stored, ECE before and after is reported for val and test.
- The playground shows calibrated confidence for DistilBERT.

## Check your understanding

1. Why can temperature scaling change confidence but never accuracy?
2. Why fit `T` on val rather than on train or test?
3. How would an over-confident model make the PRD 6 threshold choice worse?
