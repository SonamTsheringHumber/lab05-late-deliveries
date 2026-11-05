# Late Deliveries

QuickBite wants to know, at the moment an order is placed, whether it will arrive more than ten minutes late, so it can offer the customer a $4 voucher before they are disappointed. This project builds a leak-free scikit-learn pipeline that predicts `late`, and a decision threshold chosen from the company's own costs ($4 for every order flagged, $15 for every late order not flagged).

## Setup

```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```
python -m latedeliveries.train
```

This recreates `models/late_orders.joblib`, `models/model_info.json`, `reports/cv_results.csv` and `reports/metrics.json`. It also writes `reports/threshold_costs.csv` and `reports/leakage_check.json`, which the notebook uses.

## Check it

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns used and columns left out

The model is used at the moment an order is placed, so a column is a legitimate feature only if its value is known at that moment.

| Column | Used? | Reason |
|---|---|---|
| distance_km, items, hour, weather, day_type, vehicle, courier_months | Yes | All known when the order is placed. |
| prep_minutes | Yes (judgement call) | I treat it as the kitchen's prep-time estimate at order time. It is sometimes missing, so it is filled with the median. If it were only recorded after cooking, it would have to be dropped and the model retrained. |
| order_id | No | An identifier with no meaning. A flexible model could memorise it instead of learning from real signal. |
| customer_rating | No | The customer gives it after delivery, so it is not known when the order is placed. It is leakage. |
| late | No | It is the target. |

**Seeing leakage for myself.** I ran the comparison a second time with `customer_rating` added back. The cross-validated ROC-AUC of logistic regression rose from 0.795 to 0.907, an increase of 0.112. A score that jumps this much from one column is a warning, not good news, so the column stays out.

### Split

A stratified 75/25 split with `random_state=SEED` (42, held in one constant in `pipeline.py`) gives 2,250 training orders (15.6% late) and 750 test orders. The test set is not used for anything until the final scoring.

### Pipeline

`build_pipeline(model)` returns a `ColumnTransformer` followed by the model. Numeric columns are filled with the median and then scaled. Categorical columns are one-hot encoded with `handle_unknown="ignore"`. The comparison, the grid search, the out-of-fold predictions and the final fit all use this pipeline, so preprocessing is learned inside each training fold only. The pipeline can predict for an order with a missing `prep_minutes` and a weather value it has never seen.

### Model choice

Five-fold stratified cross-validation on the training data only (mean ± standard deviation across folds):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best threshold | Best CV cost |
|---|---|---|---|---|---|---|---|
| dummy_baseline | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.2 | $5,265 |
| logistic_regression | 0.862 ± 0.009 | 0.654 ± 0.065 | 0.242 ± 0.053 | 0.352 ± 0.064 | 0.795 ± 0.034 | 0.3 | $4,031 |
| logistic_regression_balanced | 0.737 ± 0.019 | 0.333 ± 0.027 | 0.681 ± 0.049 | 0.447 ± 0.034 | 0.795 ± 0.034 | 0.7 | $4,098 |
| random_forest | 0.859 ± 0.008 | 0.736 ± 0.121 | 0.148 ± 0.036 | 0.246 ± 0.056 | 0.786 ± 0.037 | 0.25 | $4,199 |
| gradient_boosting | 0.861 ± 0.006 | 0.635 ± 0.050 | 0.265 ± 0.024 | 0.373 ± 0.029 | 0.788 ± 0.031 | 0.3 | $4,152 |

"Best CV cost" is the lowest out-of-fold total cost over the threshold grid. Flagging nobody costs $5,265 on the training set, which is what the baseline reaches.

Accuracy is misleading here. The baseline scores 0.844 by never flagging anybody, which gives recall 0 and saves nothing. I chose **logistic regression** for three reasons:

1. **It has the lowest cost of any real candidate.** Cost is what the company cares about. The random forest and gradient boosting do not beat it, so their extra complexity buys nothing.
2. **Its probabilities can be used directly with the cost rule.** Its ROC-AUC ties with the `class_weight="balanced"` version, and balancing raises recall (0.681) but inflates the probabilities, so its best threshold moves to 0.70 and precision falls to 0.333. The plain model's probabilities are well calibrated, so the break-even point of $4 / $15 = 0.27 applies almost directly.
3. **It is simple, fast and easy to explain**, which matters for the interface built on it in Module 06.

The ROC-AUC differences between the models are smaller than the fold-to-fold spread (about 0.03), so cost and simplicity decide the choice rather than the single highest number.

### Tuning

I ran `GridSearchCV` on logistic regression over `model__C` in [0.01, 0.1, 1, 10, 100] and `model__class_weight` in [None, "balanced"], which gives 10 combinations, with five-fold stratified cross-validation. The scoring metric is ROC-AUC. I used it because it does not depend on a threshold, and the threshold is chosen afterwards from the costs.

The best settings were `C=100` and `class_weight=None`. The best cross-validated ROC-AUC was 0.7952, against 0.7951 for the untuned model, a gain of about 0.0002. The fold-to-fold spread is 0.034. **Tuning did not beat the untuned model by more than the fold-to-fold spread**, so it changed essentially nothing.

### Threshold

I took out-of-fold predicted probabilities on the training data with `cross_val_predict`, and turned them into flags at every threshold from 0.05 to 0.95 in steps of 0.05 (an order is flagged when its probability is at least the threshold). The function `order_cost` computes $4 for every flagged order plus $15 for every late order not flagged, and `choose_threshold` picks the threshold with the lowest total, the smallest threshold winning a tie. The result is **0.30**. The test set did not influence it.

A threshold is needed because the two mistakes cost different amounts. A missed late order costs almost four times as much as a wasted voucher, so the default 0.5 would miss too many late orders. On the training data, the cost at 0.5 is $4,473, against $4,035 at 0.30 (see `reports/threshold_costs.csv`).

The $4,031 in the comparison table is for the untuned model. The $4,035 is for the tuned model whose out-of-fold probabilities set the threshold. The $4 difference is noise.

## Result

I fitted the final pipeline on the whole training set and scored the test set once, at threshold 0.30.

| | Not flagged | Flagged |
|---|---|---|
| Actually on time | 578 | 55 |
| Actually late | 64 | 53 |

At threshold 0.30: precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796.

| Policy (750 test orders) | Total cost |
|---|---|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Model at threshold 0.30 | $1,392 |

The model saves $363 (20.7%) compared with flagging nobody, about $0.48 per order. The test set contains only 117 late orders, so I would trust the direction of the saving but not its exact size.

## What the tests cover

There are 20 tests (`python -m pytest`). They protect the following behaviour:

- The feature list excludes `order_id`, `customer_rating` and `late` (parametrized).
- The feature list has no duplicates and matches the columns the split produces.
- The split is stratified, repeatable and has no overlap between train and test.
- Predicted probabilities are valid: between 0 and 1, and summing to 1 per order.
- The pipeline predicts for an order with a missing `prep_minutes` and an unseen weather value.
- `order_cost` is correct on small examples worked out by hand (parametrized).
- `choose_threshold` is correct on a hand-worked example, including one case where flagging nobody is cheapest and one where flagging everybody is.
- `policy_costs` gives the right cost for the three policies on a small example.
- A saved model predicts the same after reloading it (using `tmp_path`).

One thing I chose not to test is the exact model scores. They depend on the scikit-learn version, so such a test would fail for the wrong reason. The cost logic, which is the part that can go wrong without anyone noticing, is tested by hand instead.

## Known limitations

- The data are invented, so real-world performance is unknown.
- The test set has only 117 late orders, so the saving is noisy.
- `prep_minutes` is assumed to be known when the order is placed. If it is only recorded afterwards, it must be removed and the model retrained.
- The costs are fixed at $4 and $15. If they change, the threshold must be chosen again.
- The threshold grid has a step of 0.05, so a finer grid might find a slightly cheaper threshold.
- A joblib file can run code when it is loaded, so only load models you made yourself. `model_info.json` records the scikit-learn version used.