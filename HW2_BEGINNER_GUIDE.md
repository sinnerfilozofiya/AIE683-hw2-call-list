# HW2 Beginner Guide: Who Should the Bank Call?

A plain-language walkthrough for someone new to machine learning.

---

## What is this homework about? (simple version)

A bank calls people to sell a **term deposit** (a savings product). Only about **12 out of every 100** people say yes.

Calling everyone is expensive. So the call centre wants a **ranked list**: call the people most likely to say yes first.

Your job:

1. Build several prediction models that score each client.
2. Compare them fairly (same tuning budget, same test data).
3. Pick one model and tell the manager what results to expect.
4. Watch out for common traps (especially using information you would **not** have before the call).

**Bottom line:** recommend which model should rank the call list, and how much better the top of that list should be than calling people at random.

---

## The real-world story

| Piece | Meaning |
|---|---|
| **Clients** | People the bank already contacted in past campaigns |
| **Features / inputs** | Facts about each client (age, job, balance, past contacts, …) |
| **Label / target** | Did they subscribe? Yes = 1, No = 0 |
| **Model** | A program that learns patterns from past data and scores new clients |
| **Call list** | Clients sorted by the model’s score (highest first) |

Data size: about **45,211** past contacts, **16** attributes, rows in **time order** (older → newer).

---

## What you must do (5 steps)

### 1. Protocol (rules of the game)
Decide:
- Which columns the model may use **before** calling (do not use “cheating” columns)
- How to split data into train / validation / test
- How you will tune hyperparameters

### 2. Compare models (fair fight)
Train and tune:
- **Logistic regression** (baseline = simple starting point)
- **Random forest**
- **HistGradientBoosting** (and/or LightGBM / XGBoost)
- **MLP** (a neural network)

Each family gets the **same number of hyperparameter tries**.

Score them with:
- **ROC AUC** (overall ranking quality)
- **Lift in top 10%** (how much better the best 10% of scored clients are vs random)

### 3. Uncertainty
On a held-out test set, estimate whether your best model is really better than the baseline — not just lucky. Use a **paired bootstrap confidence interval**.

### 4. Obstacle: next campaign
Past campaigns may differ from future ones. Check whether your evaluation still makes sense for the **next** campaign (time order / drift), and quantify what you find.

### 5. Decision
Write a short memo for the manager: deploy which model, expected lift, main risk.

---

## Glossary (terms you will meet)

### Core machine learning ideas

#### Machine learning (ML)
Teaching a computer to find patterns in data instead of writing every rule by hand.

**Example:** Instead of “if age > 60 and balance > 5000 then call,” the model learns which combinations of age, job, balance, etc. predict “yes.”

#### Supervised learning
You have inputs **and** the correct answers (labels). The model learns to map inputs → labels.

**Example:** Age, job, balance → “subscribed or not.”

#### Classification
Predicting a **category**, not a continuous number.

**Example:** Subscribe = yes/no (this homework). Predicting house price would be regression instead.

#### Binary classification
Exactly **two** classes.

**Example:** 1 = subscribed, 0 = did not subscribe.

#### Feature (also: attribute, input, column, `X`)
A piece of information used to make a prediction.

**Example:** `age`, `job`, `balance`, `month`.

#### Label / target (also: `y`)
The thing you want to predict.

**Example:** Whether the client subscribed to the term deposit.

#### Positive class / positive rate
The “interesting” class (here: subscribed = 1). Positive rate = fraction that are positive.

**Example:** Positive rate ≈ 0.117 means about 11.7% subscribed.

#### Tabular data
Data in a table: rows = examples, columns = features.

**Example:** One row per client; columns like age, job, balance.

#### Training
Fitting the model on data so it learns patterns.

**Example:** Show the model 30,000 past clients and their outcomes.

#### Prediction / score / `predict_proba`
After training, the model outputs a score (often a probability) for a new client.

**Example:** Client A gets score 0.82 (“likely yes”); Client B gets 0.05 (“likely no”). Rank by score for the call list.

---

### Data pitfalls

#### Leakage (data leakage)
Using information that would **not be available at decision time**. The model looks amazing in testing but fails in real life.

**Example in this homework:** `duration` = how long the call lasted. You only know that **after** the call. Using it to decide **who to call** is cheating. That column is the classic leak here.

#### Decision time
The moment you must choose who to call — **before** the call happens.

**Rule of thumb:** If you would not know a value yet when building the call list, do not use that feature.

#### Chronological order
Rows sorted by time (older first, newer last).

**Why it matters:** Random splits can mix future into training. For “will this work next month?”, a time-aware split is more honest.

#### Next-campaign obstacle / distribution shift / drift
The future campaign may not look like the past one (different months, economic conditions, client mix). A model that scores well on a random test set might do worse later.

**Example:** Train on early months, test on the last 20% of rows in time, and compare AUC to your normal test AUC.

---

### Splitting and evaluation

#### Train / validation / test split
- **Train:** learn patterns
- **Validation:** pick hyperparameters / choose among models
- **Test:** final honest score (look once)

**Example:** 70% train, 15% validation, 15% test — or train+val with cross-validation, then a final held-out test.

#### Held-out data
Data the model never saw during training or tuning. Used for the final check.

#### Cross-validation (CV)
Split training data into folds; train on some folds, validate on the left-out fold; rotate. Gives a more stable estimate while tuning.

**Example:** 5-fold CV = five rounds, each time 1/5 of the data is the validation fold.

#### Baseline
A simple model you must beat. If a fancy model is not clearly better, prefer the simple one.

**Example:** Logistic regression is the required baseline here.

#### Metric
A number that measures how good predictions are.

#### ROC AUC
A score from 0 to 1 for ranking quality.
- **0.5** ≈ random guessing
- **1.0** ≈ perfect ranking
- Higher is better

**Intuition:** Probability that a random “yes” client gets a higher score than a random “no” client.

#### Lift (top 10%)
How much denser “yes” clients are in the top 10% of your ranked list vs overall.

**Example:** Overall subscription rate = 12%. In your top 10% by score, rate = 36%.  
Lift = 36% / 12% = **3.0** → the top of your list is 3× better than calling at random.

#### Class imbalance
One class is much rarer than the other.

**Example:** ~12% yes, ~88% no. Accuracy alone is misleading (“always predict no” is ~88% accurate but useless). That is why AUC and lift matter more.

#### Chance-level on permuted labels
Shuffle the yes/no labels and re-evaluate. A leak-free model should fall to ~random performance (~0.5 AUC). If it still looks great, something is wrong.

---

### Models you will compare

#### Logistic regression
A linear classifier: combines features with weights, then squeezes the result into a probability.

**Pros:** Simple, fast, strong baseline, easy to interpret roughly.  
**Cons:** Struggles if relationships are complex and nonlinear (unless you engineer features).

#### Decision tree (idea behind forests / boosting)
Asks yes/no questions on features (“balance > 1000?”) and ends in a prediction.

#### Random forest
Many trees trained on random subsets; average their votes/scores.

**Pros:** Strong on tabular data, handles nonlinear patterns.  
**Cons:** Less “smooth” probabilities; needs careful encoding of categories sometimes.

#### Gradient boosting (HistGradientBoosting, LightGBM, XGBoost)
Builds trees **one after another**; each new tree tries to fix the previous mistakes.

**Pros:** Often excellent on tabular data.  
**Cons:** Can overfit if not tuned; more knobs to set.

#### MLP (Multi-Layer Perceptron) / neural network
Layers of neurons that learn nonlinear transformations of the inputs.

**Pros:** Flexible.  
**Cons:** Needs scaling, careful tuning, often weaker “out of the box” on small/medium tabular data than boosting.

#### Pipeline
A single object that chains preprocessing + model, so the same steps always happen together (fit on train, transform on test).

**Example:** Encode categories → scale numbers → logistic regression.

---

### Preprocessing

#### Encoding (categorical features)
Turn text categories into numbers models can use.

**Example:** `job = "technician"` → one-hot columns or integer codes, depending on the model family.

#### One-hot encoding
Make a 0/1 column per category.

**Example:** `marital`: married / single / divorced → three columns, one of them is 1.

#### Scaling
Rescale numeric features so they are on similar ranges (important for logistic regression and MLPs; trees often care less).

**Example:** StandardScaler turns each column roughly into mean 0, std 1.

#### Why “per-family pipelines”?
Trees, linear models, and neural nets need different preprocessing. You build a suitable pipeline for each family.

---

### Tuning and fairness

#### Hyperparameters
Settings you choose **before** training (not learned from the data directly).

**Example:** Number of trees, learning rate, network width, regularization strength.

#### Hyperparameter tuning
Trying different settings and keeping the ones that work best on validation data.

#### Grid search
Try every combination from a fixed menu of values.

#### Random search
Try random combinations from ranges — often more efficient when there are many hyperparameters.

#### Equal budget
Each model family gets the **same number of configurations** tried. Otherwise “model A is better” might just mean “we tuned A harder.”

**Example:** 30 random configs for logistic regression, 30 for random forest, 30 for boosting, 30 for MLP.

#### Overfitting
Memorizing training quirks instead of learning general patterns. Looks great on train, worse on new data.

#### Underfitting
Model is too simple to capture real patterns. Weak on both train and test.

---

### Uncertainty and comparison

#### Confidence interval (CI)
A range that likely contains the true difference (given the data and method).

**Example:** Chosen model AUC − baseline AUC = 0.04, 95% CI = [0.01, 0.07].  
If the interval is entirely above 0, evidence favors the chosen model.

#### Paired bootstrap
Resample the **same** test rows many times; each time compute metric(A) − metric(B). That keeps the comparison paired (same clients for both models).

**Why “paired”?** Both models scored the same people, so you measure the difference more fairly.

#### Seed / fixed seeds
A starting number for random number generators so runs are reproducible.

**Example:** `SEED = 0` so your notebook results match when re-run.

---

### Experiment tracking

#### MLflow
A tool that logs each experiment run: parameters, metrics, run IDs.

**Why here?** Every reported number must link to an MLflow run ID so results are traceable and reproducible.

#### Run ID
A unique ID for one logged training/evaluation experiment.

---

### Submission-specific terms

#### `FEATURES`
The list of input columns your final model uses (no leak column).

#### `CHOSEN_MODEL`
Which family you pick, e.g. `"hist_gradient_boosting"` or `"logistic_regression"`.

#### `build_pipeline()`
Function that returns an **unfitted** sklearn-style pipeline with your chosen, tuned settings. The grader fits it on its own data.

#### `answers.json`
JSON file with your findings: leak column, AUCs, CI, memo to the manager, etc.

#### Memo fields
- **decision:** deploy the model / deploy baseline / do not deploy  
- **expected_lift_top10:** lift you expect in the top 10%  
- **main_risk:** e.g. campaign conditions change, leakage, imbalance, data quality  
- **justification:** ≤150 words for a non-ML manager

#### AI-use statement
A markdown cell saying which AI tools you used (or “None”), for what, and how you checked the output. Missing it costs points.

---

## Tiny mental checklist before you submit

1. Did I drop features that are only known **after** the call (especially `duration`)?
2. Did every model family get the **same tuning budget**?
3. Did I evaluate the final comparison on **held-out** data?
4. Do I have a **confidence interval** for chosen − baseline?
5. Did I check a **time-forward / next-campaign** style evaluation?
6. Are reported numbers logged in **MLflow** with matching run IDs?
7. Does the notebook run top-to-bottom with fixed seeds in under ~20 minutes?

---

## One-sentence summary

**Teach models to rank which clients to call first, compare them fairly without cheating features, prove the winner is really better than a simple baseline, and warn the manager if the next campaign might look different.**
