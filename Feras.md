# Notebook overview

This notebook builds and compares two binary intrusion-detection models:

1. A centralized neural network trained on all training data.
2. A federated neural network trained across five simulated factories using Flower’s FedAvg strategy.

The target is `Attack_label`:

- `0` = normal network traffic
- `1` = attack traffic

The main concern is data leakage: preventing the model from learning capture identifiers, attack names, timestamps, duplicated packets, or other shortcuts that would create unrealistically high accuracy.

The pipeline is:

```text
2,219,201 raw rows
        ↓
Create capture groups and normalize values
        ↓
Remove contaminated features and duplicates
        ↓
15,827 clean, unique rows
        ↓
Group-disjoint train/validation/test split
        ↓
Balanced 10% sample: 1,582 rows
        ↓
Training-only preprocessing
        ↓
Centralized DNN ───────── Federated DNN
        ↓                       ↓
        └──── Compare on the same test set ────┘
```

---

## Cell 1 — Notebook title and purpose

### Main goal

This markdown cell states what the notebook is trying to achieve.

It builds:

- A centralized binary DNN.
- A five-client federated DNN.
- An IID Flower/FedAvg simulation.
- A new data-cleaning pipeline designed to reduce leakage.

The statement that no artifacts from the earlier contaminated experiment are loaded is important. It means old preprocessors, feature lists, or trained models cannot silently influence this experiment.

### Connection

The remaining cells implement the clean pipeline described here.

---

## Cell 2 — Imports, CUDA, and configuration heading

### Main goal

This markdown cell introduces the setup stage.

It explains an important device decision:

- Centralized training may use the GPU.
- Federated clients use the CPU.

The federated clients run sequentially, so this avoids unnecessary GPU competition and makes the simulation more stable on a laptop.

### Connection

Cell 3 performs this configuration.

---

## Cell 3 — Imports, paths, experiment settings, and reproducibility

### Main goal

This cell prepares the Python environment and centralizes all experiment settings.

### Important imports

The imports can be divided into groups:

- Data handling: `numpy`, `pandas`
- Machine learning preprocessing: `SimpleImputer`, `StandardScaler`
- Leakage checking: `DecisionTreeClassifier`
- Metrics: F1, precision, recall, ROC-AUC, PR-AUC
- Deep learning: `torch`, `torch.nn`
- Saving artifacts: `json`, `joblib`, `Path`
- Plotting: `matplotlib`

### Project-path detection

```python
HERE = Path.cwd().resolve()
```

This gets the directory from which the notebook is running.

The following block allows the notebook to be run from either:

- The experiment root.
- Its `notebooks` directory.

```python
if (HERE / 'notebooks' / '02_simple_clean_federated_pipeline.ipynb').exists():
    PROJECT_ROOT = HERE
elif HERE.name == 'notebooks' and (HERE / '02_simple_clean_federated_pipeline.ipynb').exists():
    PROJECT_ROOT = HERE.parent
else:
    raise RuntimeError(...)
```

If the notebook is started from an unexpected location, it stops rather than constructing incorrect paths.

### Dataset and output paths

```python
DATA_PATH = Path(os.environ.get(
    'EDGEIIOT_DATASET',
    REPO_ROOT / 'data' / 'raw' / 'DNN-EdgeIIoT-dataset.csv'
))
```

This means:

- If an `EDGEIIOT_DATASET` environment variable exists, use that path.
- Otherwise, use the repository’s default dataset path.

Separate clean output directories are created:

```python
ARTIFACTS = PROJECT_ROOT / 'artifacts_clean'
RESULTS = PROJECT_ROOT / 'results_clean'
```

Artifacts contain reusable objects such as models and preprocessors. Results contain metrics, tables, and summaries.

### Configuration

The `CONFIG` dictionary contains the experiment’s important settings:

```python
'seed': 42
'sample_fraction': 0.10
'train_fraction': 0.70
'val_fraction': 0.15
'test_fraction': 0.15
'batch_size': 256
```

The neural network is configured as:

```python
'hidden_layers': [128, 64, 32]
'dropout': [0.20, 0.10]
'learning_rate': 0.001
```

Federated learning uses:

```python
'local_epochs': 2
'federated_rounds': 10
'num_clients': 5
```

Each client is given a readable simulated-factory name.

### Reproducibility

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

Random operations affect:

- Sampling
- Client partitions
- Neural-network initialization
- Batch shuffling
- Dropout

Using the same seed makes repeated runs more reproducible.

### Device selection

```python
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

The saved output confirms centralized training used:

```text
NVIDIA GeForce RTX 3050
Centralized device: cuda
```

### Connection

After confirming that the environment and paths are correct, the notebook can safely load the dataset.

---

## Cell 4 — Dataset-loading heading

### Main goal

This markdown cell explains that the entire CSV must be loaded before sampling.

This is important because grouping and duplicate removal should happen before drawing the 10% sample. Sampling first could allow related or duplicated packets to be distributed incorrectly.

It also makes clear that only `Attack_label` is the prediction target. `Attack_type` is not a model target.

### Connection

Cell 5 loads and validates the complete dataset.

---

## Cell 5 — Load and validate the dataset

### Main goal

This cell reads the CSV and checks basic assumptions about the target.

### Important lines

```python
assert DATA_PATH.exists(), DATA_PATH
```

This stops immediately if the dataset path is wrong.

```python
raw = pd.read_csv(DATA_PATH, low_memory=False)
```

`low_memory=False` asks pandas to inspect columns more consistently instead of inferring data types from small chunks.

```python
assert 'Attack_label' in raw and 'Attack_type' in raw
```

The pipeline needs:

- `Attack_label` for model training.
- `Attack_type` for capture-group construction.

```python
assert set(raw['Attack_label'].dropna().unique()) == {0, 1}
```

This confirms that the task is binary and that the target contains exactly `0` and `1`.

### Output

The dataset contains:

- 2,219,201 rows
- 63 columns
- 1,615,643 normal packets
- 603,558 attack packets

The raw dataset is therefore imbalanced: normal traffic is much more common.

### Connection

The next stage constructs groups before removing capture-related fields.

---

## Cell 6 — Normalization and capture-group heading

### Main goal

This cell explains two upcoming operations:

1. Related packets are placed into capture groups.
2. inconsistent textual values are normalized.

Groups must be created before identifiers such as `Attack_type` and `frame.time` are removed, because those columns are needed to construct the groups.

### Why `Attack_type` can be used for grouping

`Attack_type` describes the attack category and is strongly connected to the target. Feeding it into the model would make prediction trivial and cause leakage.

Using it only to keep related capture sections together is different. It influences where rows are assigned, but it is removed before training.

### Connection

Cell 7 implements grouping and normalization.

---

## Cell 7 — Construct capture groups and normalize values

### Main goal

This cell creates approximate capture-session groups and cleans inconsistent column values.

## Capture-group construction

```python
parts = df['frame.time'].astype('string').str.extract(
    r'(\d{2}):(\d{2}):(\d{2})'
)
```

This extracts hours, minutes, and seconds from `frame.time`.

The result is converted into seconds since midnight:

```python
seconds = hours * 3600 + minutes * 60 + seconds
```

### Detecting new capture runs

```python
new_run = (
    df['Attack_type'].ne(df['Attack_type'].shift())
    | (seconds.notna() & seconds.shift().notna() & (delta < -300))
)
```

A new run begins when:

- `Attack_type` changes, or
- Time moves backwards by more than five minutes.

A backwards time jump can indicate that the dataset has moved to a new capture block.

```python
run_id = new_run.cumsum()
```

Every `True` in `new_run` increases the run number.

### Five-minute windows

```python
time_window = (seconds // 300).fillna(-1).astype('int64')
```

There are 300 seconds in five minutes. Integer division therefore maps packets into five-minute time windows.

### Bounded chunks

```python
chunk = base.groupby(base, sort=False).cumcount() // max_group_rows
```

If a group contains more than 10,000 rows, it is divided into smaller chunks. This prevents one enormous group from dominating a partition.

The final group identifier contains:

```text
Attack type + run number + five-minute window + chunk number
```

### Group-label assertion

```python
assert raw.groupby('_capture_group')['Attack_label'].nunique().max() == 1
```

This confirms that every capture group contains only one binary label. A group cannot contain both normal and attack rows.

## Semantic-value normalization

The cell defines common missing-value tokens:

```python
MISSING_TOKENS = {
    '', 'na', 'n/a', 'nan', 'none',
    'null', '?', 'inf', '+inf', '-inf'
}
```

Text values are stripped:

```python
s = out[col].astype('string').str.strip()
```

Missing tokens are converted to proper missing values, and equivalent representations of zero are changed to `"0"`.

### Numeric conversion

```python
numeric = pd.to_numeric(s, errors='coerce')
```

If at least 98% of a text column’s non-missing values can be converted to numbers, the entire column is treated as numeric:

```python
if nonmissing and ratio >= 0.98:
    out[col] = numeric.astype('float64')
```

Rare non-numeric values become missing values and will later be imputed.

Finally:

```python
out = out.replace([np.inf, -np.inf], np.nan)
```

Infinity is converted to missing data.

### Output

Ten text columns were found to be mostly numeric and converted. Examples include:

- `tcp.srcport`
- `dns.qry.name.len`
- `mqtt.conack.flags`

Some of these are later removed because being numeric does not mean they are safe model features.

### Connection

Now that groups exist and values have consistent representations, contaminated fields can be removed.

---

## Cell 8 — Feature-removal and deduplication heading

### Main goal

This markdown cell introduces the most aggressive cleaning stage.

It removes:

- Known label proxies.
- Host and capture identifiers.
- Raw payloads.
- Absolute sequence numbers.
- Unusable text columns.

Duplicates are handled only after the final candidate feature set has been selected. This matters because two packets may look different in removed identifier fields but be identical from the model’s perspective.

### Connection

Cell 9 performs these operations.

---

## Cell 9 — Remove contaminated features and deduplicate rows

### Main goal

This cell defines what the model is allowed to see and removes ambiguous or repeated feature patterns.

## Explicit feature removal

`DROP_REASONS` records both the column and the reason for removal.

Examples include:

```python
'Attack_type': 'attack name; grouping only'
'frame.time': 'capture timestamp; grouping only'
```

`Attack_type` would almost reveal the answer directly. `frame.time` could allow the model to memorize when attacks occurred.

Host and port identities are removed:

```python
'ip.src_host'
'ip.dst_host'
'tcp.srcport'
'tcp.dstport'
```

These could let the model memorize devices, addresses, or attack-specific capture settings instead of learning general traffic behaviour.

Raw content is also removed:

```python
'tcp.payload'
'mqtt.msg'
'http.file_data'
'http.request.full_uri'
```

Some protocol fields are explicitly known to contain source-format leakage:

```python
CONFIRMED_CONTAMINATED = {
    'mqtt.conack.flags',
    'mqtt.protoname',
    'mqtt.topic',
    'dns.qry.name.len',
    'http.request.version'
}
```

### Numeric-only teaching pipeline

Any remaining non-numeric feature is removed:

```python
if not pd.api.types.is_numeric_dtype(normalized[col]):
    normalized = normalized.drop(columns=col)
```

This simplifies preprocessing and avoids high-cardinality categorical features.

Constant or completely missing columns are also removed because they provide no predictive information.

### Safety assertions

```python
assert 'Attack_type' not in feature_cols
assert CONFIRMED_CONTAMINATED.isdisjoint(feature_cols)
assert all(pd.api.types.is_numeric_dtype(...) ...)
```

These assertions ensure that known contaminated fields cannot accidentally reach the model.

The final model has 25 features.

## Conflicting-label removal

```python
feature_hash = pd.util.hash_pandas_object(
    normalized[feature_cols], index=False
)
```

Each row’s final 25-feature pattern is represented by a hash.

The code then checks whether the same feature pattern appears with both labels:

```python
label_span = ...groupby('h')['y'].agg(['min', 'max'])
```

If `min != max`, the same input pattern has been labeled both normal and attack.

All rows belonging to those ambiguous patterns are removed:

```python
clean = normalized.loc[~ambiguous_mask].copy()
```

Why? Given identical model inputs, a deterministic classifier cannot know which label is correct. Retaining these rows would create contradictory training examples.

## Same-label duplicate removal

```python
duplicate_mask = clean.duplicated(
    subset=feature_cols, keep='first'
)
```

If identical model inputs have the same label, one copy is retained.

This also prevents large numbers of duplicates from appearing across the train and test sets and inflating performance.

### Output

The cleaning is extremely aggressive:

- 1,628,811 conflicting-label rows removed.
- 574,563 same-label duplicates removed.
- 15,827 clean, unique rows remain.
- 25 final features remain.

All 2,219,201 raw rows are accounted for by these three categories.

This large reduction is important when interpreting the final accuracy: the model is evaluated on a much smaller, cleaner subset of the original dataset.

### Connection

The clean rows can now be split without allowing related capture groups to cross partitions.

---

## Cell 10 — Group-aware splitting heading

### Main goal

This markdown cell explains that entire capture groups—not individual packets—are assigned to train, validation, and test sets.

If individual packets were randomly split, very similar packets from the same capture could appear in both training and testing. The model might then appear to generalize when it is effectively recognizing the same capture.

### Connection

Cell 11 performs the group-level split.

---

## Cell 11 — Create train, validation, and test partitions

### Main goal

This cell creates group-disjoint 70/15/15 partitions.

### Group summary

```python
group_table = clean.groupby('_capture_group').agg(
    label=('Attack_label', 'first'),
    rows=('Attack_label', 'size')
)
```

Each row in `group_table` represents one capture group and records:

- Its binary label.
- Its number of rows.

### First split

```python
train_groups, temp_groups = train_test_split(
    all_groups,
    test_size=0.30,
    stratify=group_table.loc[all_groups, 'label']
)
```

Seventy percent of groups go to training. Thirty percent temporarily remain for validation and testing.

`stratify` tries to preserve the normal/attack group proportions.

### Second split

```python
val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    stratify=...
)
```

The remaining 30% is divided equally:

- 15% validation.
- 15% test.

### No-overlap assertions

```python
assert train_groups.isdisjoint(validation_groups)
```

Equivalent checks are performed for every pair of partitions.

These assertions are central to the notebook’s leakage-control claim.

### Output

The group allocation is:

- Training: 71 groups and 10,337 rows.
- Validation: 15 groups and 3,923 rows.
- Test: 16 groups and 1,567 rows.

The row counts are not exactly 70/15/15 because the code splits groups, and groups have different sizes. Group separation is more important here than exact row ratios.

### Connection

The next cell samples and balances rows within these already-separated partitions.

---

## Cell 12 — Balanced-sampling heading

### Main goal

This markdown cell describes the 10% balanced sample.

Sampling happens after group assignment so it cannot change which partition owns a group.

Each sampled partition is intended to contain:

- 50% normal rows.
- 50% attack rows.

### Connection

Cell 13 creates the sample.

---

## Cell 13 — Create a balanced 10% sample

### Main goal

This cell reduces training cost while creating equally balanced classes.

### Target size

```python
target_total = int(round(
    len(clean) * CONFIG['sample_fraction']
))
```

Ten percent of 15,827 is approximately 1,583.

The target is divided approximately 70/15/15 between the three partitions.

### Balanced sampling function

```python
normal = frame[frame['Attack_label'] == 0]
attack = frame[frame['Attack_label'] == 1]
```

The two classes are separated.

```python
n_each = min(
    target_rows // 2,
    len(normal),
    len(attack)
)
```

The sample uses the same number from each class, limited by whichever class has fewer available rows.

Rows are sampled without replacement and then shuffled:

```python
pd.concat([
    normal.sample(n_each),
    attack.sample(n_each)
]).sample(frac=1)
```

### Why the final total is 1,582

The requested total is approximately 1,583, but a perfectly balanced binary sample must have an even number of rows. The validation target is odd, so integer division produces one fewer row.

Final sample:

- Training: 1,108 rows, 554 per class.
- Validation: 236 rows, 118 per class.
- Test: 238 rows, 119 per class.
- Total: 1,582 rows.

### Sampled groups

Not every assigned group appears in the sample because only a small number of rows is selected:

- 59 of 71 training groups appear.
- 11 of 15 validation groups appear.
- 12 of 16 test groups appear.

### Final overlap check

The code hashes final feature rows in each split and verifies that identical feature vectors do not cross partitions.

It therefore confirms:

- Zero capture-group overlap.
- Zero exact final-feature overlap.

### Connection

The selected model features can now be audited and preprocessed.

---

## Cell 14 — Leakage-audit and preprocessing heading

### Main goal

This markdown cell introduces:

1. A simple per-feature leakage test.
2. Median imputation.
3. Feature standardization.

All learned preprocessing values must come from training data only.

### Connection

Cell 15 performs this audit and preprocessing.

---

## Cell 15 — Check individual features and preprocess the data

### Main goal

This cell checks whether any single feature almost predicts the target and converts the three splits into model-ready arrays.

## Separate features and labels

```python
X_train_raw = train_df[feature_cols].copy()
y_train = train_df['Attack_label'].to_numpy(np.float32)
```

Equivalent arrays are created for validation and testing.

The `X` variables contain features. The `y` variables contain labels.

## Decision-stump leakage audit

For every feature, the code trains a one-level decision tree:

```python
DecisionTreeClassifier(
    max_depth=1,
    min_samples_leaf=50
)
```

A depth-one tree makes only one decision, such as:

```text
Is tcp.len <= some threshold?
```

If one simple threshold achieves nearly perfect training accuracy, that feature may directly encode the label.

The flag threshold is:

```python
accuracy >= 0.995
```

This audit uses only training data, which prevents test information from influencing feature review.

### Output

The strongest single feature was `dns.qry.name`, with 93.95% training accuracy.

No feature reached 99.5%, so no feature was flagged as a near-perfect individual proxy.

This does not prove that no leakage remains. Multiple features could still combine into a powerful shortcut. It is a useful diagnostic rather than a guarantee.

## Preprocessing pipeline

```python
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

The imputer replaces missing values with each feature’s training median.

The scaler transforms features approximately as:

```text
scaled value = (original value - training mean) / training standard deviation
```

Standardization helps neural-network optimization because features measured on very different scales become more comparable.

### Training-only fitting

```python
X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)
```

This distinction is crucial:

- `fit_transform` learns medians, means, and standard deviations.
- `transform` only applies previously learned values.

The validation and test sets do not determine any preprocessing statistics.

### Saved artifacts

The cell saves:

- `preprocessor.joblib`
- `feature_list.json`
- `dropped_columns.csv`

### Output

Final shapes are:

- Training: `(1108, 25)`
- Validation: `(236, 25)`
- Test: `(238, 25)`

All missing values and infinities have been removed from the processed arrays.

### Federated-learning caveat

This preprocessing is leakage-safe with respect to validation and test data, but it is centrally fitted using the combined training set. A privacy-preserving federated system would need distributed methods for calculating medians, means, and variances.

### Connection

The processed arrays are now ready for neural-network training.

---

## Cell 16 — Centralized DNN heading

### Main goal

This markdown cell introduces the centralized baseline.

The default neural network is trained first. Two alternative configurations are only tried if default validation accuracy is below 96%.

The model is selected using validation F1, and the untouched test set is used only after selection.

### Connection

Cell 17 defines and trains the model.

---

## Cell 17 — Define and train the centralized neural network

### Main goal

This cell defines the neural network, evaluation metrics, training loop, early stopping, and validation-based model selection.

## Neural-network architecture

```python
class BinaryMLP(nn.Module):
```

MLP means multilayer perceptron: a standard fully connected neural network.

The selected default architecture is approximately:

```text
25 inputs
  ↓
128 neurons + ReLU + 20% dropout
  ↓
64 neurons + ReLU + 10% dropout
  ↓
32 neurons + ReLU
  ↓
1 output logit
```

### Linear layers

```python
nn.Linear(previous, width)
```

Every neuron receives information from all outputs of the previous layer.

### ReLU

```python
nn.ReLU()
```

ReLU introduces nonlinearity:

```text
ReLU(x) = max(0, x)
```

Without nonlinear activation functions, multiple linear layers would behave like one linear transformation.

### Dropout

```python
nn.Dropout(dropout[i])
```

Dropout randomly disables some neurons during training. This reduces reliance on particular neurons and can help prevent overfitting.

### Final output

```python
layers.append(nn.Linear(previous, 1))
```

The model produces one raw number called a logit. It does not directly produce a probability.

## Data loader

`loader_for` turns NumPy arrays into PyTorch tensors and divides them into batches of 256 rows.

Training batches are shuffled; validation and test batches are not.

## Metrics

`metric_bundle` converts probabilities into class predictions:

```python
pred = (prob >= 0.5).astype(int)
```

A probability of at least `0.5` means attack.

The metrics mean:

- Accuracy: proportion of all correct predictions.
- Precision: among predicted attacks, how many were attacks.
- Recall: among real attacks, how many were detected.
- F1: harmonic balance between precision and recall.
- ROC-AUC: ranking quality across all possible thresholds.
- PR-AUC: precision-recall performance across thresholds.
- FPR: proportion of normal rows incorrectly flagged as attacks.
- FNR: proportion of attacks incorrectly classified as normal.

For an intrusion detector, false negatives are particularly important because they represent missed attacks.

## Logits, sigmoid, and BCE loss

The training loss is:

```python
nn.BCEWithLogitsLoss()
```

BCE means binary cross-entropy. This loss compares the true label with the model’s confidence.

`BCEWithLogitsLoss` expects raw logits and internally performs the numerically stable sigmoid calculation.

During evaluation:

```python
torch.sigmoid(logits)
```

converts logits into probabilities between zero and one.

## Evaluation mode

```python
@torch.no_grad()
def evaluate_array(...):
```

`no_grad` prevents PyTorch from storing gradient information during evaluation, saving memory and computation.

```python
model.eval()
```

Evaluation mode disables dropout.

## Training loop

The core update is:

```python
optimizer.zero_grad()
loss = criterion(model(xb), yb)
loss.backward()
optimizer.step()
```

This means:

1. Clear gradients from the previous batch.
2. Calculate predictions and loss.
3. Compute how parameters contributed to the error.
4. Update parameters with Adam.

## Validation and early stopping

After each epoch, the model is evaluated on validation data.

```python
if val_metrics['f1'] > best_f1:
    best_state = copy.deepcopy(model.state_dict())
```

The model parameters with the best validation F1 are saved in memory.

```python
if stale >= CONFIG['patience']:
    break
```

Training stops after five consecutive epochs without validation-F1 improvement.

## Candidate configurations

The notebook defines:

- `default`
- `wider`
- `slower_lr`

However, alternatives are only trained when default validation accuracy is below 96%.

### Output

The default model achieved:

- Validation accuracy: 99.58%.
- Validation F1: 99.57%.
- Epochs executed: 7.

Because it passed 96%, the wider and slower-learning-rate alternatives were not trained.

The selected model is saved as `centralized_model.pt`.

### Connection

Cell 18 evaluates the selected model once on the test set.

---

## Cell 18 — Final centralized test evaluation

### Main goal

This cell evaluates the finalized centralized model on the untouched test set.

```python
TEST_TOUCHED_FOR_FITTING = False
```

This variable documents that the test set was not intentionally used for fitting. It is a declaration and assertion, not an automatic tracker of every previous data access.

### Evaluation

```python
centralized_test = evaluate_array(
    centralized_model, X_test, y_test, DEVICE
)
```

Results are saved in `centralized_final_metrics.json`.

### Test results

There are 238 test rows: 119 normal and 119 attacks.

The confusion matrix is:

```text
                 Predicted
                 Normal   Attack
True Normal        118       1
True Attack          6     113
```

Therefore:

- True negatives: 118 normal packets correctly identified.
- False positives: 1 normal packet incorrectly flagged.
- False negatives: 6 attacks missed.
- True positives: 113 attacks detected.

Metrics:

- Accuracy: 97.06%.
- Precision: 99.12%.
- Recall: 94.96%.
- F1: 97.00%.
- ROC-AUC: 99.17%.
- PR-AUC: 99.35%.
- FPR: 0.84%.
- FNR: 5.04%.

Precision is higher than recall because the model produces very few false alarms but misses six attacks.

The loss can appear relatively high despite strong accuracy because BCE also measures confidence. A few confidently incorrect predictions can produce substantial loss.

### Connection

This becomes the baseline against which the federated model is compared.

---

## Cell 19 — Five IID clients heading

### Main goal

This markdown cell explains how the training rows will be distributed among five clients.

Only training data is partitioned. Validation and test data remain centralized for model selection and comparison.

IID means the clients are intended to have similar statistical distributions—in this case, especially similar class ratios.

### Connection

Cell 20 constructs these partitions.

---

## Cell 20 — Divide training data among five clients

### Main goal

This cell creates five disjoint, approximately equal, balanced training partitions.

### Class-by-class splitting

```python
for label in [0, 1]:
    idx = np.where(y_train.astype(int) == label)[0]
    rng.shuffle(idx)
```

Normal and attack indices are shuffled separately.

```python
np.array_split(idx, CONFIG['num_clients'])
```

Each class is divided into five pieces. Every client receives one normal piece and one attack piece.

This ensures approximately equal 50/50 class balance.

### Coverage and overlap assertions

```python
assert len(np.unique(all_client_idx)) == len(all_client_idx) == len(X_train)
```

This confirms:

- Every training row belongs to a client.
- No row belongs to more than one client.

### Architecture consistency

A model is constructed for every client and converted to a text representation:

```python
architecture_signatures.append(str(m))
assert len(set(architecture_signatures)) == 1
```

This checks that all clients use the same network structure, which is required for parameter averaging.

### Output

Client distributions are:

- Factory A: 222 rows, 111 normal and 111 attacks.
- Factory B: 222 rows, 111 normal and 111 attacks.
- Factory C: 222 rows, 111 normal and 111 attacks.
- Factory D: 222 rows, 111 normal and 111 attacks.
- Factory E: 220 rows, 110 normal and 110 attacks.

The last client is slightly smaller because 1,108 cannot be divided equally by five.

### Connection

These partitions become the local client datasets used during federated training.

---

## Cell 21 — Flower FedAvg heading

### Main goal

This markdown cell describes the federated-learning setup.

Important qualifications:

- All clients run sequentially.
- They run in one Python process.
- Client training happens on CPU.
- All five clients participate in every round.
- Preprocessing was performed centrally.

Therefore, this demonstrates federated optimization logic, but it is not a real distributed deployment.

### FedAvg concept

After local training, client model parameters are combined using a weighted average:

```text
global parameters =
    Σ(client parameters × client sample count)
    ─────────────────────────────────────────
              total sample count
```

Because the five clients have nearly equal sizes, their influence is also nearly equal.

### Connection

Cell 22 implements this process with Flower’s strategy and record APIs.

---

## Cell 22 — Run the Flower FedAvg simulation

### Main goal

This cell performs ten rounds of local client training and server aggregation.

## Parameter conversion

```python
def get_parameters(model):
    return [
        v.detach().cpu().numpy()
        for v in model.state_dict().values()
    ]
```

This extracts model weights and biases as NumPy arrays.

```python
def set_parameters(model, parameters):
```

This performs the reverse operation, loading received arrays into a PyTorch model.

The ordering of `state_dict()` keys must remain consistent, because each received array is matched with a parameter name by position.

## FedAvg strategy

```python
strategy = FedAvg(
    fraction_train=1.0,
    fraction_evaluate=0.0,
    min_train_nodes=5,
    min_available_nodes=5,
    weighted_by_key='num-examples'
)
```

This means:

- `fraction_train=1.0`: use all available clients.
- `fraction_evaluate=0.0`: Flower does not request client-side evaluation.
- Five clients must be available.
- Aggregate updates using each client’s `num-examples`.

Validation is instead performed centrally by `evaluate_global`.

## Global-model evaluation

```python
def evaluate_global(server_round, parameters):
```

This function:

1. Creates a new model.
2. Loads the current global parameters.
3. Evaluates on centralized validation data.
4. Records loss, accuracy, and F1.
5. Saves the best checkpoint.

```python
if server_round > 0 and metrics['f1'] > best_f1:
```

Round zero is not saved because it is the untrained initialization.

The comparison uses strict `>`, so if later rounds have equal F1, the earliest matching round remains the best checkpoint.

## Initial global model

```python
initial_model = BinaryMLP(...)
global_parameters = get_parameters(initial_model)
evaluate_global(0, global_parameters)
```

Round zero measures the randomly initialized model before federated training begins.

## Federated rounds

```python
for server_round in range(1, 11):
```

There are ten rounds.

Inside each round, every client performs the following:

### 1. Receive the global model

```python
set_parameters(local_model, global_parameters)
```

Every client starts the round from identical server parameters.

### 2. Train locally

```python
for _ in range(CONFIG['local_epochs']):
```

Each client trains for two local epochs using only its partition.

### 3. Package the update

```python
ArrayRecord(
    numpy_ndarrays=get_parameters(local_model)
)
```

The updated model parameters are placed in a Flower record.

The client also reports:

```python
'num-examples': len(idx)
'train-loss': loss_sum / seen
```

`num-examples` is what FedAvg uses for weighting.

### 4. Aggregate all clients

```python
aggregated, _ = strategy.aggregate_train(
    server_round, replies
)
```

Flower combines the five local updates.

```python
global_parameters = aggregated.to_numpy_ndarrays()
```

The aggregated result becomes the global model for the next round.

## Output

Federated validation progression:

- Round 0: 50% accuracy, F1 0.
- Round 1: 50% accuracy, F1 0.
- Round 2: 55.93% accuracy, F1 0.212.
- Round 3: 99.58% accuracy, F1 0.9957.
- Rounds 4–10: same threshold-based accuracy and F1.

Validation loss continues falling from round 3 through round 10, even though accuracy and F1 remain unchanged. This means predicted probabilities continue changing, but they produce the same decisions at the `0.5` threshold.

Round 3 is selected because it is the first round to achieve the best validation F1. Later ties do not overwrite it.

The simulation finishes in only 2.6 seconds because the sampled dataset and client models are small.

### Connection

The round-3 checkpoint is evaluated on the test set in Cell 24.

---

## Cell 23 — Federated evaluation heading

### Main goal

This markdown cell explains that the selected federated model will now be evaluated once on the untouched test set.

Both centralized and federated models use exactly the same test rows, allowing a direct comparison.

### Connection

Cell 24 loads the federated checkpoint and compares the results.

---

## Cell 24 — Evaluate and compare the federated model

### Main goal

This cell evaluates the best federated checkpoint and compares it with the centralized baseline.

### Load the checkpoint

```python
checkpoint = torch.load(
    FED_MODEL_PATH,
    map_location='cpu',
    weights_only=True
)
```

`map_location='cpu'` ensures that the checkpoint can be loaded without requiring a GPU.

The saved architecture information is used to reconstruct the correct model:

```python
federated_model = BinaryMLP(
    checkpoint['input_dim'],
    checkpoint['spec']['hidden'],
    checkpoint['spec']['dropout']
)
```

The saved round-3 parameters are then loaded.

### Test comparison

Both models achieve:

- Accuracy: 97.06%.
- Precision: 99.12%.
- Recall: 94.96%.
- F1: 97.00%.
- FPR: 0.84%.
- FNR: 5.04%.

Both also produce the same confusion matrix:

```text
[[118, 1],
 [  6, 113]]
```

Therefore, both models make exactly the same binary decisions at the `0.5` threshold.

The federated model has slightly higher ranking metrics:

- Centralized ROC-AUC: 99.167%.
- Federated ROC-AUC: 99.237%.
- Centralized PR-AUC: 99.349%.
- Federated PR-AUC: 99.431%.

This is possible because ROC-AUC and PR-AUC use probability rankings across many thresholds. Two models can make the same decisions at `0.5` while assigning somewhat different probabilities.

### Connection

The final metrics and experiment metadata are collected into a summary.

---

## Cell 25 — Artifact-saving heading

### Main goal

This markdown cell clarifies that all saved files belong to the new clean experiment.

Client arrays remain in memory. No client-shard cache is saved to disk.

This reduces the risk that future runs accidentally reuse stale client partitions.

### Connection

Cell 26 writes and displays the final summary.

---

## Cell 26 — Save the final run summary

### Main goal

This cell records the major experiment properties and final results.

The saved summary includes:

- Number of sampled rows.
- Number of model features.
- Grouping method.
- Group overlap.
- Centralized metrics.
- Federated metrics.
- Whether the 96% accuracy target was reached.

```python
'accuracy_target_reached': bool(
    max(
        centralized_test['accuracy'],
        federated_test['accuracy']
    ) >= 0.96
)
```

The target is considered reached if either model obtains at least 96% test accuracy.

### Important naming detail

```python
'clean_unique_rows': int(
    sum(len(x) for x in splits.values())
)
```

This records `1,582`, which is the size of the balanced sample.

The name `clean_unique_rows` is slightly misleading because there were actually 15,827 clean unique rows before sampling. A clearer name would be something like `sampled_clean_rows`.

This naming issue does not affect model training or evaluation.

### Saved outputs

Across the notebook, the important saved artifacts include:

- Preprocessor.
- Feature list.
- Dropped-column report.
- Centralized model.
- Best federated model.
- Centralized final metrics.
- Federated final metrics.
- Client distributions.
- Per-round federated metrics.
- Overall run summary.

### Output

The final headline results are:

```text
Balanced sample size: 1,582
Final features: 25
Group overlap: 0

Centralized:
accuracy = 0.9706
F1       = 0.9700

Federated:
accuracy = 0.9706
F1       = 0.9700
```

The 96% accuracy target was reached.

### Connection

The final markdown cell explains what these results do and do not prove.

---

## Cell 27 — Limitations

### Main goal

This cell prevents overinterpreting the experiment.

### IID clients from one dataset

The five factories are randomly created partitions from one laboratory dataset. They are not five genuinely different organizations or physical factories.

Real clients usually have non-IID data. For example:

- Different devices.
- Different network protocols.
- Different attack frequencies.
- Different feature distributions.
- Different amounts of data.

That would make federated training more difficult.

### Centralized preprocessing

The preprocessor is fitted on the combined training data. In a privacy-preserving deployment, raw training data would remain at each client, and preprocessing statistics would need to be calculated without central access.

### Capture groups are proxies

The constructed groups approximate capture sessions using attack type, time windows, time resets, and chunks. They are not guaranteed to represent genuine independent collection sessions.

### No privacy technology

The notebook does not implement:

- Differential privacy.
- Secure aggregation.
- Encryption of model updates.
- Protection against information leakage from gradients.
- Authentication or network security.
- Real client/server communication.

Federated averaging alone does not guarantee privacy.

### Lower score than contaminated experiments

A previous contaminated pipeline may have achieved higher performance because the model could learn identifiers or source-format shortcuts.

A lower but more realistic result is preferable to a higher result caused by leakage.

---

# Final interpretation

The notebook successfully demonstrates the mechanics of a leakage-aware centralized-versus-federated comparison:

- Known contaminated fields are removed.
- Exact duplicate and conflicting feature patterns are removed.
- Capture groups do not cross dataset partitions.
- Preprocessing is fitted only on training rows.
- Validation controls model and checkpoint selection.
- Test data is reserved for final evaluation.
- All five clients contain disjoint training rows.
- FedAvg combines client models using sample-count weighting.

Both trained models classify 231 of 238 test rows correctly:

- 118 normal rows correctly recognized.
- 113 attacks correctly detected.
- 1 false alarm.
- 6 missed attacks.

The main limitation is external validity: the result shows that this educational simulation works on a small balanced subset of one laboratory dataset. It does not yet show that the model would generalize to independent factories, non-IID clients, real federated infrastructure, or privacy-sensitive deployment.