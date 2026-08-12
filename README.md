# Ensemble CNN for Fashion-MNIST Classification

An ensemble of 5 Convolutional Neural Networks (CNNs), each trained on a
bootstrap resample of the Fashion-MNIST training data, compared against a
single CNN trained on the same data.

## Dataset

**Fashion-MNIST** — 28x28 grayscale images of clothing items across 10
classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
Sneaker, Bag, Ankle boot).

> **Note on loading the data:** `keras.datasets.fashion_mnist.load_data()`
> normally downloads the dataset from `storage.googleapis.com`, which isn't
> reachable from the network this project was built in. Instead, the exact
> same four IDX files Keras would download are included in this repo
> (sourced
> from the official [Zalando Research Fashion-MNIST
> repo](https://github.com/zalandoresearch/fashion-mnist)) and loaded via
> `data_loader.py`, a small IDX-format reader that returns data in the
> identical shape/dtype as `keras.datasets.fashion_mnist.load_data()`:
> `(x_train, y_train), (x_test, y_test)`. If you have unrestricted internet
> access, you can swap in `keras.datasets.fashion_mnist.load_data()`
> directly with no other code changes needed.
>
> **Note on file sizes:** the full Fashion-MNIST training-images file is
> ~26MB, just over GitHub's 25MB web-upload limit. Since the task only uses
> the first 50 records anyway, the `.gz` files in this repo are pre-trimmed
> to contain just those 50 train / 50 test records (each under 25KB) rather
> than the full 60,000 / 10,000. `data_loader.py` reads them exactly the
> same way either way — the trimming only removes images/labels the script
> would never touch.

## Data Preprocessing

Per the task spec, only the **first 50 records** of the training and test
sets are used (not the full 60,000 / 10,000):

1. `x_train_full[:50]`, `y_train_full[:50]` and `x_test_full[:50]`,
   `y_test_full[:50]` are selected.
2. Pixel values are normalized from `[0, 255]` to `[0, 1]`.
3. Images are reshaped to `(28, 28, 1)` for CNN input.
4. The 50 training records are split into **40 train / 10 validation**
   (80/20 split via `train_test_split`).
5. The 50 test records are kept aside as a held-out test set.

Because only 40 images cover 10 classes (~4 images per class on average),
this is an intentionally tiny, fast-to-run dataset — expect noisy,
lower-than-usual accuracy compared to training on the full 60k images.

## Model Architecture

A single small CNN, per the task spec:

```
Input (28, 28, 1)
  -> Conv2D(32 filters, 3x3 kernel, ReLU activation)
  -> MaxPooling2D(2x2 pool size)
  -> Flatten
  -> Dense(10 units, softmax activation)
```

Compiled with the **Adam** optimizer and **sparse categorical
crossentropy** loss.

## Ensemble Approach

- **5 CNN models** are trained, each with the architecture above.
- Each model is trained on its own **bootstrap sample**: the 40 training
  images resampled with replacement to another set of 40 (some images
  repeated, some left out) — a classic bagging approach.
- Each model trains for **3 epochs**.
- Predictions are made by running all 5 models on the validation/test sets
  and **averaging their predicted class probabilities**; the final
  prediction is the class with the highest averaged probability.

## Single Model (Baseline)

The same CNN architecture is trained once on the full 40-image training
set (no bootstrapping) for 3 epochs, as a baseline for comparison.

## Results

| Model            | Validation Accuracy | Test Accuracy |
|------------------|---------------------|----------------|
| Ensemble (5 CNNs)| 0.40                | 0.36           |
| Single CNN       | 0.50                | 0.46           |

Full numeric results (including config) are in `results_summary.json` and
`results_summary.txt`.

### Comparison

With only 40 training images across 10 classes, results are noisy and
sample-size-dependent — this run shows the single model outperforming the
ensemble on both validation (+0.10) and test (+0.10).
This isn't a stable trend; it reflects how volatile accuracy is at this
data scale, where each class has only a handful of examples and a single
image or two flipping right/wrong changes accuracy by several percentage
points. Ensembling with bagging generally helps stabilize variance and
reduce overfitting to any one training draw, but its benefit becomes
visible only with enough data/iterations for that variance-reduction
effect to average out — 50 total records isn't enough to reliably show it.
At full scale (60,000 training images), an ensemble of CNNs like this would
be expected to outperform a single CNN more consistently, at the cost of
~5x the training time and inference compute.

## Files in This Repo

- `ensemble_cnn.py` — main script (run this)
- `data_loader.py` — loads Fashion-MNIST from local IDX files
- `requirements.txt`
- `README.md`
- `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
  `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` — Fashion-MNIST
  data files (gzip, IDX format)
- `accuracy_comparison.png` — ensemble vs. single model bar chart
- `confusion_matrix_ensemble_test.png`
- `confusion_matrix_single_test.png`
- `sample_predictions_ensemble.png`
- `sample_predictions_single.png`
- `results_summary.json`
- `results_summary.txt`

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python ensemble_cnn.py
```

`data_loader.py` reads the four `.gz` files from the same directory as the
script, so just keep everything in one folder as uploaded. Output charts
and result files are written to an `output/` subfolder, which the script
creates automatically if it doesn't exist.
