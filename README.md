# Wine Classification using Perceptron and ANN

## Problem Statement

Given the chemical composition of a wine sample (13 numeric features like alcohol content, acid level, color intensity, etc.), predict which of 3 wine cultivars it belongs to.

This is a **multi-class classification** problem — each wine sample belongs to exactly one of 3 classes (0, 1, or 2).

## Dataset

- **Source:** `sklearn.datasets.load_wine()`
- **Samples:** 178
- **Features:** 13 (all numeric, chemical measurements)
- **Classes:** 3 (roughly 59 / 71 / 48 samples per class)
- **Missing values:** None — dataset is already clean

## Steps Followed

1. **Data Exploration**
   - Checked class distribution, feature ranges, and correlations
   - Found that features have very different scales (e.g. proline in hundreds, hue below 1)

2. **Preprocessing**
   - Split data into train (80%) and test (20%) sets, keeping class ratio equal using `stratify=y`
   - Scaled all features using `StandardScaler`, since models perform better and faster when features are on the same scale

3. **Baseline Model — Perceptron**
   - A simple linear model was trained first to see how well a basic approach performs
   - Result: 100% accuracy on test set, ~97.9% average accuracy across 5-fold cross-validation

4. **Deep Learning Model — ANN**
   - Built using Keras `Sequential` model with `Dense` and `Dropout` layers, softmax output for 3 classes
   - Result: 94.4% accuracy on test set, ~97.2% average accuracy across 5-fold cross-validation

5. **Evaluation**
   - Used `classification_report` and `confusion_matrix` to check precision, recall, and where each model made mistakes
   - Used cross-validation (not just one test split) to confirm results were reliable, not a lucky split

## Final Result

| Model      | Test Accuracy | Mean CV Accuracy |
|------------|---------------|-------------------|
| Perceptron | 100%          | 97.9%             |
| ANN        | 94.4%         | 97.2%             |

Both models perform almost equally well. The small difference in test accuracy disappeared once cross-validation was used, showing that a single train/test split can be misleading.

## Why Both Models Performed Similarly

The Wine dataset's 3 classes are **linearly separable** — meaning a simple straight-line boundary is enough to tell them apart. A Perceptron can only draw straight-line boundaries, and that turned out to be sufficient here. So the extra complexity of an ANN didn't add much benefit for this particular dataset.

## Real-World Relevance

This project is a small, clean dataset used for **learning and practicing** a machine learning pipeline: preprocessing, baseline modeling, deep learning, and proper evaluation.

In real-world problems, data is usually **not** this clean or linearly separable — for example:
- Sensor data from wearables (activity recognition)
- Images
- Text or speech data

In those cases, a simple model like Perceptron usually performs poorly because it can only draw straight-line boundaries. This is where ANNs (and deeper networks) show their real advantage — they can learn curved, complex, non-linear boundaries between classes.

**Key takeaway:** Model choice should match problem complexity. Simple problems don't need complex models; complex, real-world data usually does.

## Tools Used

- `pandas`, `numpy` — data handling
- `seaborn`, `matplotlib` — visualization
- `scikit-learn` — preprocessing, Perceptron, evaluation metrics, cross-validation
- `TensorFlow / Keras` — building and training the ANN
# Wine-Classification-using-Perceptron-and-ANN
