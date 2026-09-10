# GFCFA Face Recognition Model

Face recognition project that combines Gabor filter banks with the Cuttlefish
Optimization Algorithm (CFA). Gabor filters extract texture-based features from face
images, and CFA is used to pick the most effective subset of those features before
classification.

## Approach

1. Extract Gabor features from face images with a bank of filters at 5 scales and
   8 orientations.
2. Reduce the feature set with CFA, a bio-inspired optimization algorithm that searches
   for the subset of features that keeps accuracy high.
3. Classify with SVM and KNN, comparing accuracy with all features against accuracy
   using only the CFA-selected features.

## Datasets

Tested on three face datasets:

- ATT (AT&T)
- Georgia Tech
- Yale

## Usage

```bash
python main.py
```

The script runs in stages and asks before each step: reading the dataset, extracting
Gabor features, saving them as pickles, running SVM and KNN with all features, then
running the same classifiers on CFA-selected feature subsets.

## Results

Per-dataset outputs in `Results/` include confusion matrices, classification reports,
accuracy tables, and saved pickle files for a range of training image counts.

## Structure

```text
main.py                entry point that runs extraction, CFA, and classification
Code/
  GaborFeatures.py     Gabor filter bank and feature extraction
  CuttleFish.py        Cuttlefish Optimization Algorithm
  MyFRClassifiers.py   SVM and KNN wrappers with metrics
  Read_IMG.py          image loading
  PlotTable.py         accuracy table plots
  SaveMyPickles.py     pickle saving and loading
  message.py           prompts and helpers
Results/               outputs for each dataset
```