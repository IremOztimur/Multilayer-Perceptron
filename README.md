# Multilayer-Perceptron 42

The goal of this project is to implement a **_multilayer perceptron_**, in order to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis in the [Wisconsin](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names).

I will build a makefile so don't worry about compiling the project,

guideish structure

## Data Preparation
First things first, the given data is raw and should be preprocessed before being used for the training phase.
### To-Do:
- [X] Encode labels as 0 (B) or 1 (M)
- [ ] Normalize/scale features
- [ ] Split the dataset into:\
**Training set:** _Used for learning_.\
**Validation set:** _Used to evaluate performance on unseen data_.
