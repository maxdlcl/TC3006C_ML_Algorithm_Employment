# Decision Tree Classifier with Scikit-learn

---

This repository contains an implementation of a Decision Tree Classifier using Scikit-learn, evaluated on the [Palmer Penguins Dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris)

The dataset includes three classes (Adelie, Chinstrap, Gentoo) with numerical features (bill length, bill depth, flipper length, body mass) and categorical features (island, sex), for a total of 344 samples.

The implementation explores multiple models:

1. Default Decision Tree (parameters found with GridSearchCV).
2. Tree with high impurity threshold (example of underfitting).
3. Deep Tree (example of overfitting).
4. Bagging Classifier with Decision Trees (regularized ensemble).

After running the file, the following plots are displayed:

1. A pie chart showing the class distribution of the dataset.
2. Confusion matrices with the model’s performance on the validation set.
3. Confusion matrices with the model’s performance on the test set.

## Running the implementation

```
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
```

Then, clone the repository and execute the script:

```
git clone https://github.com/maxdlcl/TC3006C_ML_Algorithm_Employment
cd TC3006C_ML_Algorithm_Employment
python3 decision_tree_framework.py
```

## Submission info

* Maximiliano De La Cruz Lima
* A01798048
* Submission date: September 10th, 2025
