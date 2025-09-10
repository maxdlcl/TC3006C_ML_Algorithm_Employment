from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data():
    """Loads the Palmer Penguins dataset.

    Tries first through seaborn's `load_dataset('penguins')`. If that fails
    (e.g., offline or older seaborn), falls back to the raw CSV from GitHub.

    Returns:
        X (DataFrame): feature matrix containing all columns except 'species'.
        y (Series): label vector with penguin species.
        class_names (list): sorted list of observed class names (strings).
    """
    try:
        data = sns.load_dataset('penguins')
    except Exception:
        data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

    # Separate features and target; keep NaNs as the pipeline will impute.
    X = data.drop(columns=['species'])
    y = data['species']

    # Collect class names that appear in the data (excluding NaNs).
    class_names = sorted(y.dropna().unique().tolist())
    return X, y, class_names


def plot_data(y, class_names):
    """Plots class distribution as a pie chart.

    Args:
        y (Series): label vector possibly containing NaNs.
        class_names (list): ordered class names to display.
    """
    class_counts = y.value_counts().reindex(class_names, fill_value=0)
    plt.figure(figsize=(6, 6))
    plt.pie(
        class_counts,
        labels=class_names,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Class Distribution")


def evaluate_model(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Prints standard classification metrics and plots a confusion matrix.

    Args:
        y_true (array-like): ground-truth labels.
        y_pred (array-like): predicted labels.
        class_names (list): label order to align rows/cols in the confusion matrix.
        title (str): title for the heatmap figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

    # Heatmap for quick visual inspection of errors by class.
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

if __name__ == "__main__":
  # Load data
  X, y, class_names = load_data()

  # Plot class distribution
  plot_data(y, class_names)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

  print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

  # Identify numerical and categorical columns
  numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
  categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
  print(f"Numerical columns: {numerical_cols}")
  print(f"Categorical columns: {categorical_cols}\n")

  # Preprocessing pipelines
  numerical_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median'))
  ])

  categorical_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('onehot', OneHotEncoder(handle_unknown='ignore'))
  ])

  preprocessor = ColumnTransformer(transformers=[
      ('num', numerical_pipeline, numerical_cols),
      ('cat', categorical_pipeline, categorical_cols)
  ])

  # Create the full pipeline with a Decision Tree classifier with best found hyperparameters, which are the default ones:
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', DecisionTreeClassifier(ccp_alpha=0.0, 
                                            criterion='gini', 
                                            max_depth=None, 
                                            min_samples_leaf=1, 
                                            min_samples_split=2 , random_state=42))
  ])

  print("Evaluating Decision Tree Classifier with default parameters:")

  # Train the model 
  pipeline.fit(X_train, y_train) 
  
  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = pipeline.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 1")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = pipeline.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 1")

  print("\nEvaluating Decision Tree Classifier with min_impurity_decrease=0.2:")

  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', DecisionTreeClassifier(min_impurity_decrease=0.2, random_state=42))
  ])

  # Train the model 
  pipeline.fit(X_train, y_train) 
  
  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = pipeline.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 2")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = pipeline.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 2")

  print("\nEvaluating Decision Tree with an example of an unregularized tree:")

  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42))
  ])

  # Train the model 
  pipeline.fit(X_train, y_train)

  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = pipeline.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 3")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = pipeline.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 3")

  print("Evaluating Decision Tree with regularization for overfitting mitigation:")

  # Create the full pipeline with a Bagging Classifier wrapping a Decision Tree classifier
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', BaggingClassifier(
          estimator=DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=4, random_state=42),
          n_estimators=50,
          random_state=42,
          n_jobs=-1
      ))
  ])

  # Train the model 
  pipeline.fit(X_train, y_train)

  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = pipeline.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 4")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = pipeline.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 4")
  
  plt.show()


# Used for obtaining best hyperparameters through GridSearchCV
'''
  # Hyperparameter tuning using GridSearchCV
  param_grid = {
      'classifier': [DecisionTreeClassifier(random_state=42)],
      'classifier__criterion': ['gini', 'entropy', 'log_loss'],
      'classifier__max_depth': [None, 3, 4, 5, 6, 8, 10],
      'classifier__min_samples_split': [2, 5, 10, 20],
      'classifier__min_samples_leaf': [1, 2, 4, 8, 10],
      'classifier__ccp_alpha': [0.0, 0.001, 0.01],
    }

  grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
    refit='f1_macro',
    cv=5,
    n_jobs=-1,
    verbose=1,
  )
  grid.fit(X_train, y_train)
  
  print(f"Best parameters: {grid.best_params_}")
  print(f"Best cross-validation F1 Macro: {grid.best_score_:.4f}")
  print(f"Best estimator: {grid.best_estimator_.named_steps['classifier']}")
  best_model = grid.best_estimator_

  # Evaluate on validation set
  y_val_pred = best_model.predict(X_val)
  print("\nValidation Set Evaluation:")
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation)")
  
  # Evaluate on test set
  y_test_pred = best_model.predict(X_test)
  print("\nTest Set Evaluation:")
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test)")
  '''
