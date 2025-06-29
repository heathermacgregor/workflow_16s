# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import itertools
import logging
import os
import re
import warnings
from argparse import Namespace as Args
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

# Third‑Party Imports
import numpy as np
import pandas as pd
import shap
from biom import load_table
from catboost import (
    CatBoostClassifier, 
    cv,
    Pool    
)
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from skbio.stats.composition import clr
from sklearn.feature_selection import (
    chi2, 
    f_classif, 
    RFE, 
    SelectFromModel,
    SelectKBest, 
    VarianceThreshold
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split    
)
from sklearn.metrics import (
    accuracy_score, 
    auc,
    average_precision_score,
    confusion_matrix, 
    f1_score, 
    matthews_corrcoef, 
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve 
)

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.figures.models.models import (
    plot_confusion_matrix, 
    plot_roc_curve,  
    plot_precision_recall_curve
)

# ========================== INITIALISATION & CONFIGURATION ========================== #

warnings.filterwarnings("ignore") # Hide all warnings
logger = logging.getLogger('workflow_16s')

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42

DEFAULT_METHOD = 'rfe'
DEFAULT_USE_PERMUTATION_IMPORTANCE = True
DEFAULT_THREAD_COUNT = 4
DEFAULT_STEP_SIZE = 1000
DEFAULT_NUM_FEATURES = 500

DEFAULT_ITERATIONS_RFE = 500
DEFAULT_LEARNING_RATE_RFE = 0.1
DEFAULT_DEPTH_RFE = 4

DEFAULT_PENALTY_LASSO = 'l1'
DEFAULT_SOLVER_LASSO = 'liblinear'
DEFAULT_MAX_ITER_LASSO = 1000

DEFAULT_ITERATIONS_SHAP = 1000
DEFAULT_LEARNING_RATE_SHAP = 0.1
DEFAULT_DEPTH_SHAP = 4

# ==================================== FUNCTIONS ===================================== #

# Helper Functions
def _validate_inputs(X_train, y_train, X_test, y_test):
    """Helper function to validate input alignment"""
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
          f"X_train [{X_train.shape[0]}], y_train [{y_train.shape[0]}], "
          f"X_test [{X_test.shape[0]}], and y_test [{y_test.shape[0]}] "
          f"must have the same number of samples"
        )


def _save_dataframe(
    df: pd.DataFrame, 
    output_path: Union[str, Path], 
    file_format: str = 'csv'
):
    """Helper function to save a DataFrame to a file."""
    if file_format == 'csv':
        df.to_csv(output_path, index=False)
    elif file_format == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("file_format must be 'csv' or 'excel'")


def _check_shap_installed():
    """Helper function to check if SHAP is installed"""
    try:
        import shap
    except ImportError:
        raise ImportError(
          "SHAP library is not installed"
        )


# Feature Selection Functions
def filter_data(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    contamination_status_col: str = DEFAULT_GROUP_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets while maintaining the proportion of 
    contaminated and non-contaminated samples.

    Args:
        X:                        Features DataFrame.
        y:                        Target Series.
        metadata:                 DataFrame containing metadata, including the contamination 
                                  status column.
        contamination_status_col: Column name in metadata indicating contamination status.
        test_size:                Proportion of the dataset to include in the test split.
        random_state:             Seed for random number generation.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    # Input validation
    if not all(X.index == y.index) or not all(X.index == metadata.index):
        raise ValueError("X, y, and metadata must have the same index.")

    if contamination_status_col not in metadata.columns:
        raise ValueError(f"'{contamination_status_col}' not found in metadata.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Split the data while maintaining the proportion of contaminated samples
    stratify = metadata[contamination_status_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def rfe_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    step_size: int,
    threads: int,
    random_state: int,
    iterations: int = DEFAULT_ITERATIONS_RFE,
    learning_rate: float = DEFAULT_LEARNING_RATE_RFE,
    depth: int = DEFAULT_DEPTH_RFE,
    verbose: int = 1,
    catboost_params: Union[Dict, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform Recursive Feature Elimination (RFE) using a CatBoostClassifier.

    Args:
        X_train:           Training feature DataFrame.
        y_train:           Training target Series.
        X_test:            Testing feature DataFrame.
        y_test:            Testing target Series.
        num_features:      Number of features to select.
        step_size:         Number of features to remove at each step.
        threads:           Number of threads to use for CatBoost.
        random_state:      Random seed for reproducibility.
        iterations:        Number of boosting iterations for CatBoost.
        learning_rate:     Learning rate for CatBoost.
        depth:             Depth of the trees in CatBoost.
        verbose:           Controls verbosity (0 for silent, 1 for progress).
        catboost_params:   Dictionary of additional CatBoost parameters (optional).

    Returns:
        X_train_selected:  Training data with selected features.
        X_test_selected:   Testing data with selected features.
        selected_features: List of selected feature names.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0 or step_size <= 0:
        raise ValueError(
          "num_features and step_size must be positive integers."
        )

    if num_features > X_train.shape[1]:
        raise ValueError(
          "num_features cannot be greater than the total number of features."
        )

    if threads <= 0:
        raise ValueError(
          "threads must be a positive integer."
        )

    if iterations <= 0 or depth <= 0:
        raise ValueError(
          "iterations and depth must be positive integers."
        )

    if learning_rate <= 0:
        raise ValueError(
          "learning_rate must be a positive float."
        )

    # Initialize CatBoostClassifier
    if catboost_params is None:
        catboost_params = {}

    rfe_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        thread_count=threads,
        random_state=random_state,
        verbose=10,
        **catboost_params
    )

    # Perform RFE
    rfe = RFE(
        estimator=rfe_model,
        n_features_to_select=num_features,
        step=step_size,
        verbose=10
    )

    rfe.fit(X_train, y_train)

    # Transform datasets
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[rfe.support_].tolist()

    if verbose > 0:
        logger.info(f"Selected features: {selected_features}")

    return X_train_selected, X_test_selected, selected_features


def select_k_best_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    score_func: Callable = f_classif,
    verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using SelectKBest.

    Args:
        X_train:           Training feature DataFrame.
        y_train:           Training target Series.
        X_test:            Testing feature DataFrame.
        y_test:            Testing target Series.
        num_features:      Number of top features to select.
        score_func:        Scoring function to use for feature selection 
                           (default: f_classif).
        verbose:           Controls verbosity (0 for silent, 1 for progress).

    Returns:
        X_train_selected:  Training data with selected features
        X_test_selected:   Testing data with selected features
        selected_features: List of selected feature names
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError(
          "num_features must be a positive integer."
        )

    if num_features > X_train.shape[1]:
        raise ValueError(
          "num_features cannot be greater than the total number of features."
        )

    # Perform SelectKBest
    skb = SelectKBest(score_func=score_func, k=num_features)
    skb.fit(X_train, y_train)

    # Transform datasets
    X_train_selected = skb.transform(X_train)
    X_test_selected = skb.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[skb.get_support()].tolist()

    if verbose > 0:
        logger.info(
          f"Selected {num_features} features using {score_func.__name__}:"
          f"\n{selected_features}"
        )

    return X_train_selected, X_test_selected, selected_features


def chi_squared_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using the Chi-Squared test.

    Args:
        X_train:           Training feature DataFrame.
        y_train:           Training target Series.
        X_test:            Testing feature DataFrame.
        y_test:            Testing target Series.
        num_features:      Number of top features to select.
        verbose:           Controls verbosity (0 for silent, 1 for progress).

    Returns:
        X_train_selected:  Training data with selected features.
        X_test_selected:   Testing data with selected features.
        selected_features: List of selected feature names.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError(
          "num_features must be a positive integer."
        )

    if num_features > X_train.shape[1]:
        raise ValueError(
          "num_features cannot be greater than the total number of features."
        )

    if (X_train < 0).any().any() or (X_test < 0).any().any():
        raise ValueError(
          "Chi-Squared test requires non-negative feature values."
        )

    # Perform Chi-Squared feature selection
    chi2_selector = SelectKBest(score_func=chi2, k=num_features)
    chi2_selector.fit(X_train, y_train)

    # Transform datasets
    X_train_selected = chi2_selector.transform(X_train)
    X_test_selected = chi2_selector.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[chi2_selector.get_support()].tolist()

    if verbose > 0:
        logger.info(
          f"Selected {num_features} features using Chi-Squared Test:"
          f"\n{selected_features}"
        )

    return X_train_selected, X_test_selected, selected_features


def lasso_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    penalty: str = DEFAULT_PENALTY_LASSO,
    solver: str = DEFAULT_SOLVER_LASSO,
    max_iter: int = 1000,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: int = DEFAULT_MAX_ITER_LASSO,
    lasso_params: Union[Dict, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using Lasso (L1-regularized Logistic Regression).

    Args:
        X_train:           Training feature DataFrame.
        y_train:           Training target Series.
        X_test:            Testing feature DataFrame.
        y_test:            Testing target Series.
        num_features:      Number of top features to select.
        penalty:           Type of regularization ('l1' or 'l2').
        solver:            Optimization algorithm to use (e.g., 'liblinear', 
                           'saga').
        max_iter:          Maximum number of iterations for the solver.
        random_state:      Random seed for reproducibility.
        verbose:           Controls verbosity (0 for silent, 1 for progress).
        lasso_params:      Dictionary of additional LogisticRegression 
                           parameters (optional).

    Returns:
        X_train_selected:  Training data with selected features.
        X_test_selected:   Testing data with selected features.
        selected_features: List of selected feature names.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError(
          "num_features must be a positive integer."
        )

    if num_features > X_train.shape[1]:
        raise ValueError(
          "num_features cannot be greater than the total number of features."
        )

    if penalty not in ['l1', 'l2']:
        raise ValueError(
          "penalty must be either 'l1' or 'l2'."
        )

    if solver not in ['liblinear', 'saga'] and penalty == 'l1':
        raise ValueError(
          "For L1 penalty, solver must be 'liblinear' or 'saga'."
        )

    # Initialize LogisticRegression
    if lasso_params is None:
        lasso_params = {}

    lasso = LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **lasso_params
    )

    # Fit the model
    lasso.fit(X_train, y_train)

    # Perform feature selection
    model = SelectFromModel(
        lasso,
        max_features=num_features,
        prefit=True
    )

    # Transform datasets
    X_train_selected = model.transform(X_train)
    X_test_selected = model.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[model.get_support()].tolist()

    if verbose > 0:
        logger.info(
          f"Selected {num_features} features using Lasso ({penalty} penalty):"
          f"\n{selected_features}"
        )

    return X_train_selected, X_test_selected, selected_features


def shap_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: int,
    threads: int,
    iterations: int = DEFAULT_ITERATIONS_SHAP,
    learning_rate: float = DEFAULT_LEARNING_RATE_SHAP,
    depth: int = DEFAULT_DEPTH_SHAP,
    verbose: int = 1,
    catboost_params: Union[Dict, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using SHAP (SHapley Additive exPlanations) values.

    Args:
        X_train:           Training feature DataFrame.
        y_train:           Training target Series.
        X_test:            Testing feature DataFrame.
        y_test:            Testing target Series.
        num_features:      Number of top features to select.
        threads:           Number of threads to use for CatBoost.
        iterations:        Number of boosting iterations for CatBoost.
        learning_rate:     Learning rate for CatBoost.
        depth:             Depth of the trees in CatBoost.
        verbose:           Controls verbosity (0 for silent, 1 for progress).
        catboost_params:   Dictionary of additional CatBoost parameters (optional).

    Returns:
        X_train_selected:  Training data with selected features.
        X_test_selected:   Testing data with selected features.
        selected_features: List of selected feature names.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")

    if threads <= 0:
        raise ValueError("threads must be a positive integer.")

    if iterations <= 0 or depth <= 0:
        raise ValueError("iterations and depth must be positive integers.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive float.")

    # Check if SHAP is installed
    _check_shap_installed()

    # Initialize CatBoostClassifier
    if catboost_params is None:
        catboost_params = {}

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        thread_count=threads,
        verbose=0,
        **catboost_params
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Explain model with SHAP values
    if verbose > 0:
        logger.info("Explaining model with SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate mean absolute SHAP values
    shap_sum = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_sum)[-num_features:]
    selected_features = X_train.columns[top_indices].tolist()

    # Select features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    if verbose > 0:
        logger.info(
          f"Selected {num_features} features using SHAP values: "
          f"{selected_features}"
        )

    return X_train_selected, X_test_selected, selected_features


def perform_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_selection: str = DEFAULT_METHOD,
    use_permutation_importance: bool = DEFAULT_USE_PERMUTATION_IMPORTANCE,
    thread_count: int = DEFAULT_THREAD_COUNT,
    step_size: int = DEFAULT_STEP_SIZE,
    num_features: int = DEFAULT_NUM_FEATURES,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: int = 1,
    feature_selection_params: Union[Dict, None] = None,
    perm_importance_scorer: Callable = make_scorer(matthews_corrcoef),
    perm_importance_n_repeats: int = 10,
    catboost_params: Union[Dict, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using one of the supported methods and optionally apply 
    permutation importance.

    Argss:
        X_train:                    Training feature DataFrame.
        y_train:                    Training target Series.
        X_test:                     Testing feature DataFrame.
        y_test:                     Testing target Series.
        feature_selection:          Feature selection method to use ('rfe', 'select_k_best', 
                                    'chi_squared', 'lasso', 'shap').
        use_permutation_importance: Whether to apply permutation importance after feature 
                                    selection.
        thread_count:               Number of threads to use for parallel computation.
        step_size:                  Step size for RFE feature selection.
        num_features:               Number of top features to select.
        random_state:               Random seed for reproducibility.
        verbose:                    Controls verbosity (0 for silent, 1 for progress).
        feature_selection_params:   Dictionary of additional parameters for the feature 
                                    selection method (optional).
        perm_importance_scorer:     Scoring function for permutation importance (default: 
                                    Matthews correlation coefficient).
        perm_importance_n_repeats:  Number of repeats for permutation importance.
        catboost_params:            Dictionary of additional CatBoost parameters for permutation 
                                    importance (optional).

    Returns:
        X_train_selected:           Training data with selected features.
        X_test_selected:            Testing data with selected features.
        selected_features:          List of selected feature names.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if feature_selection not in ['rfe', 'select_k_best', 'chi_squared', 'lasso', 'shap']:
        raise ValueError(
            "feature_selection must be one of 'rfe', 'select_k_best', 'chi_squared',"
            "'lasso', 'shap'."
        )

    if num_features <= 0 or step_size <= 0 or thread_count <= 0:
        raise ValueError(
            "num_features, step_size, and thread_count must be positive integers."
        )

    if feature_selection_params is None:
        feature_selection_params = {}

    if catboost_params is None:
        catboost_params = {}

    # Perform feature selection
    if feature_selection == 'rfe':
        X_train_selected, X_test_selected, selected_features = rfe_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            step_size=step_size,
            threads=thread_count,
            random_state=random_state,
            **feature_selection_params
        )
    elif feature_selection == 'select_k_best':
        X_train_selected, X_test_selected, selected_features = select_k_best_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **feature_selection_params
        )
    elif feature_selection == 'chi_squared':
        X_train_selected, X_test_selected, selected_features = chi_squared_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **feature_selection_params
        )
    elif feature_selection == 'lasso':
        X_train_selected, X_test_selected, selected_features = lasso_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            **feature_selection_params
        )
    elif feature_selection == 'shap':
        X_train_selected, X_test_selected, selected_features = shap_feature_selection(
            X_train, y_train, X_test, y_test,
            num_features=num_features,
            threads=thread_count,
            **feature_selection_params
        )

    # Apply permutation importance if required
    if use_permutation_importance:
        if verbose > 0:
            logger.info("Computing permutation importances...")

        perm_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=4,
            verbose=10,
            thread_count=thread_count,
            random_state=random_state,
            **catboost_params
        )
        perm_model.fit(
            X_train_selected,
            y_train,
            eval_set=(X_test_selected, y_test),
            verbose=10,
            early_stopping_rounds=100
        )

        perm_importance = permutation_importance(
            perm_model,
            X_test_selected,
            y_test,
            scoring=perm_importance_scorer,
            n_repeats=perm_importance_n_repeats,
            random_state=random_state
        )

        importances = pd.Series(
            perm_importance.importances_mean, index=selected_features
        )
        selected_features = importances[importances > 0].index.tolist()
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

    if verbose > 0:
        logger.info(
          f"Selected {len(selected_features)} features:"
          f"\n{selected_features}"
        )

    return X_train_selected, X_test_selected, selected_features


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    output_dir: Union[str, Path],
    plot: bool = False,
    verbose: int = 1,
    early_stopping_rounds: int = 100,
    custom_metrics: Union[Dict[str, Callable], None] = None
) -> Tuple[CatBoostClassifier, float, float, float, pd.Series]:
    """
    Train and evaluate a CatBoostClassifier model.

    Args:
        X_train:               Training feature DataFrame.
        y_train:               Training target Series.
        X_test:                Testing feature DataFrame.
        y_test:                Testing target Series.
        params:                Dictionary of CatBoostClassifier parameters.
        output_dir:            Directory to save the trained model and logs.
        plot:                  Whether to plot training progress (default: False).
        verbose:               Controls verbosity (0 for silent, 10 for 
                               detailed output).
        early_stopping_rounds: Number of rounds for early stopping.
        custom_metrics:        Dictionary of custom evaluation metrics (optional).

    Returns:
        model:                 Trained CatBoostClassifier model.
        accuracy:              Accuracy score on the test set.
        f1:                    F1 score on the test set.
        mcc:                   Matthews correlation coefficient on the test set.
        y_pred:                Predicted labels for the test set.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CatBoostClassifier
    train_dir = str(output_dir / 'catboost_info')
    model = CatBoostClassifier(
        **params,
        train_dir=train_dir
    )

    if verbose > 0:
        logger.info(f"Training with parameters: {params}")

    # Train the model
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        plot=plot,
        verbose=10,
        early_stopping_rounds=early_stopping_rounds
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            metric_value = metric_func(y_test, y_pred)
            logger.info(f"{metric_name}: {metric_value}")

    if verbose > 0:
        logger.info(f"Training completed. Params: {params}")
        logger.info(f"Accuracy: {accuracy}, F1 Score: {f1}, MCC: {mcc}")

    # Save the model
    model.save_model(str(output_dir / 'model.cbm'))

    return model, accuracy, f1, mcc, y_pred


def grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict[str, List],
    output_dir: Union[str, Path],
    n_splits: int = 5,
    scoring: Dict[str, Callable] = None,
    refit: str = 'mcc',
    verbose: int = 1
) -> Tuple[CatBoostClassifier, Dict, float]:
    """
    Enhanced grid search with cross-validation and comprehensive model evaluation.
    
    Args:
        X_train:     Training feature DataFrame
        y_train:     Training target Series
        X_test:      Testing feature DataFrame
        y_test:      Testing target Series
        param_grid:  Dictionary of hyperparameter grids
        output_dir:  Directory to save results
        n_splits:    Number of cross-validation folds (default: 5)
        scoring:     Dictionary of scoring metrics (name: function)
        refit:       Metric to use for selecting best model
        verbose:     Verbosity level (0-2)
    
    Returns:
        best_model:  Best trained CatBoostClassifier
        best_params: Best hyperparameters
        best_score:  Best score of the reference metric
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Default scoring metrics
    if scoring is None:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'mcc': make_scorer(matthews_corrcoef),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'pr_auc': make_scorer(average_precision_score, needs_proba=True)
        }
    
    # Initialize results storage
    results = []
    best_score = -np.inf
    best_model = None
    best_params = None
    cv_model = None
    
    # Create parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    total_combinations = len(param_combinations)
    
    if verbose:
        logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        logger.info(f"Using {n_splits}-fold cross-validation")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    
    for i, params in enumerate(param_combinations, 1):
        current_params = dict(zip(param_grid.keys(), params))
        fold_scores = {metric: [] for metric in scoring}
        
        if verbose:
            logger.info(f"\n[{i}/{total_combinations}] Testing params: {current_params}")
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Initialize and train model
            model = CatBoostClassifier(
                **current_params,
                train_dir=str(output_dir / 'catboost_info'),
                verbose=False
            )
            
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=100,
                verbose=verbose > 1
            )
            
            # Evaluate on validation fold
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)[:, 1]
            
            for metric_name, scorer in scoring.items():
                if "auc" in metric_name or "pr_auc" in metric_name:
                    score = scorer(y_fold_val, y_proba)
                else:
                    score = scorer(y_fold_val, y_pred)
                fold_scores[metric_name].append(score)
            
            # Clean up memory
            del model
        
        # Calculate mean CV scores
        cv_means = {f"mean_{k}": np.mean(v) for k, v in fold_scores.items()}
        cv_stds = {f"std_{k}": np.std(v) for k, v in fold_scores.items()}
        current_result = {
            **current_params,
            **cv_means,
            **cv_stds
        }
        results.append(current_result)
        
        # Check if best model
        current_ref_score = cv_means[f"mean_{refit}"]
        if current_ref_score > best_score:
            best_score = current_ref_score
            best_params = current_params
            if verbose:
                logger.info(f"🔥 New best {refit} (CV): {current_ref_score:.4f}")
            
            # Train on full training set with best params
            cv_model = CatBoostClassifier(
                **best_params,
                train_dir=str(output_dir / 'catboost_info'),
                verbose=False
            )
            cv_model.fit(
                X_train,
                y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=100,
                verbose=verbose > 1
            )
    
    # Final evaluation on test set
    if cv_model:
        y_pred = cv_model.predict(X_test)
        y_proba = cv_model.predict_proba(X_test)[:, 1]
        
        test_scores = {}
        for metric_name, scorer in scoring.items():
            if "auc" in metric_name or "pr_auc" in metric_name:
                test_scores[metric_name] = scorer(y_test, y_proba)
            else:
                test_scores[metric_name] = scorer(y_test, y_pred)
        
        # Save final model
        best_model = cv_model
        best_model.save_model(str(output_dir / "best_model.cbm"))
        
        # Generate evaluation plots
        plot_confusion_matrix(
            confusion_matrix(y_test, y_pred),
            str(output_dir / "best_confusion_matrix.png")
        )
        plot_roc_curve(
            *roc_curve(y_test, y_proba),
            roc_auc_score(y_test, y_proba),
            str(output_dir / "best_roc_curve.png")
        )
        plot_precision_recall_curve(
            *precision_recall_curve(y_test, y_proba)[:2],
            average_precision_score(y_test, y_proba),
            str(output_dir / "best_precision_recall_curve.png")
        )
        
        # Add test scores to results
        for result in results:
            if result.get('params') == best_params:
                result.update({f"test_{k}": v for k, v in test_scores.items()})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    if verbose:
        logger.info("\nGrid search completed")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV {refit}: {best_score:.4f}")
        if test_scores:
            logger.info("Test set performance:")
            for metric, score in test_scores.items():
                logger.info(f"{metric}: {score:.4f}")
    
    return best_model, best_params, best_score


def save_feature_importances(
    model,
    X_train_selected: pd.DataFrame,
    output_dir: Union[str, Path],
    verbose: int = 1,
    file_format: str = 'csv'
) -> None:
    """
    Save feature importances from a trained model to a file.

    Args:
        model:            Trained model with a `get_feature_importance` method.
        X_train_selected: DataFrame containing the selected features used for training.
        output_dir:       Directory to save the feature importances file.
        verbose:          Controls verbosity (0 for silent, 1 for progress).
        file_format:      File format for saving feature importances ('csv' or 'excel').
    """
    # Input validation
    if not hasattr(model, 'get_feature_importance'):
        raise ValueError("The model must have a `get_feature_importance` method.")

    if not isinstance(X_train_selected, pd.DataFrame):
        raise ValueError("X_train_selected must be a pandas DataFrame.")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get feature importances
    feature_importances = model.get_feature_importance()
    

    if len(feature_importances) != X_train_selected.shape[1]:
        raise ValueError(
            "The number of feature importances does not match the number "
            "of selected features."
        )

    # Create DataFrame
    selected_features = X_train_selected.columns
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Save to file
    output_path = output_dir / f"feature_importances.{file_format}"
    _save_dataframe(importance_df, output_path, file_format)

    if verbose > 0:
        logger.info(f"Feature importances saved to {output_path}")


def catboost_feature_selection(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Union[str, Path],
    contamination_status_col: str,
    method: str = 'rfe',
    n_top_features: int = 100,  # New parameter
    filter_col: Union[str, None] = None,
    filter_val: Union[str, None] = None
) -> Any:
    """
    Perform feature selection using CatBoost and save results.

    Args:
        metadata:                 DataFrame containing metadata.
        features:                 DataFrame containing features.
        output_dir:               Directory to save results.
        contamination_status_col: Column name in metadata indicating contamination 
                                  status.
        method:                   Feature selection method to use ('rfe', 
                                  'select_k_best', 'chi_squared', 'lasso', 'shap').
        filter_col:               Column to filter metadata by (optional).
        filter_val:               Value to filter metadata by (optional).
    """
    output_dir = Path(output_dir) / method
    os.makedirs(output_dir, exist_ok=True)

    if filter_col and filter_val:
        metadata = metadata[metadata[filter_col].str.contains(
            filter_val, case=False, na=False
        )]
        output_dir = output_dir / f"{filter_col}-{filter_val}"
        os.makedirs(output_dir, exist_ok=True)

    # Filter data
    X, y = features, metadata[contamination_status_col]
    X_train, X_test, y_train, y_test = filter_data(
        X, y, metadata, contamination_status_col
    )

    num_features = 500
    if X_train.shape[1] < 500:
        print(X_train.shape)
        num_features = X_train.shape[1]
        
    # Perform feature selection
    X_train_selected, X_test_selected, final_selected_features = perform_feature_selection(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_selection=method,
        num_features=num_features
    )
    logger.info(X_train_selected.var().describe())  # Check feature variance after correction


    # Define CatBoost parameters
    params = {
        'iterations': [1000],
        'learning_rate': [0.1],
        'depth': [4],
        'loss_function': ['Logloss'],
        'thread_count': [4]
    }

    # Save selected features
    X_train_selected.to_csv(output_dir / "X_train_selected.csv", index=False)

    # Run grid search with selected features
    best_model, best_params, best_mcc = grid_search(
        X_train_selected, y_train, X_test_selected, y_test, params, output_dir
    )
    logger.info(f"Best Model Parameters: {best_params}, MCC: {best_mcc}")

    # Save feature importances
    save_feature_importances(
        best_model, 
        pd.DataFrame(X_train_selected, columns=final_selected_features), 
        output_dir
    )

    # Save model
    best_model_path = output_dir / "best_model.cbm"
    best_model.save_model(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # SHAP summary plots
    _check_shap_installed()
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(
        shap_values, X_train, plot_type="bar", 
        class_names=best_model.classes_
    )
    shap.summary_plot(
        shap_values, X_train.values, feature_names=X_train.columns
    )

    # Save SHAP summary plot (bar plot)
    plt.figure()
    shap.summary_plot(
        shap_values, X_train, plot_type="bar", 
        class_names=best_model.classes_, show=False
    )
    plt.savefig(output_dir / "shap_summary_bar.png", bbox_inches='tight')
    logger.info(str(output_dir / "shap_summary_bar.png"))
    plt.close()
    
    # Save SHAP summary plot (beeswarm plot)
    plt.figure()
    shap.summary_plot(
        shap_values, X_train.values, 
        feature_names=X_train.columns, show=False
    )
    plt.savefig(output_dir / "shap_summary_beeswarm.png", bbox_inches='tight')
    logger.info(str(output_dir / "shap_summary_beeswarm.png"))
    plt.close()

    # Get the mean absolute SHAP values for each feature
    shap_values_mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance and select the top N
    top_n = 10
    top_features_indices = np.argsort(shap_values_mean_abs)[-top_n:]
    shap_values_filtered = shap_values[:, top_features_indices]
    top_features_indices = [i for i in top_features_indices 
                            if i < shap_values.shape[1]]

    for feature in top_features_indices:
        logger.info("shap_values shape:", shap_values.shape)
        logger.info("X shape:", X_train.shape)
        logger.info("Feature index:", feature)

        shap.dependence_plot(
            feature, shap_values, X_train, feature_names=X_train.columns
        )
        plt.savefig(output_dir / f"shap_dependence_{feature}.png", bbox_inches='tight')
        logger.info(str(output_dir / f"shap_dependence_{feature}.png"))
        plt.close()

    model = best_model
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For RFE, use ranking (1 = selected)
        importances = [1 if s else 0 for s in selector.support_]
    
    # Create feature importance Series
    feat_imp = pd.Series(importances, index=features.columns, name='importance')
    feat_imp = feat_imp.sort_values(ascending=False)
    
    # Get top N features
    top_features = feat_imp.head(n_top_features).index.tolist()
    
    return {
        'model': model,
        'feature_importances': feat_imp,
        'top_features': top_features
    }
