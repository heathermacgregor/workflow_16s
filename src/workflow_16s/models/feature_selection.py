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
        raise ImportError("SHAP library is not installed")

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
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0 or step_size <= 0:
        raise ValueError("num_features and step_size must be positive integers.")

    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")

    if threads <= 0:
        raise ValueError("threads must be a positive integer.")

    if iterations <= 0 or depth <= 0:
        raise ValueError("iterations and depth must be positive integers.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive float.")

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

    # Transform datasets using the fitted RFE object
    X_train_selected = pd.DataFrame(
        rfe.transform(X_train), 
        columns=X_train.columns[rfe.support_],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        rfe.transform(X_test),
        columns=X_train.columns[rfe.support_],
        index=X_test.index
    )

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
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")

    # Perform SelectKBest
    skb = SelectKBest(score_func=score_func, k=num_features)
    skb.fit(X_train, y_train)

    # Transform datasets
    X_train_selected = pd.DataFrame(
        skb.transform(X_train),
        columns=X_train.columns[skb.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        skb.transform(X_test),
        columns=X_train.columns[skb.get_support()],
        index=X_test.index
    )

    # Get selected feature names
    selected_features = X_train.columns[skb.get_support()].tolist()

    if verbose > 0:
        logger.info(f"Selected {num_features} features using {score_func.__name__}:\n{selected_features}")

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
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")

    if (X_train < 0).any().any() or (X_test < 0).any().any():
        raise ValueError("Chi-Squared test requires non-negative feature values.")

    # Perform Chi-Squared feature selection
    chi2_selector = SelectKBest(score_func=chi2, k=num_features)
    chi2_selector.fit(X_train, y_train)

    # Transform datasets
    X_train_selected = pd.DataFrame(
        chi2_selector.transform(X_train),
        columns=X_train.columns[chi2_selector.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        chi2_selector.transform(X_test),
        columns=X_train.columns[chi2_selector.get_support()],
        index=X_test.index
    )

    # Get selected feature names
    selected_features = X_train.columns[chi2_selector.get_support()].tolist()

    if verbose > 0:
        logger.info(f"Selected {num_features} features using Chi-Squared Test:\n{selected_features}")

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
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    if num_features > X_train.shape[1]:
        raise ValueError("num_features cannot be greater than the total number of features.")

    if penalty not in ['l1', 'l2']:
        raise ValueError("penalty must be either 'l1' or 'l2'.")

    if solver not in ['liblinear', 'saga'] and penalty == 'l1':
        raise ValueError("For L1 penalty, solver must be 'liblinear' or 'saga'.")

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
    X_train_selected = pd.DataFrame(
        model.transform(X_train),
        columns=X_train.columns[model.get_support()],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        model.transform(X_test),
        columns=X_train.columns[model.get_support()],
        index=X_test.index
    )

    # Get selected feature names
    selected_features = X_train.columns[model.get_support()].tolist()

    if verbose > 0:
        logger.info(f"Selected {num_features} features using Lasso ({penalty} penalty):\n{selected_features}")

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
    shap_obj = explainer(X_train)  # Use new SHAP API
    shap_values = shap_obj.values

    # Calculate mean absolute SHAP values
    shap_sum = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_sum)[-num_features:]
    selected_features = X_train.columns[top_indices].tolist()

    # Select features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    if verbose > 0:
        logger.info(f"Selected {num_features} features using SHAP values: {selected_features}")

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
    perm_importance_scorer: Callable = matthews_corrcoef,  # Changed to raw metric function
    perm_importance_n_repeats: int = 10,
    catboost_params: Union[Dict, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using one of the supported methods and optionally apply 
    permutation importance.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)

    if feature_selection not in ['rfe', 'select_k_best', 'chi_squared', 'lasso', 'shap']:
        raise ValueError(
            "feature_selection must be one of 'rfe', 'select_k_best', 'chi_squared', 'lasso', 'shap'."
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

        # Compute permutation importance using raw metric function
        perm_importance = permutation_importance(
            perm_model,
            X_test_selected.values,  # Use numpy array
            y_test,
            scoring=perm_importance_scorer,
            n_repeats=perm_importance_n_repeats,
            random_state=random_state
        )

        importances = pd.Series(
            perm_importance.importances_mean, index=selected_features
        )
        selected_features = importances[importances > 0].index.tolist()
        X_train_selected = X_train_selected[selected_features]
        X_test_selected = X_test_selected[selected_features]

    if verbose > 0:
        logger.info(f"Selected {len(selected_features)} features:\n{selected_features}")

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
    refit: str = 'mcc',
    verbose: int = 1,
    fixed_params: Dict = None
) -> Tuple[CatBoostClassifier, Dict, float, Dict]:
    """
    Enhanced grid search with cross-validation and comprehensive model evaluation.
    """
    # Input validation
    _validate_inputs(X_train, y_train, X_test, y_test)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
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
        # Combine grid parameters with fixed parameters
        current_params = dict(zip(param_grid.keys(), params))
        if fixed_params:
            current_params.update(fixed_params)
        
        # Initialize fold scores with all metrics
        fold_scores = {
            'accuracy': [],
            'f1': [],
            'mcc': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
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
            y_proba = model.predict_proba(X_fold_val)[:, 1]  # Probability for positive class
            
            # Compute metrics directly
            scores = {
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'f1': f1_score(y_fold_val, y_pred),
                'mcc': matthews_corrcoef(y_fold_val, y_pred),
                'roc_auc': roc_auc_score(y_fold_val, y_proba),
                'pr_auc': average_precision_score(y_fold_val, y_proba)
            }
            
            for metric_name, score_val in scores.items():
                fold_scores[metric_name].append(score_val)
            
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
    test_scores = {}
    if cv_model:
        y_pred = cv_model.predict(X_test)
        y_proba = cv_model.predict_proba(X_test)[:, 1]
        
        test_scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }
        
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
            if result == best_params:  # Match by parameter set
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
    
    return best_model, best_params, best_score, test_scores

def save_feature_importances(
    model: CatBoostClassifier, 
    X_train: pd.DataFrame, 
    output_dir: Union[str, Path]
):
    """
    Save feature importances from a trained model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(X_train.shape[1])
    
    feat_imp = pd.Series(importances, index=X_train.columns, name='importance')
    feat_imp = feat_imp.sort_values(ascending=False)
    feat_imp.to_csv(Path(output_dir) / "feature_importances.csv")

def catboost_feature_selection(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Union[str, Path],
    contamination_status_col: str,
    method: str = 'rfe',
    n_top_features: int = 100,
    filter_col: Union[str, None] = None,
    filter_val: Union[str, None] = None
) -> Dict:
    """
    Perform feature selection using CatBoost and save results.
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

    num_features = min(500, X_train.shape[1])
        
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

    # Comprehensive parameter grid with fixed parameters
    param_grid = {
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }
    
    fixed_params = {
        'loss_function': 'Logloss',
        'thread_count': 4,
        'random_state': DEFAULT_RANDOM_STATE
    }

    # Run grid search with selected features
    best_model, best_params, best_score, test_scores = grid_search(
        X_train_selected, 
        y_train, 
        X_test_selected, 
        y_test, 
        param_grid, 
        output_dir,
        fixed_params=fixed_params
    )
    
    # Save feature importances
    save_feature_importances(
        best_model, 
        pd.DataFrame(X_train_selected, columns=final_selected_features), 
        output_dir
    )

    # Get feature importances
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.zeros(len(final_selected_features))
    
    # Create feature importance Series
    feat_imp = pd.Series(importances, index=final_selected_features, name='importance')
    feat_imp = feat_imp.sort_values(ascending=False)
    
    # Get top N features
    top_features = feat_imp.head(n_top_features).index.tolist()
    
    # SHAP summary plots
    try:
        _check_shap_installed()
        explainer = shap.TreeExplainer(best_model)
        shap_obj = explainer(X_train_selected)  # Use new SHAP API
        
        # Save SHAP summary plot (bar plot)
        plt.figure()
        shap.summary_plot(
            shap_obj.values, X_train_selected, plot_type="bar", 
            class_names=best_model.classes_, show=False
        )
        bar_path = output_dir / "shap_summary_bar.png"
        plt.savefig(bar_path, bbox_inches='tight')
        plt.close()
        
        # Save SHAP summary plot (beeswarm plot)
        plt.figure()
        shap.summary_plot(
            shap_obj.values, X_train_selected.values, 
            feature_names=X_train_selected.columns, show=False
        )
        beeswarm_path = output_dir / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"SHAP plot generation failed: {e}")
        bar_path = beeswarm_path = None

    # Return comprehensive results
    return {
        'model': best_model,
        'feature_importances': feat_imp.to_dict(),
        'top_features': top_features,
        'best_params': best_params,
        'test_scores': test_scores,
        'shap_summary_bar_path': str(bar_path) if bar_path else None,
        'shap_summary_beeswarm_path': str(beeswarm_path) if beeswarm_path else None
    }
