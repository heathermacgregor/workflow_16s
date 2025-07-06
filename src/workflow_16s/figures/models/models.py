# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third-Party Imports
import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import shap
from matplotlib import (
    colors as mcolors,
    pyplot as plt
)
from matplotlib.colors import LogNorm
from plotly.subplots import make_subplots

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.figures.figures import (
    attach_legend_to_figure, largecolorset, plot_legend, plotly_show_and_save,
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #


# ==================================== FUNCTIONS ===================================== #

def plot_confusion_matrix(
    cm_flipped, 
    output_path: Union[str, Path]
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_flipped, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Predicted 1', 'Predicted 0'], 
        yticklabels=['Actual 1', 'Actual 0']
    )
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {output_path}")

        
def plot_roc_curve(
    fpr, 
    tpr, 
    roc_auc, 
    output_path: Union[str, Path]
) -> None:
    plt.figure()
    plt.plot(
        fpr, 
        tpr, 
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curve plot saved to: {output_path}")


def plot_precision_recall_curve(
    precision, 
    recall, 
    average_precision, 
    output_path: Union[str, Path]
) -> None:
    plt.figure()
    plt.step(
        recall, 
        precision, 
        where='post'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Precision recall curve plot saved to: {output_path}")
    

def plot_feature_importance(
  shap_values, 
  X_test, 
  X_train, 
  cbmpf
) -> None:
    shap.summary_plot(
        shap_values, X_train, plot_type="bar", class_names=cbmpf.classes_
    )
    shap.summary_plot(
        shap_values, X_train.values, feature_names = X_train.columns
    )
    
    sns.set_context("talk", font_scale=0.5)
    fea_imp = pd.DataFrame({
      'imp': cbmpf.feature_importances_, 
      'col': X_test.columns
    }).sort_values(['imp', 'col'], ascending=[True, False])
    fea_imp=fea_imp[fea_imp['imp']>0.5]
    fea_imp['fn']=['fn:'+str(i) for i in fea_imp.index]
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
  

def shap_summary_bar_plotly(
    shap_values: np.array, 
    feature_names: List, 
    max_display: int = 20
) -> go.Figure:
    """
    Convert SHAP summary bar plot to a Plotly figure.
    
    Args:
        shap_values:   SHAP values array (n_samples, n_features).
        feature_names: List of feature names.
        max_display:   Maximum number of features to display.
    
    Returns:
        Horizontal bar plot of mean absolute SHAP values.
    """
    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Select top features
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    
    # Create horizontal bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features,
        x=top_values,
        orientation='h',
        marker_color='#1e88e5'
    ))
    
    # Update layout
    fig.update_layout(
        title='SHAP Summary Bar Plot',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Features',
        showlegend=False,
        height=600,
        margin=dict(l=150)
    )
    return fig
    

def shap_beeswarm_plotly(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: List, 
    max_display: int = 20
) -> go.Figure:
    """
    Convert SHAP beeswarm plot to a Plotly figure.
    
    Args:
        shap_values:    SHAP values array (n_samples, n_features).
        feature_values: Feature values array (n_samples, n_features).
        feature_names:  List of feature names.
        max_display:    Maximum number of features to display.
    
    Returns:
        Beeswarm plot of SHAP values.
    """
    # Compute mean absolute SHAP for feature ordering
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    # Prepare figure
    fig = go.Figure()
    y_offset = 0.3  # Vertical spread for jitter
    
    # Add scatter traces for each feature
    np.random.seed(42)  # Consistent jitter
    for idx, feature_idx in enumerate(top_indices):
        shap_vals = shap_values[:, feature_idx]
        feat_vals = feature_values[:, feature_idx]
        
        # Generate jittered y-coordinates
        jitter = np.random.uniform(
            -y_offset, y_offset, size=len(shap_vals)
        )
        y_pos = idx + jitter
        
        # Normalize feature values for coloring
        vmin, vmax = np.min(feat_vals), np.max(feat_vals)
        normalized_vals = (feat_vals - vmin) / (vmax - vmin + 1e-8)
        
        # Custom red-blue color scale
        colors = [
            f'rgb({int(30 + 225*(1 - nv) if nv < 0.5 else 255}, '
            f'{int(136 + 119*(1 - 2*abs(nv - 0.5)) if nv < 0.5 else 67 + 188*(1 - nv))}, '
            f'{int(229 - 229*nv) if nv < 0.5 else 54 + 201*(1 - nv)})'
            for nv in normalized_vals
        ]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=y_pos,
            mode='markers',
            marker=dict(size=5, color=colors),
            name=feature_names[feature_idx],
            hoverinfo='text',
            text=[
                f"<b>Feature</b>: {feature_names[feature_idx]}<br>"
                f"<b>SHAP</b>: {shap_val:.4f}<br>"
                f"<b>Value</b>: {fv:.4f}"
                for shap_val, fv in zip(shap_vals, feat_vals)
            ],
            showlegend=False
        ))
    
    # Add zero line
    fig.add_shape(
        type='line',
        x0=0, y0=-0.5, x1=0, y1=len(top_features) - 0.5,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    # Update layout
    fig.update_layout(
        title='SHAP Beeswarm Plot',
        xaxis_title='SHAP Value',
        yaxis=dict(
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
            title='Features'
        ),
        height=600,
        hovermode='closest',
        margin=dict(l=150),
        plot_bgcolor='white'
    )
    fig.update_yaxes(range=[-0.5, len(top_features) - 0.5])
    return fig
    

def shap_dependency_plot_plotly(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: List, 
    feature: str, 
    max_points: int = 1000
) -> go.Figure:
    """
    Create a SHAP dependency plot for a single feature.
    
    Args:
        shap_values:    SHAP values array.
        feature_values: Feature values array.
        feature_names:  List of feature names.
        feature:        Feature to plot.
        max_points:     Maximum points to show (downsample if exceeded).
    
    Returns:
        Dependency plot figure.
    """
    # Find feature index
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found in feature_names")
    idx = feature_names.index(feature)
    
    # Get values
    x = feature_values[:, idx]
    y = shap_values[:, idx]
    
    # Downsample if too many points
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='markers',
        marker=dict(
            size=6,
            opacity=0.5,
            color=y,
            colorscale='RdBu',
            colorbar=dict(title='SHAP Value'),
        name=feature,
        hovertemplate="<b>Value</b>: %{x:.4f}<br><b>SHAP</b>: %{y:.4f}<extra></extra>"
    ))
    
    # Add trend line using LOWESS smoothing
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.3, it=2)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode='lines',
            line=dict(color='black', width=3),
            name='Trend'
        ))
    except ImportError:
        # Fallback to rolling average if statsmodels not available
        df = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
        df['rolling'] = df['y'].rolling(50, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['rolling'],
            mode='lines',
            line=dict(color='black', width=3),
            name='Trend'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'SHAP Dependency Plot: {feature}',
        xaxis_title=f'Feature Value: {feature}',
        yaxis_title='SHAP Value',
        showlegend=False,
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_shap_plots(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: list, 
    n_features: int = 20, 
    output_dir: Union[str, Path] = None,
    verbose: bool = False
) -> Tuple[go.Figure, go.Figure, List[go.Figure]]:
    """
    Generate both SHAP bar plot, beeswarm plot, and dependency plots as Plotly figures.
    
    Args:
        shap_values:    SHAP values array.
        feature_values: Feature values array.
        feature_names:  List of feature names.
        n_features:     Maximum features to display.
        output_dir:
        verbose:
    
    Returns:
        Tuple of bar_plot_fig, beeswarm_plot_fig, dependency_plot_figs.
    """
    bar_fig = shap_summary_bar_plotly(
        shap_values, feature_names, n_features
    )
    plotly_show_and_save(
        bar_fig, show, 
        output_dir / f"shap.summary.bar.{n_features}", 
        ['png', 'html'], verbose
    )
    beeswarm_fig = shap_beeswarm_plotly(
        shap_values, feature_values, feature_names, n_features
    )
    plotly_show_and_save(
        beeswarm_fig, show, 
        output_dir / f"shap.summary.beeswarm.{n_features}",
        ['png', 'html'], verbose
    )
    # Create dependency plots for top features
    dependency_figs = []
    if dependency_top_features > 0:
        # Get top n features from bar plot data
        top_features = bar_fig.data[0].y[:n_features]
        
        for feature in top_features:
            try:
                dep_fig = shap_dependency_plot_plotly(
                    shap_values, feature_values, feature_names, feature
                )
                plotly_show_and_save(
                    dep_fig, show, output_dir / f"shap.dependency.{feature}", 
                    ['png', 'html'], verbose
                )
                dependency_figs.append(dep_fig)
            except Exception as e:
                print(
                    f"Error creating dependency plot for {feature}: {str(e)}"
                )
    return bar_fig, beeswarm_fig, dependency_figs
