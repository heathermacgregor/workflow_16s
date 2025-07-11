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
from workflow_16s.figures.merged import _apply_common_layout

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally
warnings.filterwarnings("ignore") # Suppress warnings

# ==================================== FUNCTIONS ===================================== #

def plot_confusion_matrix(
    cm_flipped: np.ndarray,
    output_path: Union[str, Path],
    class_names: List[str] = ['Positive', 'Negative'],
    show: bool = False,
    verbose: bool = True
) -> None:
    """
    Create an interactive confusion matrix plot using Plotly.
    
    Args:
        cm_flipped:  Confusion matrix (2x2) as [[TN, FP], [FN, TP]].
        output_path: Output path for saving the plot (without extension).
        class_names: Names for the classes [Actual, Predicted].
        show:        Whether to display the plot.
        verbose:     Verbosity flag.
    """
    # Create annotation text with values and percentages
    annotations = []
    total = cm_flipped.sum()
    for i in range(cm_flipped.shape[0]):
        for j in range(cm_flipped.shape[1]):
            count = cm_flipped[i, j]
            percentage = f"{count/total:.1%}" if total > 0 else "0%"
            annotations.append(
                f"<b>{count}</b><br>({percentage})"
            )
    
    # Reshape annotations to match matrix shape
    annotations = np.array(annotations).reshape(cm_flipped.shape).tolist()
    
    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_flipped,
        annotation_text=annotations,
        colorscale='Blues',
        x=[f'Predicted {name}' for name in class_names],
        y=[f'Actual {name}' for name in class_names],
        hoverinfo='z',
        showscale=True
    )
    
    # Add title and labels
    fig.update_layout(
        title_text='<b>Confusion Matrix</b>',
        title_x=0.5,
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        width=600,
        height=600,
        margin=dict(t=100, l=100),
        font=dict(size=12)
    )
    
    # Customize hover text
    fig.update_traces(
        hovertemplate=(
            "<b>Actual</b>: %{y}<br>"
            "<b>Predicted</b>: %{x}<br>"
            "<b>Count</b>: %{z}<br>"
            "<b>Percentage</b>: %{text}"
        ),
        texttemplate="%{text}",
        textfont_size=14
    )
    
    # Reverse y-axis to match typical confusion matrix orientation
    fig.update_yaxes(autorange="reversed")
    
    # Add border lines
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", width=2)
    )
    
    fig = _apply_common_layout(fig, 'Predicted Label', 'Actual Label', '<b>Confusion Matrix</b>') 
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    if verbose:
        logger.info(f"Confusion matrix plot saved to: {output_path}")

        
def plot_roc_curve(
    fpr, 
    tpr, 
    roc_auc, 
    output_path: Union[str, Path],
    show: bool = False,
    verbose: bool = False
) -> go.Figure:
    """
    Plot ROC curve using Plotly.
    
    Args:
        fpr:         False Positive Rates.
        tpr:         True Positive Rates.
        roc_auc:     Area Under ROC Curve.
        output_path: Output path for saving the plot.
        show:        Whether to display the plot.
        verbose:     Whether to log output.
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(width=3, color='#1f77b4')
    ))
    
    # Random chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.50)',
        line=dict(dash='dash', color='#444')
    ))
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1.05], scaleanchor='x', scaleratio=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template='plotly_white',
        width=700,
        height=600
    )
    fig = _apply_common_layout(fig, 'False Positive Rate', 'True Positive Rate', 'Receiver Operating Characteristic') 
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    if verbose:
        logger.info(f"ROC curve plot saved to: {output_path}")
    return fig
    

def plot_precision_recall_curve(
    precision, 
    recall, 
    average_precision, 
    output_path: Union[str, Path],
    show: bool = False,
    verbose: bool = False
) -> go.Figure:
    """
    Plot Precision-Recall curve using Plotly.
    
    Args:
        precision:         Precision values.
        recall:            Recall values.
        average_precision: Average precision score.
        output_path:       Output path for saving the plot.
        show:              Whether to display the plot.
        verbose:           Verbosity flag.
    """
    fig = go.Figure()
    
    # Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'PR curve (AP = {average_precision:.2f})',
        line=dict(width=3, color='#ff7f0e'),
        fill='tozeroy'
    ))
    
    # Update layout
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template='plotly_white',
        width=700,
        height=600
    )
    fig = _apply_common_layout(fig, 'Recall', 'Precision', 'Precision-Recall Curve')
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    if verbose:
        logger.info(f"Precision-Recall curve plot saved to: {output_path}")
    return fig


def plot_feature_importance(
    feature_importances: pd.Series,
    output_path: Union[str, Path],
    threshold: float = 0.5,
    show: bool = False,
    verbose: bool = False
) -> go.Figure:
    """
    Plot horizontal bar chart of feature importances using Plotly.
    
    Args:
        feature_importances: Series with feature importances.
        threshold:           Minimum importance to display.
        output_path:         Output path for saving the plot.
        show:                Whether to display the plot.
        verbose:             Verbosity flag.
    """
    # Filter and sort features
    filtered = feature_importances[feature_importances > threshold]
    sorted_features = filtered.sort_values(ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_features.index,
        x=sorted_features.values,
        orientation='h',
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='Feature Importances',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=600 + len(sorted_features) * 10,  # Dynamic height
        width=800,
        margin=dict(l=150)  # Extra margin for long feature names
    )
    fig = _apply_common_layout(
        fig, 'Importance Score', 'Features', 'Feature Importances'
    )
    
    fig.update_layout(
        'yaxis': {
            'showticklabels': True
        }
    )
    plotly_show_and_save(fig, show, output_path, ['png', 'html'], verbose)
    if verbose:
        logger.info(f"Feature importance plot saved to: {output_path}")
    return fig
    

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
    fig = _apply_common_layout(fig, 'Mean |SHAP Value|', 'Features', 'SHAP Summary Bar Plot')
    fig.update_layout(
        'yaxis': {
            'showticklabels': True
        }
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
            f'rgb({int(30 + 225*(1 - nv)) if nv < 0.5 else 255}, '
            f'{int(136 + 119*(1 - 2*abs(nv - 0.5))) if nv < 0.5 else 67 + 188*(1 - nv)}, '
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
    fig = _apply_common_layout(fig, 'SHAP Value', 'Features', 'SHAP Beeswarm Plot')
    fig.update_layout(
        'yaxis': {
            'showticklabels': True
        }
    )
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
        ),
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
    fig = _apply_common_layout(fig, f'Feature Value: {feature}', 'SHAP Value', f'SHAP Dependency Plot: {feature}')
    return fig


def plot_shap(
    shap_values: np.array, 
    feature_values: np.array, 
    feature_names: list, 
    n_features: int = 20, 
    output_dir: Union[str, Path] = None,
    show: bool = False,
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
        show:
        verbose:        Verbosity flag.
    
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
    if n_features > 0:
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
