# ===================================== IMPORTS ====================================== #
import warnings
warnings.filterwarnings("ignore") 

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging

import os
import numpy as np
import pandas as pd 
import shap

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_style('whitegrid')

import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorcet as cc

# ================================= GLOBAL VARIABLES ================================= #

largecolorset = list(
    cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark
)

# Define the plot template
pio.templates["heather"] = go.layout.Template(
    layout={
        'title': {
            'font': {
                'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif', 
                'size': 30, 
                'color': '#000'
            }
        }, 
        'font': {
            'family': 'Helvetica Neue, Helvetica, Sans-serif', 
            'size': 16, 
            'color': '#000'
        }, 
        'paper_bgcolor': 'rgba(0, 0, 0, 0)', 
        'plot_bgcolor': '#fff', 
        'colorway': largecolorset, 
        'xaxis': {'showgrid': False}, 
        'yaxis': {'showgrid': False}
    }
)

#from workflow_16s.figures.legends import marker_color_map, plot_legend
logger = logging.getLogger('workflow_16s')

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
    shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=cbmpf.classes_)
    shap.summary_plot(shap_values, X_train.values, feature_names = X_train.columns)
    
    sns.set_context("talk", font_scale=0.5)
    fea_imp = pd.DataFrame({
      'imp': cbmpf.feature_importances_, 
      'col': X_test.columns
    })
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False])
    fea_imp=fea_imp[fea_imp['imp']>0.5]
    fea_imp['fn']=['fn:'+str(i) for i in fea_imp.index]
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
  
