# ===================================== IMPORTS ====================================== #

from typing import Any, List, Tuple, Union
from pathlib import Path

import shutil
import pandas as pd

import logging
logger = logging.getLogger('workflow_16s')


# ==================================== FUNCTIONS ===================================== #

def create_dir(dir_path: Union[str, Path]) -> None:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {dir_path}")


def remove_dir(dir_path: Union[str, Path]) -> None:
    dir_path = Path(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)
        logger.info(f"Directory removed: {dir_path}")


class SubDirs:
    def __init__(self, dir_path: Union[str, Path]):
        self.main = Path(dir_path)
        self.logs = self.main / 'logs'
        self.tmp = self.main / 'tmp'
        self.data = self.main / 'data'
        self.data_per_dataset = self.data / 'per_dataset'
        self.metadata_per_dataset = self.data_per_dataset / 'metadata'
        self.seq_data_per_dataset = self.data_per_dataset / 'seqs'
        self.raw_seq_data_per_dataset = self.seq_data_per_dataset / 'raw'
        self.trimmed_seq_data_per_dataset = self.seq_data_per_dataset / 'trimmed'
        self.qiime_data_per_dataset = self.data_per_dataset / 'qiime'
        self.final = self.main / 'final'
        self.tables = self.final / 'tables'
        self.figures = self.final / 'figures'
        self.create_dirs()

    def create_dirs(self):
        dirs = [
            self.main,
            self.logs,
            self.tmp,
            self.data,
            self.data_per_dataset,
            self.metadata_per_dataset,
            self.seq_data_per_dataset,
            self.raw_seq_data_per_dataset,
            self.trimmed_seq_data_per_dataset,
            self.qiime_data_per_dataset,
            self.final,
            self.tables,
            self.figures,
        ]
        for _dir in dirs:
            _dir.mkdir(parents=True, exist_ok=True)
    
    def dataset_dirs(self, dataset: str):
        dataset_dirs = {}
        dirs = {
            'tmp': self.tmp,
            'metadata': self.metadata_per_dataset,
            'raw_seqs': self.raw_seq_data_per_dataset,
            'trimmed_seqs': self.trimmed_seq_data_per_dataset,
            'qiime': self.qiime_data_per_dataset,
        }
        for name, _dir in dirs.items():
            dataset_dir = _dir / dataset
            dataset_dirs[name] = dataset_dir
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
        return dataset_dirs
    
