# ==================================================================================== #

# Standard Imports
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third Party Imports
import pandas as pd

# Local Imports
from workflow_16s.constants import (
    DEFAULT_GEM_PATH, DEFAULT_GEM_COLUMNS, DEFAULT_NFCIS_PATH, DEFAULT_NFCIS_COLUMNS,
    REFERENCES_DIR
)

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class NFCFacilityDB:
    """A class that handles loading and processing of NFC facility databases.
    
    Attributes:
        databases:  List of database names to process.
        output_dir: Directory where processed results will be saved.
        result:     Combined DataFrame containing processed database information.
    """
    DBConfig = {
        "GEM": (0, False, DEFAULT_GEM_COLUMNS, DEFAULT_GEM_PATH),
        "NFCIS": (8, True, DEFAULT_NFCIS_COLUMNS, DEFAULT_NFCIS_PATH)
    }
    def __init__(
        self, 
        databases: List[str] = ["GEM", "NFCIS"], 
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
    ):
        """Initialize NFC facility database processor.
        
        Args:
            databases:  List of database names to process. Defaults to ["GEM", "NFCIS"].
            output_dir: Output directory for processed results. Defaults to REFERENCES_DIR.
        """
        self.databases = databases
        self.database_names = [db['name'] for db in self.databases]
        self.output_dir = output_dir
        self.result = None

    def _process_dbs(self):
        """Process configured databases and combine results."""
        dfs = []
        valid_dbs = [database for database in self.database_names if database in list(self.DBConfig.keys())]
        for name in valid_dbs:
            skip_rows, skip_first_col, column_names, file_path = self.DBConfig[name]
            # Detect and load file (only needed columns)
            ext = os.path.splitext(file_path)[1].lower()
            use_cols = column_names
            try:
                if ext in ['.xlsx', '.xls']:
                    df_raw = pd.read_excel(
                        file_path, header=None, 
                        skiprows=skip_rows, usecols=use_cols
                    )
                else:
                    df_raw = pd.read_csv(
                        file_path, sep='\t', header=None, 
                        skiprows=skip_rows, usecols=use_cols, 
                        encoding_errors='replace'
                    )
            except Exception as e:
                df_raw = pd.read_csv(
                    file_path, sep='\t', header=None, 
                    skiprows=skip_rows, 
                    encoding_errors='replace'
                )
        
            # Drop first column if needed
            df = df_raw.iloc[:, 1:] if skip_first_col else df_raw.copy()
        
            # Set header and reset
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            logger.info(f"Loaded '{name}' data with {df.shape[0]} NFC facilities")

            # Filter and rename
            df = df[list(column_names.values())]
            df = df.rename(columns={v: k for k, v in column_names.items()})
            df = df[list(column_names.keys())]
            df['data_source'] = name
            dfs.append(df)
          
        if dfs:
            self.result = pd.concat(dfs, ignore_index=True)
            if self.output_dir:
                tsv_path = Path(self.output_dir) / f"{'_'.join(valid_dbs)}.tsv"
                self.result.to_csv(tsv_path, sep='\t', index=True)
        else:
            self.result = pd.DataFrame()
          
        return self.result

# ==================================================================================== #

def load_nfc_facilities(
    config: Dict, 
    output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
) -> pd.DataFrame:
    """Load NFC facilities from configured databases.
    
    Args:
        config:     Configuration dictionary containing database settings.
        output_dir: Directory where processed results will be saved. Defaults to REFERENCES_DIR.
        
    Returns:
        DataFrame containing combined facilities from all configured databases.
        
    Note:
        Can use locally cached version if configured and available.
    """
    DBConfig = {
        "GEM": (0, False, DEFAULT_GEM_COLUMNS, DEFAULT_GEM_PATH),
        "NFCIS": (8, True, DEFAULT_NFCIS_COLUMNS, DEFAULT_NFCIS_PATH)
    }
    databases = config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    db_names = [db['name'] for db in databases]
    valid_dbs = [database for database in db_names if database in list(DBConfig.keys())]
    use_local = config.get("nfc_facilities", {}).get('use_local', False)
    if output_dir:
        tsv_path = Path(output_dir) / f"{'_'.join(valid_dbs)}.tsv"
    if use_local and tsv_path.exists():
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        db_loader = NFCFacilityDB(databases=databases, output_dir=output_dir)
        df = db_loader._process_dbs()
    
    logger.info(f"NFC facilities from databases ({', '.join(valid_dbs)}): {df.shape}")
    return df
  
