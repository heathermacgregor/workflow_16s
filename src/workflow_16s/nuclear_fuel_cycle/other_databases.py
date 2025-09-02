import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from workflow_16s.constants import (
    DEFAULT_GEM_PATH, DEFAULT_GEM_COLUMNS, DEFAULT_NFCIS_PATH, DEFAULT_NFCIS_COLUMNS,
    REFERENCES_DIR
)


class NFCFacilityDB:
    DBConfig = {
        "GEM": (0, False, DEFAULT_GEM_COLUMNS, DEFAULT_GEM_PATH),
        "NFCIS": (8, True, DEFAULT_NFCIS_COLUMNS, DEFAULT_NFCIS_PATH)
    }
    def __init__(
        self, 
        databases: List[str] = ["GEM", "NFCIS"], 
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
    ):
        self.databases = databases
        self.output_dir = output_dir
        self.result = None

    def _process_dbs(self):
        dfs = []
        for database in self.databases:
            if database in list(DBConfig.keys()):
                file_path, skip_rows, skip_first_col, column_names = DBConfig[database]
            else:
                raise ValueError(f"Unknown database: {database}")

            # Detect and load file (only needed columns)
            ext = os.path.splitext(file_path)[1].lower()
            use_cols = None
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
            logger.info(f"Loaded '{database}' data with {df.shape[0]} NFC facilities")

            # Filter and rename
            df = df[list(column_names.values())]
            df = df.rename(columns={v: k for k, v in column_names.items()})
            df = df[list(column_names.keys())]
            df['data_source'] = database
            dfs.append(df)
          
        if dfs:
            self.result = pd.concat(dfs, ignore_index=True).dropna(subset=['latitude_deg', 'longitude_deg'])
            if self.output_dir:
                tsv_path = Path(self.output_dir) / f"nfc_facilities{'_'.join(self.databases)}.tsv"
                self.result.to_csv(tsv_path, sep='\t', index=True)
        else:
            self.result = pd.DataFrame()
          
        return self.result


# API
def load_nfc_facilities(
    config: Dict, 
    output_dir: Optional[Union[str, Path]] = REFERENCES_DIR
) -> pd.DataFrame:
    databases = config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    use_local = config.get("nfc_facilities", {}).get('use_local', False)
    if output_dir:
        tsv_path = Path(output_dir) / f"nfc_facilities{'_'.join(db['name'] for db in databases)}.tsv"
    if use_local and tsv_path.exists():
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        db_loader = NFCFacilityDB(databases=databases, output_dir=output_dir)
        df = db_loader._process_dbs()
    
    logger.info(f"NFC facilities from databases ({', '.join(databases)}): {df.shape}")
    return df
  
