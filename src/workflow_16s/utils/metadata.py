# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import pandas as pd

# Local Imports
from workflow_16s.constants import GROUP_COLUMNS, SAMPLE_ID_COLUMN
logger = logging.getLogger("workflow_16s")
# ==================================================================================== #

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    group_columns: List[Dict] = GROUP_COLUMNS,
    columns_to_rename: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:          Path to metadata TSV file.
        group_columns:     [Placeholder]
        columns_to_rename: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")
    
    # Load DataFrame from TSV
    df = pd.read_csv(tsv_path, sep='\t')
  
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    sample_id_col = next((col 
                          for col in ['run_accession', '#sampleid', 'sample-id'] 
                          if col in df.columns), None)
    df['SAMPLE ID'] = (df[sample_id_col] 
                       if sample_id_col 
                       else [f"{tsv_path.parents[5].name}_x{i}" 
                             for i in range(1, len(df)+1)])

    dataset_id_col = next((col 
                           for col in ['project_accession', 'dataset_id', 'dataset_name'] 
                           if col in df.columns), None)
    df['DATASET ID'] = (df[dataset_id_col] 
                        if dataset_id_col 
                        else tsv_path.parents[5].name)
  
    for col in group_columns:
        name = col.get('name')
        type = col.get('type')
        if type == 'bool' and name and name not in df.columns:
            df[name] = False

    if columns_to_rename is not None:
        for old, new in columns_to_rename:
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

    return df

# ==================================================================================== #

def get_group_column_values(group_column: Union[str, Dict], 
                            metadata: pd.DataFrame) -> List[Any]:
    """Extract values from group column.
    
    Args:
        group_column: [Placeholder]
        metadata:     [Placeholder]
        
    Returns:
        List [Placeholder]
    """
    if isinstance(group_column, dict):
        if 'values' in group_column and group_column['values']:
            return group_column['values']
        
        if 'type' in group_column and group_column['type'] == 'bool':
            return [True, False]
        
        if 'name' in group_column and group_column['name'] in metadata.columns:
            return metadata[group_column['name']].drop_duplicates().tolist()
    elif isinstance(group_column, str):
        if group_column in metadata.columns:
            return metadata[group_column].drop_duplicates().tolist()
    else:
        return []


def import_merged_metadata_tsv(
    tsv_paths: List[Union[str, Path]],
    columns_to_rename: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Merge multiple metadata files into a single DataFrame.
    
    Args:
        tsv_paths:         List of paths to metadata TSV files.
        columns_to_rename: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Concatenated metadata DataFrame.
    
    Raises:
        FileNotFoundError: If no valid metadata files could be loaded.
    """
    dfs: List[pd.DataFrame] = []
    with get_progress_bar() as progress:
        task_desc = "Loading metadata files"
        task = progress.add_task(_format_task_desc(task_desc), total=len(tsv_paths))
        for tsv_path in tsv_paths:
            try:
                dfs.append(import_metadata_tsv(tsv_path, columns_to_rename))
            except Exception as e:
                logger.error(f"Loading metadata failed for {tsv_path}: {e!r}")
            finally:
                progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError("No valid metadata files loaded. Check paths and file formats.")

    return pd.concat(dfs, ignore_index=True)

# ==================================================================================== #

class MetadataCleaner:
    """[Placeholder]"""
    # Precompile regex patterns for efficiency
    NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
    LETTER_PATTERN = re.compile(r'[NnSsEeWw]')

    # Define potential source columns
    lat_sources = ['lat', 'lat_study', 'latitude_deg_ena', 'latitude_deg.1']
    lon_sources = ['lon', 'lon_study', 'longitude_deg.1']
    pair_sources = ['lat_lon', 'location', 'location_ena', 'location_start', 
                    'location_end', 'location_start_study', 'location_end_study']
    def __init__(self, config: Dict, metadata: pd.DataFrame):
        self.df = metadata
        # Handle empty metadata
        if self.df.empty:
            return 
        self.config = config
        self.sample_id_column = self.config.get("metadata_id_column", SAMPLE_ID_COLUMN)

    def run_all(self):  
        """[Placeholder]"""
        self._clean_columns()
        self._clean_sample_ids()
        self._collapse_suffix_columns(suffix='_ena')
        self._collapse_ph_columns()
        self._fill_missing_coordinates()

    def _clean_columns(self) -> None:
        """Remove duplicate columns."""
        if self.df.columns.duplicated().any():
            duplicated_columns = self.df.columns[self.df.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]

    def _clean_sample_ids(self) -> None:
        """Normalize sample IDs and remove duplicates."""
        # Validate sample column exists
        col = self.sample_id_column
        if col not in self.df.columns:
            raise ValueError(f"Sample ID column '{col}' not found in metadata")
        
        self.df[col] = self.df[col].astype(str).str.lower()
        self.df = self.df.drop_duplicates(subset=[col])

    def _collapse_suffix_columns(self, suffix: str = '_ena'):
        df = self.df.copy()
        # Identify all columns ending with the suffix
        columns = [col for col in df.columns if col.endswith(suffix)]
        # Sort columns by length in descending order to handle nested suffixes
        columns = sorted(columns, key=len, reverse=True)
        
        for col in columns:
            # Skip if the column is exactly the suffix 
            if col == suffix:
                continue
            # Determine the base column name by removing the suffix
            base_col = col[:-len(suffix)]
            if base_col in df.columns:
                # Combine values: prioritize base_col, fill missing from col
                df[base_col] = df[base_col].combine_first(df[col])
            else:
                # Create base_col from ena_col if it doesn't exist
                df[base_col] = df[col]
            # Drop the col after processing
            df = df.drop(columns=[col])
        self.df = df
      
    def _collapse_ph_columns(self) -> None:
        """Collapses all columns in the DataFrame that start with 'ph' followed by a non-alphabet 
        character or exactly 'ph' into a single 'ph' column. The first non-null value from these 
        columns is retained for each row."""
        # Compile regex pattern to match columns starting with 'ph' followed by non-alphabet or exactly 'ph'
        pattern = re.compile(r'^ph[^a-zA-Z]|^ph$')
        ph_columns = [col for col in self.df.columns if pattern.match(col)]
        
        # Return original dataframe if no ph_columns found
        if not ph_columns:
            return self.df
        
        # Prioritize exact 'ph' column if present
        if 'ph' in ph_columns:
            ph_columns.remove('ph')
            ph_columns = ['ph'] + sorted(ph_columns)
        else:
            ph_columns = sorted(ph_columns)
        
        # Create a temporary DataFrame with the selected columns
        temp_df = self.df[ph_columns]
        
        # Backfill values along rows and take the first column to get the first non-null value
        new_ph = temp_df.bfill(axis=1).iloc[:, 0]
        
        # Drop original ph_columns and add the new coalesced column
        self.df = self.df.drop(columns=ph_columns)
        self.df['ph'] = new_ph

    def _extract_lat_lon(self, s: str) -> Tuple:
        """Extract latitude and longitude from a string using precompiled regex patterns.
        
        Args:
            s: Input string containing coordinate information.
            
        Returns:
            (latitude, longitude) as floats, or (None, None) if extraction fails.
        """
        if not isinstance(s, str):
            return None, None
            
        # Find all number matches
        matches = list(self.NUM_PATTERN.finditer(s))
        if len(matches) < 2:
            return None, None
            
        try:
            num1 = float(matches[0].group())
            num2 = float(matches[1].group())
        except (ValueError, TypeError):
            return None, None
            
        # Check for directional letters near the numbers
        letters = []
        for match in matches[:2]:
            start, end = match.span()
            window = s[max(0, start-3):min(len(s), end+3)]
            letter_match = self.LETTER_PATTERN.search(window)
            letters.append(letter_match.group().upper() if letter_match else None)
        
        # Process coordinates based on directional letters
        coords = {}
        for i, (num, letter) in enumerate(zip([num1, num2], letters)):
            if not letter:
                continue
            if letter in ['N', 'S']:
                coords['lat'] = num if letter == 'N' else -num
            else:  # E or W
                coords['lon'] = num if letter == 'E' else -num
        
        # Return results based on what we found
        if 'lat' in coords and 'lon' in coords:
            return coords['lat'], coords['lon']
        if 'lat' in coords:
            return coords['lat'], num2
        if 'lon' in coords:
            return num1, coords['lon']
        
        return num1, num2
    
    def _fill_missing_coordinates(self) -> None:
        """Fill missing latitude/longitude values by extracting from alternative columns."""
        # Ensure required columns exist
        if 'latitude_deg' not in self.df.columns:
            self.df['latitude_deg'] = np.nan
        if 'longitude_deg' not in self.df.columns:
            self.df['longitude_deg'] = np.nan
            
        # Filter to existing columns only
        existing_lat = [col for col in self.lat_sources if col in self.df.columns]
        existing_lon = [col for col in self.lon_sources if col in self.df.columns]
        existing_pair = [col for col in self.pair_sources if col in self.df.columns]
        
        # Identify rows needing processing
        missing_mask = self.df['latitude_deg'].isna() | self.df['longitude_deg'].isna()
        missing_count = missing_mask.sum()
        
        if not missing_count:
            return  # Return early if nothing to process
        
        logger.info(f"Processing {missing_count} rows with missing coordinates...")
        
        # Process missing rows
        for idx in self.df.index[missing_mask]:
            row = self.df.loc[idx]
            new_lat, new_lon = row['latitude_deg'], row['longitude_deg']
            
            # First try pair columns (contain both coordinates)
            for col in existing_pair:
                if pd.isna(new_lat) or pd.isna(new_lon):
                    if pd.notna(row[col]) and row[col] != '':
                        lat_val, lon_val = self._extract_lat_lon(str(row[col]))
                        if pd.isna(new_lat) and lat_val is not None:
                            new_lat = lat_val
                        if pd.isna(new_lon) and lon_val is not None:
                            new_lon = lon_val
            
            # Then try single-value columns
            if pd.isna(new_lat):
                for col in existing_lat:
                    if pd.notna(row[col]) and row[col] != '':
                        try:
                            new_lat = float(row[col])
                            break
                        except (ValueError, TypeError):
                            continue
            
            if pd.isna(new_lon):
                for col in existing_lon:
                    if pd.notna(row[col]) and row[col] != '':
                        try:
                            new_lon = float(row[col])
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Update the DataFrame
            self.df.at[idx, 'latitude_deg'] = new_lat
            self.df.at[idx, 'longitude_deg'] = new_lon

# ==================================================================================== #

def clean_metadata(metadata: pd.DataFrame):
    """[Placeholder]"""
    cleaner = MetadataCleaner(
        config=self.config, 
        metadata=metadata
    )
    cleaner.run_all()
    return cleaner.df
