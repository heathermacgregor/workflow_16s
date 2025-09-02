# Standard Imports
import requests
import time  # Added missing import
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union
import logging  # Added for logging

# Third Party Imports
import pandas as pd
import numpy as np  # Added missing import
from scipy.spatial import cKDTree

# Local Imports
from workflow_16s.constants import DEFAULT_USER_AGENT, REFERENCES_DIR
from workflow_16s.nuclear_fuel_cycle import mindat, wikipedia, other_databases, utils 
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# Set up logger
logger = logging.getLogger(__name__)

class NFCFacilitiesHandler:
    def __init__(
        self, 
        config: Dict,        
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR, 
        user_agent: str = DEFAULT_USER_AGENT
    ):
        self.config = config
        enabled = self.config.get("nfc_facilities", {}).get("enabled", True) # TODO: Switch to False
        if not enabled:
            return
        
        self.databases = self.config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
        self.database_names = [db['name'] for db in self.databases]

        self.max_distance_km = self.config.get("nfc_facilities", {}).get("max_distance_km", 50)

        self.output_dir = output_dir
        self.user_agent = user_agent

    def run(self, metadata: pd.DataFrame):
        df = self._get_geocoded_data()
        updated_metadata = self._match_facilities_with_samples(facilities_df=df, samples_df=metadata)
        return df, updated_metadata

    def _get_geocoded_data(self):
        df = self._get_data()
        df = self._geocode(df)
        self.nfc_facilities = df
        return df
        
    def _get_data(self):
        database_dfs = []
        if "GEM" in self.database_names or "NFCIS" in self.database_names:
            database_dfs.append(other_databases.load_nfc_facilities(config=self.config))
        if "MinDat" in self.database_names:
            database_dfs.append(mindat.world_uranium_mines())
        if "Wikipedia" in self.database_names:
            database_dfs.append(wikipedia.world_nfc_facilities())
        facilities_df = pd.concat(database_dfs, axis=0)
        return facilities_df

    def _geocode(self, df: pd.DataFrame):
        # Prepare geocoding
        df['__query__'] = df['facility'].fillna('') + ', ' + df['country'].fillna('')
        unique_queries = df['__query__'].unique()
    
        # Geocode unique queries with progress
        coords = {}
        with get_progress_bar() as progress:
            task_desc = "Geocoding unique locations"
            task_desc_fmt = _format_task_desc(task_desc)
            task = progress.add_task(task_desc_fmt, total=len(unique_queries))
            for q in unique_queries:
                coords[q] = utils._geocode_query(q, self.user_agent)
                time.sleep(1) 
                progress.update(task, advance=1)
    
        # Map coords back to DataFrame
        df['latitude_deg']  = df['__query__'].map(lambda q: coords[q][0])
        df['longitude_deg'] = df['__query__'].map(lambda q: coords[q][1])
        df.drop(columns='__query__', inplace=True)
        return df

    def _match_facilities_with_samples(
        self,
        facilities_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        # Fixed: Call the correct method name
        matched_df = self._match_facilities_with_locations(facilities_df, samples_df)
        
        # Define required metadata columns to keep
        required_sample_cols = [
            'nuclear_contamination_status', 
            'dataset_name', 
            'country', 
            'latitude_deg', 
            'longitude_deg'
        ]
        
        # Get new facility columns (including renamed coordinates)
        new_cols = [col for col in matched_df.columns if col not in samples_df.columns]
        
        # Combine required metadata and new facility columns
        result_cols = [col for col in required_sample_cols if col in matched_df] + new_cols
        
        # Log warning if any required columns are missing
        missing_cols = set(required_sample_cols) - set(result_cols)
        if missing_cols:
            logger.warning(f"Missing required columns in output: {', '.join(missing_cols)}")
        
        # Save full matched results
        if self.output_dir:
            matched_df[result_cols].to_csv(
                Path(self.output_dir) / f"facility_matches_{self.max_distance_km}km.tsv",  # Fixed: use self.output_dir
                sep='\t', index=False
            )
        return matched_df

    def _match_facilities_with_locations(  # Fixed: Added self parameter
        self,
        facilities_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match locations to nearby facilities within a specified distance threshold.
        Handles missing coordinates by preserving original rows."""
        # Copy samples to preserve order and index
        samples_df = samples_df.reset_index(drop=True).copy()
    
        # Identify valid coordinates
        valid_mask = samples_df[['latitude_deg', 'longitude_deg']].notnull().all(axis=1)
        valid_samples = samples_df[valid_mask]
        invalid_samples = samples_df[~valid_mask]
    
        # If no valid samples, return all unmatched
        if valid_samples.empty:
            matches = pd.DataFrame([
                {
                    **{col: np.nan for col in facilities_df.columns}, 
                    'facility_distance_km': np.nan, 
                    'facility_match': False
                }
                for _ in range(len(samples_df))
            ])
            # Rename facility coordinate columns before returning
            matches = matches.rename(columns={
                'latitude_deg': 'facility_latitude_deg',
                'longitude_deg': 'facility_longitude_deg'
            })
            return pd.concat([samples_df, matches], axis=1)  # Fixed: samples_df instead of samples
    
        # Prepare facility KD-tree
        valid_fac = facilities_df.dropna(subset=['latitude_deg', 'longitude_deg']).reset_index(drop=True)
        fac_xyz = utils.sph2cart(valid_fac['latitude_deg'], valid_fac['longitude_deg'])
        tree = cKDTree(fac_xyz)
    
        # Build sample coordinates
        samp_xyz = utils.sph2cart(valid_samples['latitude_deg'], valid_samples['longitude_deg'])
        dists, idxs = tree.query(samp_xyz, distance_upper_bound=self.max_distance_km)
    
        # Build result records
        records = []
        # First handle valid_samples
        for dist, idx in zip(dists, idxs):
            if np.isfinite(dist):
                rec = valid_fac.iloc[idx].to_dict()
                rec.update({'facility_distance_km': dist, 'facility_match': True})
            else:
                rec = {col: np.nan for col in facilities_df.columns}
                rec.update({'facility_distance_km': np.nan, 'facility_match': False})
            records.append(rec)
    
        # Then handle invalid_samples: no match
        for _ in range(len(invalid_samples)):
            rec = {col: np.nan for col in facilities_df.columns}
            rec.update({'facility_distance_km': np.nan, 'facility_match': False})
            records.append(rec)
    
        # Combine matches in original order
        matches_df = pd.DataFrame(records)
        
        # Rename facility coordinate columns to avoid conflicts
        matches_df = matches_df.rename(columns={
            'latitude_deg': 'facility_latitude_deg',
            'longitude_deg': 'facility_longitude_deg',
            'country': 'facility_country'
        })
        
        return pd.concat([samples_df, matches_df.reset_index(drop=True)], axis=1)  # Fixed: samples_df instead of samples


# API
def update_nfc_facilities_data(config: Dict, metadata: pd.DataFrame):
    handler = NFCFacilitiesHandler(config=config)
    nfc_facilities, updated_metadata = handler.run(metadata=metadata)
    return nfc_facilities, updated_metadata
    
