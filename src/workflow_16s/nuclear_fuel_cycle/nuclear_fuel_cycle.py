# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import requests
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Third Party Imports
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Local Imports
from workflow_16s.constants import DEFAULT_USER_AGENT, MINDAT_API_KEY, REFERENCES_DIR
from workflow_16s.nuclear_fuel_cycle import mindat, wikipedia, other_databases, utils 
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

mindat_columns_to_keep = [
    "facility", "country", "latitude", "longitude", "elements", "refs", "wikipedia", 
    "data_source"
]
wikipedia_columns_to_keep = [
    "facility", "country", "facility_start_year", "facility_end_year", "lat_lon", 
    "location", "data_source", "wikipedia", "wikitable"
]

class NFCFacilitiesHandler:
    """Handler for managing Nuclear Fuel Cycle (NFC) facilities data.
    
    This class handles the retrieval, geocoding, and matching of NFC facilities
    with sample metadata based on geographical coordinates.
    
    Attributes:
        config:                 Configuration dictionary containing settings.
        output_dir:             Directory path for output files.
        mindat_api_key:         API key for MinDat API access.
        user_agent:             User agent string for web requests.
        verbose:                Flag for verbose logging.
        databases:              List of database configurations.
        database_names:         Names of enabled databases.
        max_distance_km:        Maximum distance in km for facility matching.
        use_local:              Flag to use locally cached data.
        facilities_output_path: Path to facilities output CSV.
        matches_output_path:    Path to facility matches output TSV.
    """
    def __init__(
        self, 
        config: Dict,        
        output_dir: Optional[Union[str, Path]] = REFERENCES_DIR, 
        mindat_api_key: str = MINDAT_API_KEY,
        user_agent: str = DEFAULT_USER_AGENT
    ):
        """Initialize NFC facilities handler.
        
        Args:
            config:        Configuration dictionary containing NFC facilities settings.
            output_dir:     Output directory path. Defaults to REFERENCES_DIR.
            mindat_api_key: Mindat API key. Defaults to MINDAT_API_KEY.
            user_agent:     User agent string for web requests. Defaults to DEFAULT_USER_AGENT.
        """
        self.config = config
        enabled = self.config.get("nfc_facilities", {}).get("enabled", True) # TODO: Switch to False
        if not enabled:
            return

        self.verbose = self.config.get("verbose", False)
        
        self.databases = self.config.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
        self.database_names = [db['name'] for db in self.databases]
        
        self.max_distance_km = self.config.get("nfc_facilities", {}).get("max_distance_km", 50)

        self.use_local = self.config.get("nfc_facilities", {}).get('use_local', False)

        self.output_dir = Path(output_dir) / "nfc_facilities"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.facilities_output_path = Path(self.output_dir) / 'facilities_raw.tsv'
        self.facilities_geocoded_output_path = Path(self.output_dir) / 'facilities.tsv'
        self.matches_output_path = Path(self.output_dir) / f"facility_matches_{self.max_distance_km}km.tsv"
        self.matches_only_output_path = Path(self.output_dir) / f"facility_matches_only_{self.max_distance_km}km.tsv"
        
        self.mindat_api_key = mindat_api_key
        self.user_agent = user_agent
        
    def log(self, msg):
        """Log message with debug level if verbose mode is enabled."""
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)
        
    def run(self, metadata: pd.DataFrame):
        """Execute the complete NFC facilities processing pipeline."""
        if self.use_local and self.facilities_output_path.exists():
            df = pd.read_csv(self.facilities_output_path, sep='\t')
        else:
            df = self._get_geocoded_data()
            
        if self.use_local and self.matches_output_path.exists():   
            updated_metadata = pd.read_csv(self.matches_output_path, sep='\t')
        else:
            updated_metadata = self._match_facilities_with_samples(facilities_df=df, samples_df=metadata)
    
        return df, updated_metadata

    def _get_geocoded_data(self):
        """Retrieve and geocode facilities data from configured databases."""
        df = self._get_data()
        df = self._geocode(df)
        self.nfc_facilities = df
        return df
        
    def _get_data(self):
        """Aggregate facilities data from all enabled databases."""
        database_dfs = []
        if "GEM" in self.database_names or "NFCIS" in self.database_names:
            other_databases_results = other_databases.load_nfc_facilities(config=self.config, output_dir=self.output_dir)
            logger.info(other_databases_results.columns)
            database_dfs.append(other_databases_results)
        if "MinDat" in self.database_names:
            mindat_results, _ = mindat.world_uranium_mines(self.config, self.mindat_api_key, self.output_dir)
            logger.info(mindat_results.columns)
            database_dfs.append(mindat_results[mindat_columns_to_keep])
        if "Wikipedia" in self.database_names:
            wikipedia_results = wikipedia.world_nfc_facilities(config=self.config, output_dir=self.output_dir)
            logger.info(wikipedia_results.columns)
            database_dfs.append(wikipedia_results[wikipedia_columns_to_keep])
        
        dfs = [df for df in database_dfs if isinstance(df, pd.DataFrame)]            
        facilities_df = pd.concat(dfs, axis=0)
        facilities_df = facilities_df.sort_values(by='facility')
        facilities_df = facilities_df.reindex(sorted(facilities_df.columns), axis=1)
        logger.info(facilities_df.columns)
        if self.facilities_output_path:
            facilities_df.to_csv(self.facilities_output_path, sep='\t', index=False)
        return facilities_df

    def _geocode(self, df: pd.DataFrame):
        """Geocode facility locations using nominatim OpenStreetMap.
        
        Args:
            df: Input DataFrame with facility and country information.
            
        Returns:
            pd.DataFrame: DataFrame with added latitude and longitude columns.
        """
        # Prepare geocoding
        df['__query__'] = df['facility'].fillna('') + ', ' + df['country'].fillna('')
        unique_queries = df['__query__'].unique()
    
        # Geocode unique queries with progress
        coords = {}
        with get_progress_bar() as progress:
            task = progress.add_task( 
                _format_task_desc("Geocoding unique locations"), 
                total=len(unique_queries)
            )
            for q in unique_queries:
                coords[q] = utils._geocode_query(q, self.user_agent)
                time.sleep(1) 
                progress.update(task, advance=1)
    
        # Map coords back to DataFrame
        df['latitude_deg']  = df['__query__'].map(lambda q: coords[q][0])
        df['longitude_deg'] = df['__query__'].map(lambda q: coords[q][1])
        df.drop(columns='__query__', inplace=True)
        if self.facilities_output_path:
            df.to_csv(self.facilities_geocoded_output_path, sep='\t', index=False)
        return df

    def _match_facilities_with_samples(
        self,
        facilities_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match samples with nearby facilities within the specified distance.
        
        Args:
            facilities_df: DataFrame containing geocoded facilities.
            samples_df:    DataFrame containing sample metadata with coordinates.
            
        Returns:
            pd.DataFrame: Updated metadata with facility match information.
        """
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
        if self.matches_output_path:
            matched_df[result_cols].to_csv(self.matches_only_output_path, sep='\t', index=False)
            matched_df.to_csv(self.matches_output_path, sep='\t', index=False)
        return matched_df

    def _match_facilities_with_locations(
        self,
        facilities_df: pd.DataFrame,
        samples_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match sample locations to nearby facilities using KD-tree spatial indexing.
        
        Args:
            facilities_df: Geocoded facilities DataFrame.
            samples_df:    Sample metadata DataFrame with coordinates.
            
        Returns:
            pd.DataFrame: Combined DataFrame with facility match results.
        """
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
                'longitude_deg': 'facility_longitude_deg',
                'country': 'facility_country'
            })
            return pd.concat([samples_df, matches], axis=1)
    
        # Prepare facility KD-tree
        valid_fac = facilities_df.dropna(subset=['latitude_deg', 'longitude_deg']).reset_index(drop=True)
        fac_xyz = utils.sph2cart(valid_fac['latitude_deg'], valid_fac['longitude_deg'])
        tree = cKDTree(fac_xyz)
    
        # Build sample coordinates
        samp_xyz = utils.sph2cart(valid_samples['latitude_deg'], valid_samples['longitude_deg'])
        
        # Convert max distance from km to the coordinate system units
        # Assuming Earth radius of 6371 km and unit sphere coordinates
        max_distance_units = self.max_distance_km / 6371.0
        
        dists, idxs = tree.query(samp_xyz, distance_upper_bound=max_distance_units)
    
        # Build result records
        records = []
        # First handle valid_samples
        for dist, idx in zip(dists, idxs):
            if np.isfinite(dist):
                rec = valid_fac.iloc[idx].to_dict()
                # Convert distance back to kilometers
                distance_km = dist * 6371.0
                rec.update({'facility_distance_km': distance_km, 'facility_match': True})
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
        
        return pd.concat([samples_df, matches_df.reset_index(drop=True)], axis=1)

# ==================================================================================== #

def update_nfc_facilities_data(
    config: Dict, 
    metadata: pd.DataFrame, 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to update NFC facilities data and match with samples.
    
    Args:
        config:   Configuration dictionary.
        metadata: Sample metadata DataFrame with coordinates.
        verbose:  Verbosity flag.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - NFC facilities DataFrame
            - Updated metadata with facility matches
    """
    handler = NFCFacilitiesHandler(config=config)
    nfc_facilities, updated_metadata = handler.run(metadata=metadata)
    if verbose:
        logger.info(nfc_facilities)
        logger.info(updated_metadata)
    return nfc_facilities, updated_metadata
    
