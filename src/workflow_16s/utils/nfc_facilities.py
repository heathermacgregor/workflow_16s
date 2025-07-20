# ===================================== IMPORTS ====================================== #

import logging
import os
import requests
import time
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix, classification_report

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N: int = 65  # Length of description for progress bar

DEFAULT_NFCIS_PATH = '/usr2/people/macgregor/amplicon/NFCISFacilityList.xlsx' #'../../../references/NFCISFacilityList.xlsx'
DEFAULT_GEM_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/references/gem_nuclearpower_2024-07.tsv' #'../../../references/gem_nuclearpower_2024-07.tsv'

NFCIS_COLUMNS = {
    'country': "Country",
    'facility': "Facility Name",
    'facility_type': "Facility Type",
    'facility_capacity': "Design Capacity",
    'facility_status': "Facility Status",
    'facility_start_year': "Start of Operation",
    'facility_end_year': "End of Operation"
}
GEM_COLUMNS = {
    'country': "Country/Area",
    'facility': "Project Name",
    'facility_type': "Reactor Type",
    'facility_capacity': "Capacity (MW)",
    'facility_status': "Status",
    'facility_start_year': "Start Year",
    'facility_end_year': "Retirement Year"
}

# ==================================== FUNCTIONS ===================================== #  

def process_and_geocode_db(
    database: str = "GEM",
    file_path: str = DEFAULT_GEM_PATH, 
    user_agent="MyGeocodingApp/1.0", 
):
    """
    Process a data file (Excel or TSV) and add latitude/longitude coordinates
    
    Args:
        file_path (str): Path to input file (.xlsx or .tsv)
        user_agent (str): Custom user agent for API requests
    
    Returns:
        pd.DataFrame: Processed DataFrame with coordinates
    """
    if database == "GEM":
        skip_rows = 0
        skip_first_col = False
        column_names = DEFAULT_GEM_COLUMNS
        
    elif database == "NFCIS":
        skip_rows = 8
        skip_first_col = True
        column_names = DEFAULT_NFCIS_COLUMNS
        
    # Detect file type and load data
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in ['.xlsx', '.xls']:
            raw_df = pd.read_excel(file_path, header=None)
        elif ext in ['.tsv', '.txt']:
            raw_df = pd.read_csv(file_path, sep='\t', header=None, encoding_errors='replace')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        # Try to load as TSV if Excel fails
        try:
            raw_df = pd.read_csv(file_path, sep='\t', header=None, encoding_errors='replace')
        except Exception as e2:
            raise ValueError(f"Failed to read file: {e}\nSecondary error: {e2}")

    # Skip rows and columns based on parameters
    if skip_first_col:
        df = raw_df.iloc[skip_rows:, 1:].copy()
    else:
        df = raw_df.iloc[skip_rows:, :].copy()
    
    # Set header from first row after skipping
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    logger.info(f"Loaded data shape: {df.shape}")

    # Filter and rename columns
    df = df[list(column_names.values())]
    df = df.rename(columns={v: k for k, v in column_names.items()})
    df = df[list(column_names.keys())]
    
    # Verify required columns exist
    required_cols = ['facility', 'country'] 
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Geocoding function
    def geocode_location(facility, country):
        """Get coordinates from Nominatim API"""
        query = f"{facility}, {country}"
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': query, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': user_agent}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            print(f"Geocoding failed for '{query}': {str(e)}")
        return None, None
    
    # Add coordinate columns
    df['latitude'], df['longitude'] = None, None
    
    # Geocode with rate limiting
    with get_progress_bar() as progress:
        task_desc = "Geocoding facilities..."
        task = progress.add_task(
            f"[white]{task_desc:<{DEFAULT_N}}", 
            total=df.shape[0]
        )
        for i, row in df.iterrows():
            facility = str(row[facility_col])
            country = str(row[country_col])
            try:
                if facility and country and facility != 'nan' and country != 'nan':
                    df.at[i, 'latitude_deg'], df.at[i, 'longitude_deg'] = geocode_location(facility, country)
                #time.sleep(1)  # Respect API rate limits
            except Exception as e:
                logger.error(f"Geocoding failed for facility '{facility}': {e!r}")
            finally:
                progress.update(task, advance=1)
    
    return df


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (in km) between two points 
    on the Earth using the Haversine formula.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    return c * r

def match_facilities_to_locations(
    facilities: pd.DataFrame, 
    samples: pd.DataFrame, 
    max_distance_km: float = 50):
    """
    Match locations to nearby facilities within a specified distance threshold.
    
    Args:
        facilities_df (pd.DataFrame): Facilities dataframe with 'latitude' and 'longitude' columns
        locations_df (pd.DataFrame): Locations dataframe with 'latitude_deg' and 'longitude_deg' columns
        max_distance_km (float): Maximum distance in kilometers for matching (default=50)
    
    Returns:
        pd.DataFrame: Modified locations_df with appended facility information
    """
    # Create a copy to avoid modifying original dataframe
    result_df = samples[['latitude_deg', 'longitude_deg']].copy()
    
    # Precompute valid facilities (non-null coordinates)
    valid_facilities = facilities.dropna(subset=['latitude_deg', 'longitude_deg'])
    facilities_coords = valid_facilities[['latitude_deg', 'longitude_deg']].values
    facilities_data = valid_facilities.to_dict('records')
    
    # Initialize columns for facility data
    for col in facilities.columns:
        result_df[f'facility_{col}'] = np.nan
    
    # Initialize match status column
    result_df[f'facility_within_{max_distance_km}_km'] = False
    result_df['facility_distance_km'] = np.nan
    
    # Iterate through each location
    for idx, row in result_df.iterrows():
        lat, lon = row['latitude_deg'], row['longitude_deg']
        # Skip if location coordinates are missing
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Find the closest facility
        min_distance = float('inf')
        closest_facility = None
        for facility_coord, facility_record in zip(facilities_coords, facilities_data):
            fac_lat, fac_lon = facility_coord
            distance = haversine(lat, lon, fac_lat, fac_lon)
            if distance < min_distance:
                min_distance = distance
                closest_facility = facility_record
        
        # Check if closest facility is within threshold
        if min_distance <= max_distance_km and closest_facility is not None:
            # Add facility data to the location row
            for col, value in closest_facility.items():
                result_df.at[idx, f'facility_{col}'] = value
            
            # Add distance and match status
            result_df.at[idx, 'facility_distance_km'] = min_distance
            result_df.at[idx, 'facility_match'] = True
    
    return result_df
    

def find_nearby_nfc_facilities(
    cfg: Dict,
    meta: pd.DataFrame
):
    databases = cfg.get("nfc_facilities", {}).get("databases", [{'name': "NFCIS"}, {'name': "GEM"}])
    facilities_dfs = []
    for db in databases:
        name = db['name']
        if name == "NFCIS":
            file_path = DEFAULT_NFCIS_PATH
        elif name == "GEM":
            file_path = DEFAULT_GEM_PATH
        facilities_df = process_and_geocode_db(database=name, file_path=file_path)
        facilities_dfs.append(facilities_df)
    facilities_df = pd.concat(facilities_dfs, ignore_index=True)
    
    logger.info(f"Merged facilities: {facilities_df.shape}")
    
    max_distance_km=cfg.get("nfc_facilities", {}).get("max_distance_km", 50)
    matched_df = match_facilities_to_locations(
        facilities_df, 
        meta.set_index('#sampleid'),
        max_distance_km=max_distance_km
    )
    df.to_csv(f"/usr2/people/macgregor/amplicoin/test/facility_matches_{max_distance_km}km.tsv", sep='\t', index=True)

    return matched_df.reset_index()


def analyze_contamination_correlation(
    df: pd.DataFrame, 
    threshold: float = 0.5
):
    """
    Analyzes correlation between facility proximity and contamination status
    
    Args:
        df (pd.DataFrame): DataFrame containing 'facility_match' and 
                           'nuclear_contamination_status' columns
        threshold (float): Probability threshold for contamination status
                           (default=0.5)
    
    Returns:
        dict: Dictionary containing analysis results and metrics
    """
    # Validate required columns
    required_cols = ['facility_match', 'nuclear_contamination_status']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Create clean working copy
    analysis_df = df.set_index('#sampleid')[required_cols].copy().dropna()
    
    # Convert contamination status to boolean
    if analysis_df['nuclear_contamination_status'].dtype in ['int64', 'float64']:
        analysis_df['contaminated'] = (
            analysis_df['nuclear_contamination_status'] > threshold
        )
    elif analysis_df['nuclear_contamination_status'].dtype == 'object':
        positive_indicators = ['contaminated', 'positive', 'high', 'yes', 'true']
        analysis_df['contaminated'] = analysis_df['nuclear_contamination_status'].str.lower().isin(positive_indicators)
    else:
        analysis_df['contaminated'] = analysis_df['nuclear_contamination_status'].astype(bool)
    
    # Prepare boolean series
    facility_nearby = analysis_df['facility_match'].astype(bool)
    is_contaminated = analysis_df['contaminated']
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        is_contaminated, 
        facility_nearby,
        labels=[False, True]
    ).ravel()
    
    # Calculate key metrics
    total = len(analysis_df)
    contamination_rate = is_contaminated.mean()
    facility_presence_rate = facility_nearby.mean()
    
    # Calculate correlation metrics
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    relative_risk = (tp / (tp + fp)) / (fn / (fn + tn)) if (fn + tn) > 0 else float('nan')
    
    # Generate classification report
    report = classification_report(
        is_contaminated, 
        facility_nearby,
        target_names=['Not Contaminated', 'Contaminated'],
        output_dict=True
    )
    
    return {
        'summary_metrics': {
            'total_locations': total,
            'contamination_rate': contamination_rate,
            'facility_presence_rate': facility_presence_rate,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'relative_risk': relative_risk
        },
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        },
        'contingency_table': pd.crosstab(
            facility_nearby, 
            is_contaminated,
            rownames=['Facility Nearby'],
            colnames=['Contaminated'],
            margins=True
        ).to_dict(),
        'classification_report': report
    }
