# ===================================== IMPORTS ====================================== #

import logging
import os
import requests
import time
from math import radians, sin, cos, asin, sqrt
from typing import Dict

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix, classification_report

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N: int = 65  # Length of description for progress bar

DEFAULT_NFCIS_PATH = '/usr2/people/macgregor/amplicon/NFCISFacilityList.xlsx' #'../../../references/NFCISFacilityList.xlsx'
DEFAULT_GEM_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/references/gem_nuclearpower_2024-07.tsv' #'../../../references/gem_nuclearpower_2024-07.tsv'

DEFAULT_NFCIS_COLUMNS = {
    'country': "Country",
    'facility': "Facility Name",
    'facility_type': "Facility Type",
    'facility_capacity': "Design Capacity",
    'facility_status': "Facility Status",
    'facility_start_year': "Start of Operation",
    'facility_end_year': "End of Operation"
}
DEFAULT_GEM_COLUMNS = {
    'country': "Country/Area",
    'facility': "Project Name",
    'facility_type': "Reactor Type",
    'facility_capacity': " Capacity (MW) ",
    'facility_status': "Status",
    'facility_start_year': "Start Year",
    'facility_end_year': "Retirement Year"
}

# ==================================== FUNCTIONS ===================================== #  

def process_and_geocode_db(database="GEM", file_path=DEFAULT_GEM_PATH, user_agent="MyGeocodingApp/1.0"):
    ...
    # Cache geocoding results to avoid redundant API calls
    geocode_cache = {}

    def geocode_location_cached(facility, country):
        key = (facility.strip().lower(), country.strip().lower())
        if key in geocode_cache:
            return geocode_cache[key]
        query = f"{facility}, {country}"
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': query, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': user_agent}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                coords = (float(data[0]['lat']), float(data[0]['lon']))
                geocode_cache[key] = coords
                return coords
        except Exception as e:
            logger.warning(f"Geocoding failed for '{query}': {e}")
        geocode_cache[key] = (None, None)
        return None, None

    lats, lons = [], []
    with get_progress_bar() as progress:
        task = progress.add_task(_format_task_desc("Geocoding facilities..."), total=len(df))
        for i, row in df.iterrows():
            facility = str(row['facility'])
            country = str(row['country'])
            lat, lon = geocode_location_cached(facility, country)
            lats.append(lat)
            lons.append(lon)
            progress.update(task, advance=1)

    df['latitude_deg'] = lats
    df['longitude_deg'] = lons
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

def match_facilities_to_locations(facilities, samples, max_distance_km=50):
    # Drop null coordinates
    facilities = facilities.dropna(subset=['latitude_deg', 'longitude_deg']).copy()
    samples = samples.dropna(subset=['latitude_deg', 'longitude_deg']).copy()

    fac_coords = np.radians(facilities[['latitude_deg', 'longitude_deg']].values)
    samp_coords = np.radians(samples[['latitude_deg', 'longitude_deg']].values)

    r = 6371  # Earth radius in km
    matched_data = []

    for i, (s_lat, s_lon) in enumerate(samp_coords):
        dlat = fac_coords[:, 0] - s_lat
        dlon = fac_coords[:, 1] - s_lon
        a = np.sin(dlat/2)**2 + np.cos(s_lat) * np.cos(fac_coords[:, 0]) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = r * c

        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        if min_distance <= max_distance_km:
            match = facilities.iloc[min_idx].to_dict()
            match.update({
                'facility_distance_km': min_distance,
                'facility_match': True
            })
        else:
            match = {'facility_match': False, 'facility_distance_km': np.nan}
            for col in facilities.columns:
                match[f'facility_{col}'] = np.nan

        matched_data.append(match)

    matched_df = pd.DataFrame(matched_data, index=samples.index)
    return matched_df    

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
