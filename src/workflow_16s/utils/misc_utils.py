# ===================================== IMPORTS ====================================== #

import requests

DEFAULT_EMAIL = "macgregor@berkeley.edu"

# ==================================== FUNCTIONS ===================================== #

def get_citation(doi_url, style: str = 'apa', email: str = DEFAULT_EMAIL):
    # Extract the DOI from the URL
    doi = doi_url.split('doi.org/')[-1].strip('/')
    
    # Prepare the request URL and headers
    url = f'https://doi.org/{doi}'
    headers = {
        'Accept': f'text/x-bibliography; style={style}',
        'User-Agent': f'AcademicScript/1.0 (mailto:{email})'  
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.text.strip()
    except requests.exceptions.HTTPError as e:
        #return f"Error: {e}"
        return None
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
        return None
