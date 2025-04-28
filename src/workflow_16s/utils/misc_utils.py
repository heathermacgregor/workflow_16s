# ===================================== IMPORTS ====================================== #

import requests

# ==================================== FUNCTIONS ===================================== #

def get_citation(doi_url, style='apa'):
    # Extract the DOI from the URL
    doi = doi_url.split('doi.org/')[-1].strip('/')
    
    # Prepare the request URL and headers
    url = f'https://doi.org/{doi}'
    headers = {
        'Accept': f'text/x-bibliography; style={style}',
        'User-Agent': 'AcademicScript/1.0 (mailto:your@email.com)'  # Replace with your contact info
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
