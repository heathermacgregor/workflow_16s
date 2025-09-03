# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Third-Party Imports
import pandas as pd
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s.constants import REFERENCES_DIR, USER_AGENT 

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ====================================== CLASSES ===================================== #

class WikipediaScraper:
    """A web scraper for extracting nuclear facility information from Wikipedia tables.
    
    - Scrapes nuclear power stations and uranium mines data from Wikipedia
    - Combinines scraped data as a pandas DataFrame
    - Saves DataFrame to a TSV file
    
    Attributes:
        BaseURL:    Base URL for Wikipedia pages.
        output_dir: Directory where output files will be saved.
        session:    HTTP session for making requests.
        data:       Combined dataset of scraped facilities.
    """
    BaseURL = "https://en.wikipedia.org/wiki/"
    def __init__(self, output_dir: Union[str, Path] = REFERENCES_DIR):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.data = pd.DataFrame()
      
    def _get_soup(self, url):
        """Retrieve and parse HTML content from a URL."""
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

    def _get_wikitables(self, soup):
        """Extract all wikitable elements from parsed HTML."""
        return soup.find_all('table', {'class': 'wikitable'})

    def _nuclear_power_stations(self, url: str = None):
        """Scrape nuclear power station data from Wikipedia."""
        if url is None:
            url = f"{self.BaseURL}List_of_nuclear_power_stations"
        soup = self._get_soup(url)
        tables = self._get_wikitables(soup)
    
        dfs = []
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
    
            data = []
            for row in rows[1:]:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                station_name = cells[0] if cells[0] else "Unknown Station"
                row_data = {
                    'facility': station_name,
                    'data_source': f"Table {i+1} - {url}",
                    'last_updated': datetime.now().isoformat()
                }
                for n in range(1, len(headers)-1):
                    row_data[headers[n]] = cells[n] 
                data.append(row_data)
            dfs.append(pd.DataFrame(data))
        df = pd.concat(dfs)
        rename = {
            "# units[note 1]": "n_units_1",
            "Capacity(MWe)[note 2]": "capacity_mwe",
            "Country or territory": "country_1",
            "Location": "lat_lon",
            "Began operation": "facility_start_year",
            "No. ofunits": "n_units_2",
            "Net capacityunder construction(MW)": "capacity_mw_during_construction",
            "Constructionstart": "facility_start_construction_year",
            "Plannedconnection": "facility_planned_connection_year",
            "Country": "country_2",
            "Past capacity (MW)": "capacity_mw_past",
        }
        df = df.rename(columns=rename)
        df['country'] = df['country_1'].combine_first(df['country_2'])
        df['n_units'] = df['n_units_1'].combine_first(df['n_units_2'])
        return df

    def _uranium_mines(self, url: str = None):
        """Scrape uranium mine data from Wikipedia."""
        if url is None:
            url = f"{self.BaseURL}List_of_uranium_mines"
        soup = self._get_soup(url)
        tables = self._get_wikitables(soup)

        dfs = []
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            headers = [th.get_text().strip() for th in rows[1].find_all(['th', 'td'])]
            data = []
            for row in rows[2:]: # Skip header
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if len(cells) >=2:
                    mine_name = cells[0] if cells[0] else "Unknown Mine"
                    if mine_name == 'Mine':
                        continue
                    location = cells[1] if len(cells) > 1 else "Unknown"
                        
                    # Try to extract country from location
                    country = "Unknown"
                    if len(cells) > 2 and cells[2]:
                        country = cells[2]
                    elif location and ',' in location:
                        country = location.split(',')[-1].strip()

                row_data = {
                    'facility': mine_name,
                    'facility_location': location,
                    'country': country,
                    'data_source': f"Table {i+1} - {url}",
                    'last_updated': datetime.now().isoformat()
                }
                if len(cells) >= 3:
                    for n in range(3, len(cells)-1):
                        row_data[headers[n]] = cells[n] 
                data.append(row_data)
            dfs.append(pd.DataFrame(data))
        df = pd.concat(dfs)
        rename = {
            "Year discovered": "facility_discovered_year",	
            "Year commenced": "facility_start_year",	
            "Grade %U[2]": "grade_%_u",	
            "Annual production (tOre)[3]": "annual_production_tons_ore",	
            "Annual production (tU)[4]": "annual_production_tons_u",	
            "Scheduled Commencement	Grade %U": "scheduled_commencement_grade_%_u",	
            "Planned Annual production (tOre)": "planned_annual_production_tons_ore",	
            "Planned Annual production (tU)": "planned_annual_production_tons_u",	
            "Probable commencement": "probable_facility_start_year",	
            "Probable annual production (tOre)": "probable_annual_production_tons_ore",	
            "Probable annual production (tU)": "probable_annual_production_tons_u",	
            "Total production (tU)": "total_production_tons_u",	
            "Year closed": "facility_end_year",
        }
        df = df.rename(columns=rename)
        return df

    def _compile_and_sort(self):
        """Combine and sort all scraped facility data."""
        nps = self._nuclear_power_stations()
        um = self._uranium_mines()
        datasets = [nps, um]
        df = pd.concat(datasets)
        df = df.sort_values(by='facility_name')
        df.to_csv(self.output_dir / "wikipedia.tsv", sep="\t", index=False)
        self.data = df
        return df

# ======================================= API ======================================== #

def world_nfc_facilities(output_dir: Union[str, Path] = REFERENCES_DIR):
    """Public API function to retrieve worldwide nuclear facility data from Wikipedia.
    
    Returns:
        DataFrame containing combined nuclear facility information from Wikipedia.
    """
    scraper = WikipediaScraper(output_dir)
    return scraper._compile_and_sort()
