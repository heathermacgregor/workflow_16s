import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class ScrapeWikipedia:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.nps = self._nuclear_power_stations()
        self.um = self._uranium_mines()
        self.data = self._compile_and_sort()
      
    def _get_soup(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

    def _get_wikitables(self, soup):
        return soup.find_all('table', {'class': 'wikitable'})

    def _nuclear_power_stations(self, url: str = "https://en.wikipedia.org/wiki/List_of_nuclear_power_stations"):
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
                    'facility_name': station_name,
                    'data_source': f"Table {i+1} - {url}",
                    'last_updated': datetime.now().isoformat()
                }
                for n in range(1, len(headers)-1):
                    row_data[headers[n]] = cells[n] 
                data.append(row_data)
            dfs.append(pd.DataFrame(data))
        return dfs

    def _uranium_mines(self, url: str = "https://en.wikipedia.org/wiki/List_of_uranium_mines"):
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
                    'facility_name': mine_name,
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
        return dfs

    def _compile_and_sort(self):
        return pd.concat([pd.concat(self.nps), pd.concat(self.um)]).sort_values(by='facility_name')

# Usage
scraper = ScrapeWikipedia()
print(scraper.data)
