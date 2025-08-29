import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pycountry
import requests
from bs4 import BeautifulSoup
from openmindat import LocalitiesRetriever
from pathlib import Path
from typing import Union

from workflow_16s.constants import MINDAT_API_KEY, REFERENCES_DIR 

def gpd_from_df(df):
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

class MinDatScraper:
    def __init__(self):
        self._get_session()
        self._get_localities()
      
    def _get_session(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
      
    def _get_soup(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

    def _get_mindattable(self, soup):
        return soup.find('table', class_='mindattable')
      
    def _get_localities(self, url: str = 'https://www.mindat.org/countrylist.php'):
        soup = self._get_soup(url)
        table = self._get_mindattable(soup)
        localities = []
        if table:
            for row in table.find_all('tr')[1:]:  # Skip the header
                cols = row.find_all('td')
                if cols:
                    a_tag = cols[0].find('a')
                    if a_tag:
                        locality_name = a_tag.get_text(strip=True)
                        localities.append(locality_name)
        self.localities = localities


class MinDatAPI:
    def __init__(
      self, 
      api_key: str = MINDAT_API_KEY, 
      output_dir: Union[str, Path] = REFERENCES_DIR,
      plot_package: str = 'mpl'
    ):
        os.environ["MINDAT_API_KEY"] = api_key
        self.output_dir = output_dir
        self.plot_package = plot_package
      
        try:
            self.localities = self._get_mindat_localities()
        except Exception as e:
            print(f"Error getting Mindat localities: {e}")
        
        # Ensure we have localities
        if not self.localities:
            print("No localities found. Using pycountry fallback.")
            self.localities = self._get_pycountry_countries()
      
    def _get_mindat_localities(self):
        scraper = MinDatScraper()
        return scraper.localities

    def _get_pycountry_countries(self):
        return [country.name for country in pycountry.countries]

    def _get_uranium_mines_locality(self, locality: str):
        lr = LocalitiesRetriever()
        lr.country(locality).description("mine").elements_inc("U")
        results = lr.get_dict()
        if 'results' in results and results['results']:
            df = pd.DataFrame(results['results'])
            df['facility'] = [i.split(',')[0] for i in df['txt']]
            df['data_source'] = "MinDat"
            return df, gpd_from_df(df)
        else:
            return pd.DataFrame(), gpd.GeoDataFrame()

    def _mpl_plot_uranium_mines_locality(self, locality: str, gdf: gpd.GeoDataFrame):
        if gdf.empty:
            print(f"No data to plot for {locality}")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(world_url)
        world.plot(ax=ax, color='lightgrey', edgecolor='white')
        gdf.plot(ax=ax, color='red', markersize=5)
        plt.title(f"{locality.capitalize()} Uranium Mines on World Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(alpha=0.3)
        plt.savefig(self.output_dir / f'{locality}_mines_map.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _get_uranium_mines_world(self):
        dfs = []
        for locality in self.localities: 
            print(f"Processing {locality}")
            try:
                df, gdf = self._get_uranium_mines_locality(locality)
                if not df.empty:
                    dfs.append(df)
                    self._mpl_plot_uranium_mines_locality(locality, gdf)
                else:
                    print(f"No uranium mines found in {locality}")
            except Exception as e:
                print(f"Error with {locality}: {e}")
        
        if dfs:
            df = pd.concat(dfs, axis=0)
            df.to_csv(self.output_dir / "mindat_world_mines.tsv", sep="\t", index=False)
            gdf = gpd_from_df(df)
            if not gdf.empty:
                if self.plot_package == 'mpl':
                    self._mpl_plot_uranium_mines_locality('world', gdf)
            return df, gdf
        else:
            print("No data found for any locality")
            return pd.DataFrame(), gpd.GeoDataFrame()


# API
def world_uranium_mines():
    mindat_api = MinDatAPI()
    return mindat_api._get_uranium_mines_world()
