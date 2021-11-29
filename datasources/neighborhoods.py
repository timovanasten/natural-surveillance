#  Copyright (c) 2021, Timo van Asten
from config import settings
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon
from typing import Union
import logging
log = logging.getLogger(__name__)


class Neighborhoods:
    def __init__(self, source='safety_index'):
        """
        Initializes an object for retrieving neighborhood geometries in the area of Amsterdam.
        :param source: 'cbs' for CBS neighborhood geometries or 'safety-index' for the geometries used in the
        Amsterdam Safety Index
        """
        self.source = source
        if self.source == 'cbs':
            self.neighborhood_gdf: GeoDataFrame = \
                geopandas.read_file(settings.neigborhoods.CBS_SHAPEFILE_PATH).to_crs(epsg=4326)
            self.neighborhood_gdf = self.neighborhood_gdf[self.neighborhood_gdf['GM_NAAM'] == "Amsterdam"]
        if self.source == 'safety_index':
            self.neighborhood_gdf: GeoDataFrame = \
                geopandas.read_file(settings.neigborhoods.AMSTERDAM_SAFETY_INDEX_GEOJSON_PATH).to_crs(epsg=4326)

    def get_geom_by_name(self, name) -> Union[Polygon, MultiPolygon, None]:
        """
        Retrieve the neighborhood borders for a neighborhood in Amsterdam by the name of the neighborhood.
        :param name: Name of the neighborhood.
        :return: Polygon or Multipolygon with the neighborhood geometry.
        """
        try:
            if self.source == 'cbs':
                neighborhood = self.neighborhood_gdf[self.neighborhood_gdf['WK_NAAM'] == name]
                neighborhood_geom = neighborhood.iloc[0].geometry
                return neighborhood_geom
            elif self.source == 'safety_index':
                neighborhood = self.neighborhood_gdf[self.neighborhood_gdf['naam'] == name]
                neighborhood_geom = neighborhood.iloc[0].geometry
                return neighborhood_geom
        except:
            log.info("Could not retrieve neighborhood geometry for %s", name)
            return None
