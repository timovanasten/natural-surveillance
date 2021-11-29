from __future__ import annotations
from geopandas import GeoDataFrame
from .latlng import LatLng
from datasources.osm import BuildingFootprints
from geojson import FeatureCollection
from util.serialization import Collection
from typing import List, Optional
from config import settings
from shapely.geometry import Polygon


class Building:
    def __init__(self, coordinates: List[LatLng]):
        self.coordinates = coordinates


class BuildingCollection(Collection[Building]):
    def __init__(self, collection_name: str,
                 object_list: Optional[List[Building]] = None,
                 sub_dir=settings.osm.SUB_DIR, json_filename=settings.osm.JSON_FILENAME):
        super().__init__(collection_name, Building, sub_dir, json_filename, object_list)

    def from_building_footprints(self, footprints: BuildingFootprints, export_to_json=True) -> BuildingCollection:
        """
        Create a BuildingCollection from a BuildingFootprints object
        :param footprints: BuildingFootprints object containing a GeoDataFrame
        :param export_to_json: If set to True the collection will be saved to a JSON file
        :return: BuildingCollection object with its object_list set
        """
        if footprints.building_gdf is not None:
            return self.from_dataframe(footprints.building_gdf, export_to_json)

    def from_dataframe(self, building_dataframe: GeoDataFrame, export_to_json=True) -> BuildingCollection:
        """
        Create a BuildingCollection from a GeoDataFrame containing shapely Polygon objects
        such as returned by osmnx.geometries_from_address
        :param building_dataframe:
        :param export_to_json: If set to True the collection will be saved to a JSON file
        :return: BuildingCollection object with its object_list set
        """
        buildings = []
        for row in building_dataframe.itertuples():
            building_coords = []
            # We are not interested in Point features, only Polygons
            if type(row.geometry) is not Polygon:
                continue

            building_exterior: Polygon = row.geometry.exterior
            for coordinate in building_exterior.coords:
                building_coords.append(LatLng(coordinate[1], coordinate[0]))  # OSM coords are (lng, lat)
            buildings.append(Building(building_coords))
        self.object_list = buildings
        if export_to_json:
            self.export_to_json()
        return self


class BuildingManager:
    def __init__(self, building_collection: BuildingCollection):
        self.buildings = building_collection.object_list
        self.create_grid()

    def create_grid(self, interval=50):
        self.building_grid = []
        self.interval = interval

        max_lat = -999999
        max_lng = -999999
        min_lat = 9999999
        min_lng = 9999999
        for b in self.buildings:
            for nd in b.coordinates:
                max_lat = max(max_lat, nd.lat)
                max_lng = max(max_lng, nd.lng)
                min_lat = min(min_lat, nd.lat)
                min_lng = min(min_lng, nd.lng)
        width = abs(LatLng(min_lat, min_lng).get_xy(LatLng(max_lat, max_lng)).x)
        height = abs(LatLng(min_lat, min_lng).get_xy(LatLng(max_lat, max_lng)).y)
        self.min_lat = min_lat
        self.min_lng = min_lng

        self.nx = int(width / interval) + 3
        self.ny = int(height / interval) + 3

        for i in range(self.nx):
            tmp = []
            for j in range(self.ny):
                tmp.append([])
            self.building_grid.append(tmp)

        for b in self.buildings:
            for nd in b.coordinates:
                x = abs(LatLng(min_lat, min_lng).get_xy(nd).x)
                y = abs(LatLng(min_lat, min_lng).get_xy(nd).y)
                self.building_grid[int(x / interval) + 1][int(y / interval) + 1].append(b)

    def find_buildings(self, point):
        x = abs(LatLng(self.min_lat, self.min_lng).get_xy(point).x)
        y = abs(LatLng(self.min_lat, self.min_lng).get_xy(point).y)
        list = []
        for i in range(-2, 4):
            for j in range(-2, 4):
                if 0 <= int(x / self.interval) + i < self.nx:
                    if 0 <= int(y / self.interval) + j < self.ny:
                        for b in self.building_grid[int(x / self.interval) + i][int(y / self.interval) + j]:
                            if b not in list:
                                list.append(b)
        return list

    def find_nearest_building(self, point):
        blist = self.find_buildings(point)
        min_dis = 9999999
        nbuilding = None
        for building in blist:
            for n in building.coordinates:
                dis = point.get_distance(n)
                if dis < min_dis:
                    min_dis = dis
                    nbuilding = building
        return nbuilding
