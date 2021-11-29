#  Copyright (c) 2021, Timo van Asten
from __future__ import annotations
import osmnx.bearing
import osmnx as nx
from networkx import MultiDiGraph
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, MultiLineString, Polygon, MultiPolygon
from shapely.ops import split
import shapely.wkt
from geopandas import GeoDataFrame
from datasources.neighborhoods import Neighborhoods
import overpass
import math
from typing import List, Optional, Union
from config import settings
import logging
from util.serialization import PickledDataFrame

log = logging.getLogger(__name__)
api = overpass.API()

# Set download timeout and cache settings
nx.config(timeout=settings.osm.DOWNLOAD_TIMEOUT)
nx.config(use_cache=False)


class BuildingFootprints(PickledDataFrame):
    def __init__(self, config: 'PipelineConfig'):
        """
        Creates a BuildingFootprints object, used to download building footprints from OpenStreetMap
        :param config: PipelineConfig object containing the settings used for data retrieval
        """
        super().__init__(config.name, settings.osm.SUB_DIR, settings.osm.BUILDING_GDF_FILENAME)
        self.building_gdf: Optional[GeoDataFrame] = None
        self.config = config

    def download(self):
        """
        Download the building footprints using the settings in the configuration file given at initialization.
        :return: Reference to this BuildingFootprints object with its building_gdf field populated.
        """
        if self.config.area_definition_method == 'address':
            self.download_using_address(self.config.address, self.config.distance_around_address)
        elif self.config.area_definition_method == 'neighborhood':
            self.download_using_polygon(Neighborhoods().get_geom_by_name(self.config.neighborhood_name))
        elif self.config.area_definition_method == 'polygon':
            self.download_using_polygon(shapely.wkt.loads(self.config.polygon_wkt_string))
        self.save()
        return self

    def download_using_address(self, address: str, distance_around_address) -> BuildingFootprints:
        """
        Downloads all building footprints within a specified square bounding box around an address
        :param address: Geocodable address, e.g. 'Mekelweg 4, Delft'.
        :param distance_around_address: Distance in meters around the address to download the data for
        :return: Reference to this BuildingFootprints object with its building_gdf attribute populated
        """
        log.info("Retrieving building footprints from OpenStreetMap...")
        dataframe = osmnx.geometries_from_address(address, {"building": True}, distance_around_address)
        # Only keep building geometry
        self.building_gdf = dataframe[['geometry']]
        return self

    def download_using_polygon(self, polygon: Union[Polygon, MultiPolygon], buffer=settings.osm.BUILDING_BUFFER) \
            -> BuildingFootprints:
        """
        Downloads all building footprints within a specified polygon, for example, neigbourhood boundaries.
        :param polygon: Shapely Polygon or Multipolygon object to download the data for.
        :param buffer: Buffer in meters outside of the specified polygon to also download the data for.
        Set to 0 to only download footprints that are entirely inside the specified polygon.
        :return: Reference to this BuildingFootprints object with its building_gdf attribute populated
        """
        log.info("Retrieving building footprints from OpenStreetMap...")
        METERS_PER_DEGREE = 111300.
        polygon = polygon.buffer(buffer/METERS_PER_DEGREE)
        dataframe = osmnx.geometries_from_polygon(polygon, {"building": True})
        # Only keep building geometry
        self.building_gdf = dataframe[['geometry']]
        return self

    def save(self):
        """
        Saves the building GeoDataFrame.
        :return: None
        """
        self.save_gdf(self.building_gdf)

    def load(self) -> Optional[BuildingFootprints]:
        """
        Loads building footprint GeoDataFrame from file
        :return: Reference to this BuildingFootprints object if load was successful, None otherwise
        """
        self.building_gdf = self.load_gdf()
        if self.building_gdf is not None:
            return self
        else:
            return None

    def plot(self) -> None:
        """
        Show a plot of the retrieved building footprints
        :return: None
        """
        self.building_gdf.plot()
        plt.show()


class StreetNetwork(PickledDataFrame):
    def __init__(self, config: 'PipelineConfig'):
        """
        Creates a StreetNetwork object, used to download the street network of a specified area from OpenStreetMap
        :param config: PipelineConfig object containing the settings used for data retrieval
        """
        super().__init__(config.name, settings.osm.SUB_DIR, settings.osm.STREET_GDF_FILENAME)
        self.street_network_gdf: Optional[GeoDataFrame] = None
        self.config = config

    def download(self):
        """
        Download the street network using the settings in the configuration file given at initialization.
        :return: Reference to this StreetNetwork object with its street_network_gdf field populated.
        """
        if self.config.area_definition_method == 'address':
            self.download_using_address(self.config.address, self.config.distance_around_address)
        elif self.config.area_definition_method == 'neighborhood':
            self.download_using_polygon(Neighborhoods().get_geom_by_name(self.config.neighborhood_name))
        elif self.config.area_definition_method == 'polygon':
            self.download_using_polygon(shapely.wkt.loads(self.config.polygon_wkt_string))
        self.save()
        return self

    def download_using_address(self, address: str, distance_around_address=150, network_type='drive') \
            -> StreetNetwork:
        """
        Retrieves the street network belonging to the area around the provided address and calculates the required data
        to query the Google Street View API.
        :param address: The address to geocode and use as the central point around which to construct the graph
        :param network_type: One of 'drive' (default) ‘walk’, ‘bike’, ‘drive_service’, ‘all’, or ‘all_private’
        :param distance_around_address: Distance around the address in meters for which to retrieve the street network
        :return: StreetNetwork object containing the downloaded data
        """
        log.info("Obtaining the street network %s meters around %s", distance_around_address, address)
        street_network = nx.graph_from_address(address, network_type=network_type, dist=distance_around_address)
        self.street_network_gdf = self._calculate_street_properties(street_network)
        return self

    def download_using_polygon(self, polygon: Union[Polygon, MultiPolygon], network_type='drive') \
            -> StreetNetwork:
        """
        Retrieves the street network belonging to the area around the provided address and calculates the required data
        to query the Google Street View API.
        :param polygon: Polygon with the area to download the street network for
        :param network_type: One of 'drive' (default) ‘walk’, ‘bike’, ‘drive_service’, ‘all’, or ‘all_private’
        :return: Reference to this StreetNetwork object containing the downloaded data
        """
        log.info("Obtaining the street network for polygon")
        street_network = nx.graph_from_polygon(polygon, network_type=network_type)
        self.street_network_gdf = self._calculate_street_properties(street_network)
        return self

    def _calculate_street_properties(self, street_network: MultiDiGraph) -> GeoDataFrame:
        """Calculates additional properties of the street network and returns a GeoDataFrame of the street network
        with columns containing those properties.
        These properties will later be used to generate Google Street View API queries"""
        # Make the street network graph undirected, since we don't care about traffic flow
        street_network = nx.get_undirected(street_network)
        # Add bearing data to the street network
        street_graph_with_bearings = nx.add_edge_bearings(street_network)
        # Convert graph to GeoDataFrame
        network_gdf = nx.graph_to_gdfs(street_graph_with_bearings, nodes=False, edges=True)
        network_gdf['sample_points'] = network_gdf.apply(self._calc_sample_points, axis=1)
        network_gdf['street_segments'] = network_gdf.apply(self._split_street_segments, axis=1)
        network_gdf['bearings'] = network_gdf.apply(self._calculate_segment_bearings, axis=1)
        network_gdf['sample_points_with_bearing'] = network_gdf.apply(self._get_sample_point_bearings, axis=1)
        # Return GeoDataFrame with all relevant columns
        return network_gdf[['name',
                            'geometry',
                            'street_segments',
                            'bearings',
                            'length',
                            'sample_points',
                            'sample_points_with_bearing']]

    def plot_street_network(self) -> None:
        """
        Show a plot of the retrieved street network.
        :return: None
        """
        self.street_network_gdf.plot()
        plt.show()

    def plot_street_orientation(self) -> None:
        """
        Show a plot visualizing the orientation of the street network.
        :return: None
        """
        # Explode the street network dataframe such that it has an entry for each street segment instead of each street
        df = self._create_segment_gdf()
        # Plot the exploded dataframe
        df.plot(column='bearings', cmap='cool', linewidth=1)
        plt.show()

    def plot_sample_points(self) -> None:
        """
        Show a plot with the sampled points along the street network.
        :return: None
        """
        self.street_network_gdf.set_geometry('sample_points').plot(markersize=0.1)
        plt.show()

    def save(self) -> None:
        self.save_gdf(self.street_network_gdf)

    def load(self) -> Optional[StreetNetwork]:
        """
        Loads street network GeoDataFrame from file
        :return: Reference to this StreetNetwork object if load was successful, None otherwise
        """
        self.street_network_gdf = self.load_gdf()
        if self.street_network_gdf is not None:
            return self
        else:
            return None

    @staticmethod
    def _normalise_bearing(row):
        if row.bearing > 180:
            return row.bearing - 180
        else:
            return row.bearing

    @staticmethod
    def _calc_sample_points(row, sample_interval_m=10):
        """
        Calculates sample points for the Google Street View API from the line geometry of the street network.
        :param row: Row of the street network GeoDataFrame
        :param sample_interval_m: distance between sample points in meters
        :return: MultiPoint object containing the sample points
        """
        line_string = row.geometry
        length = row['length']
        nr_of_samples = math.ceil(length / sample_interval_m)
        fraction_step = 1. / nr_of_samples
        points = []
        for i in range(nr_of_samples):
            fraction = i * fraction_step
            points.append(line_string.interpolate(fraction, normalized=True))
        return MultiPoint(points=list(points))

    @staticmethod
    def _get_sample_point_bearings(row):
        """
        Link each sample point to the correct bearing of the street segment.
        :param row: Row of the street network GeoDataFrame
        :return: List with (
        """
        street_segments = row.street_segments
        sample_points = row.sample_points
        segment_bearing_pairs = []
        for point in sample_points:
            for i, street_segment in enumerate(street_segments):
                # Check on which street segment this sample point lies
                if street_segment.distance(point) < 1e-8:  # basically == 0 but calculation has floating point error
                    segment_bearing_pairs.append((point, row.bearings[i]))
        return segment_bearing_pairs

    @staticmethod
    def _split_street_segments(row) -> MultiLineString:
        """
        Splits up the LineString representing a street into multiple two-point LineStrings.
        :param row: Row of the street network GeoDataFrame
        :return: MultiLineString containing all the two-point LineStrings for the street segment
        """
        line_string = row.geometry
        # Check if this LineString needs to be split up
        if len(line_string.coords) <= 2:
            return MultiLineString([line_string])
        else:
            points_to_split = MultiPoint([Point(x, y) for x, y in line_string.coords[1:-1]])
            street_segments = split(line_string, points_to_split)
            # Simplify LineStrings that have duplicate coordinates due to floating point errors
            # in the splitting function
            max_tolerance = 1e-8
            street_segments = street_segments.simplify(max_tolerance)
            return MultiLineString(lines=list(street_segments))

    @staticmethod
    def _calculate_segment_bearings(row, precision=1) -> List[int]:
        """
        Calculates the bearing of each street segment relative to compass north.
        :param row: Row of the street network GeoDataFrame
        :param precision: Precision of the returned values in the number of decimals, e.g. 1 for 12.1, 2 for 12.17 etc.
        :return: Bearing of the street segment
        """
        street_segments = row.street_segments
        bearings = []
        for segment in street_segments:
            coords = segment.coords
            # Assert that the street segments are all LineStrings consisting of two points
            assert len(coords) == 2, "Segment has more than 2 points, making this bearing invalid"
            bearing = nx.bearing.calculate_bearing(coords[0][1], coords[0][0], coords[1][1], coords[1][0])
            # Normalize the bearing
            if bearing > 180:
                bearing = bearing - 180
            bearings.append(round(bearing, precision))
        return bearings

    def _create_segment_gdf(self) -> GeoDataFrame:
        """
        Creates a GeoDataFrame with an entry for each street segment, where a segment is part of a street between
        direction changes. Used for plotting.
        :return: GeoDataFrame with the streets exploded into street segments with corresponding bearings.
        """
        bearings = self.street_network_gdf['bearings'].explode()
        df = self.street_network_gdf.explode('street_segments')[['street_segments']]
        df['bearings'] = bearings
        df = GeoDataFrame(df, geometry='street_segments', crs=4326)
        return df
