#  Copyright (c) 2021, Timo van Asten
from __future__ import annotations
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
from datasources.osm import BuildingFootprints
from localization.estimator import OpeningLocationEstimateCollection
from util.serialization import PickledDataFrame, get_file_path
import swifter
import numpy as np
from typing import List, Optional
from config import settings
import logging
log = logging.getLogger(__name__)


class Sightlines(PickledDataFrame):
    def __init__(self, config: 'PipelineConfig',
                 building_footprints: BuildingFootprints,
                 opening_locations: OpeningLocationEstimateCollection,
                 local_crs_epsg: int = settings.sightlines.LOCAL_CRS_EPSG):
        """
        Creates Sightlines object.
        :param config: PipelineConfig object to use for analysis
        :param local_crs_epsg: EPSG code for the local coordinate reference system (CRS)
        which will be used for distance and angle calculation. Defaults to EPSG:28992 which covers the Netherlands
        """
        super().__init__(config.name, settings.sightlines.SUB_DIR, settings.sightlines.GDF_FILENAME)
        self.config = config
        self.building_gdf = building_footprints.building_gdf
        # Already filter the higher level windows before starting sightline calculation to save computational time
        self.opening_gdf = opening_locations.to_gdf()
        self.WORLD_CRS_EPSG = 4326  # EPSG 4326: World Geodetic System i.e. latitude, longitude
        self.local_crs_epsg = local_crs_epsg
        self.sightline_gdf: Optional[GeoDataFrame] = None

        # Convert coordinates of geometries to the local coordinate system.
        self.opening_gdf = self._convert_to_crs(self.opening_gdf, self.WORLD_CRS_EPSG, self.local_crs_epsg,
                                                ['road_sightline', 'building_segment', 'opening_location'])
        self.building_gdf = self.building_gdf.to_crs(epsg=self.local_crs_epsg)

    def calculate(self) -> Sightlines:
        """
        Calculates all sightlines between building openings.
        Appends the 'incoming_sightlines' column in the opening_gdf GeoDataFrame,
        containing the number of openings that have a sightline to that opening.
        :return: Reference to this Sightlines object with the sightline_gdf property set.
        """
        # Pre filter all higher level openings to speed up the sightline calculation process
        old_length = len(self.opening_gdf)
        self.opening_gdf = self.opening_gdf[self.opening_gdf.altitude <= settings.localization.DEFAULT_FILTER_ALTITUDE]
        new_length = len(self.opening_gdf)
        log.info(
            "Filtered out %s building openings higher than %sm for speed purposes. Old count: %s. New count: %s",
            old_length - new_length, settings.localization.DEFAULT_FILTER_ALTITUDE, old_length, new_length)
        # Calculate all possible sightlines with a maximum length of max_distance
        self.sightline_gdf = self._calculate_sightlines_in_range(self.opening_gdf, self.config.max_sightline_distance)
        # Remove all sightlines that are blocked by buildings
        self.sightline_gdf = self._drop_blocked_sightlines(self.sightline_gdf, self.building_gdf)
        # Calculate the angles of the sightlines
        self.sightline_gdf = self._calculate_sightline_angles(self.sightline_gdf)
        # Cast DataFrame back to GeoDataFrame
        self.sightline_gdf = GeoDataFrame(self.sightline_gdf, geometry='sightline')
        # Save the GeoDataFrame before applying filters to it
        self.save()
        # Convert back to latitude longitude coordinates
        self.opening_gdf = self._convert_to_crs(self.opening_gdf, self.local_crs_epsg, self.WORLD_CRS_EPSG,
                                                ['building_segment', 'opening_location'])
        return self

    def filter(self, max_sightline_length: Optional[float] = None,
               max_viewing_angle: Optional[float] = None,
               max_altitude_viewpoint: Optional[float] = None,
               max_altitude_observed: Optional[float] = None) -> Sightlines:
        """
        Filters the sightlines on the provided properties. Use reset() to remove previously applied filters.
        :param max_sightline_length: Maximum length of the sightlines in meters.
        :param max_viewing_angle: Maximum viewing angle in degrees.
        :param max_altitude_viewpoint: Maximum altitude of the viewpoint building opening in meters
        :param max_altitude_observed: Maximum altitude of the observed building opening in meters
        :return: Reference to this Sightlines object with the filtered sightlines.
        """
        if max_viewing_angle is not None:
            old_length = len(self.sightline_gdf)
            self.sightline_gdf = self.sightline_gdf[self.sightline_gdf.angle <= max_viewing_angle]
            new_length = len(self.sightline_gdf)
            log.info(
                "Filtered out %s sightlines of the viewing angle of %s degrees. Old count: %s. New count: %s",
                old_length - new_length, max_viewing_angle, old_length, new_length)
        if max_sightline_length is not None:
            old_length = len(self.sightline_gdf)
            self.sightline_gdf = self.sightline_gdf[self.sightline_gdf.sightline_length <= max_sightline_length]
            new_length = len(self.sightline_gdf)
            log.info(
                "Filtered out %s sightlines longer than %sm. Old count: %s. New count: %s",
                old_length - new_length, max_sightline_length, old_length, new_length)
        if max_altitude_viewpoint is not None:
            old_length = len(self.sightline_gdf)
            self.sightline_gdf = self.sightline_gdf[self.sightline_gdf.altitude_viewpoint <= max_altitude_viewpoint]
            new_length = len(self.sightline_gdf)
            log.info(
                "Filtered out %s sightlines with a viewpoint higher than %sm. Old count: %s. New count: %s",
                old_length - new_length, max_altitude_viewpoint, old_length, new_length)
        if max_altitude_observed is not None:
            old_length = len(self.sightline_gdf)
            self.sightline_gdf = self.sightline_gdf[self.sightline_gdf.altitude_observed <= max_altitude_observed]
            new_length = len(self.sightline_gdf)
            log.info(
                "Filtered out %s sightlines with an observed point higher than %sm. Old count: %s. New count: %s",
                old_length - new_length, max_altitude_observed, old_length, new_length)
        return self

    def reset_filters(self):
        return self.load()

    def plot(self) -> None:
        fig, ax = plt.subplots()
        self.building_gdf.plot(ax=ax)
        self.sightline_gdf.plot(ax=ax, color='green', linewidth=0.1)
        plt.show()

    def to_csv(self) -> None:
        """
        Creates an CSV export for visualizing the sightlines in Kepler.gl
        :return: None
        """
        sightlines = self._convert_to_crs(self.sightline_gdf,
                                          self.local_crs_epsg,
                                          self.WORLD_CRS_EPSG, ['sightline'])[['sightline']]
        sightlines['viewpoint_lng'] = sightlines['sightline'].apply(lambda line: line.coords[0][0])
        sightlines['viewpoint_lat'] = sightlines['sightline'].apply(lambda line: line.coords[0][1])
        sightlines['target_lng'] = sightlines['sightline'].apply(lambda line: line.coords[1][0])
        sightlines['target_lat'] = sightlines['sightline'].apply(lambda line: line.coords[1][1])
        sightlines = sightlines.drop(columns='sightline')
        save_path = get_file_path(self.config.name, settings.sightlines.SUB_DIR, settings.sightlines.CSV_FILENAME)
        sightlines.to_csv(str(save_path))

    def save(self) -> None:
        self.save_gdf(self.sightline_gdf)

    def load(self) -> Optional[Sightlines]:
        """
        Loads the sightlines GeoDataFrame from file.
        :return: Reference to this StreetNetwork object if load was successful, None otherwise
        """
        sightline_gdf = self.load_gdf()
        if sightline_gdf is not None:
            self.sightline_gdf = GeoDataFrame(sightline_gdf, geometry='sightline', crs=self.local_crs_epsg)
            return self
        else:
            return None

    @staticmethod
    def _calculate_sightlines_in_range(opening_gdf: GeoDataFrame, max_distance: int) -> GeoDataFrame:
        """
        Calculate all possible sightlines within a specified maximum distance.
        :param opening_gdf: GeoDataFrame containing all opening locations.
        :param max_distance: Maximum distance in meters
        :return: GeoDataFrame with a row for each sightline
        """
        log.info("Computing all possible sightlines within a maximum of %s meters...", max_distance)
        radius_polygons = GeoDataFrame(geometry=opening_gdf.buffer(max_distance))
        sightline_gdf = geopandas.sjoin(opening_gdf, radius_polygons, how='left', op='within')

        sightline_gdf = sightline_gdf.merge(
            opening_gdf[['opening_location', 'altitude']],
            how='left', left_on='index_right',
            right_index=True,
            suffixes=["_viewpoint", "_observed"])
        # Opening locations are now named opening_location_viewpoint and opening_location_observed
        # The altitudes are named altitude_viewpoint and altitude_observed
        # This results in a dataframe containing all information of the viewpoint
        # (location, building segment, altitude, size, origin image, street index etc.)
        # + the location as POINT(latitude,longitude) and altitude of the observed point

        def create_sightline(row):
            viewpoint = row.opening_location_viewpoint
            observed = row.opening_location_observed
            sightline = LineString([Point(viewpoint), Point(observed)])
            # Shorten sightlines slightly to make sure the sightline end point points are just outside the
            # building geometry. This way we can detect blocked sightlines with the .intersect() method
            sightline = LineString([sightline.interpolate(0.01, True), sightline.interpolate(0.99, True)])
            return sightline

        def calculate_sightline_length(row):
            return row.sightline.length

        # Append the sightline and sightline length column to the dataframe
        sightline_gdf["sightline"] = sightline_gdf.swifter.apply(create_sightline, axis=1)
        sightline_gdf["sightline_length"] = sightline_gdf.swifter.apply(calculate_sightline_length, axis=1)
        # Remove sightlines where the observer point is the destination point
        sightline_gdf = sightline_gdf[sightline_gdf.index != sightline_gdf["index_right"]]

        log.info("Total sightlines with a maximum length of %sm: %s", max_distance, len(sightline_gdf))
        return sightline_gdf

    def calculate_road_sightline_length(self):
        """
        Calculate the distance between the gsv camera and the building openings, i.e. the road sightline length
        """
        def calculate_sightline_length(row):
            return row.road_sightline.length

        self.opening_gdf["road_sightline_length"] = self.opening_gdf.swifter.apply(calculate_sightline_length, axis=1)
        # TODO: Convert the road sightline column back to world EPSG

    def append_incoming_sightlines_count(self,
                                         column_name_postfix: str,
                                         max_viewing_angle=None,
                                         max_sightline_length=None,
                                         max_altitude_viewpoint=None,
                                         max_altitude_observed=None):
        """
        For each opening, count by how many openings it is observed, i.e. count the incoming sightlines and add this as
        the incoming_sightlines column to the GeoDataFrame with openings.
        :param column_name_postfix: How to postfix the column to be appended, e.g.' _1f_reliable' will result in a
        column called "incoming_sightlines_1f_reliable"
        :param max_sightline_length: Maximum length of the sightlines in meters.
        :param max_viewing_angle: Maximum viewing angle in degrees.
        :param max_altitude_viewpoint: Maximum altitude of the viewpoint building opening in meters
        :param max_altitude_observed: Maximum altitude of the observed building opening in meters
        :return: None
        """
        # TODO: Check if I can reset the index of sightline gdf to improve performance
        column_name = "incoming_sightlines" + column_name_postfix
        log.info("Computing column for building opening DataFrame: %s", column_name)
        self.filter(max_sightline_length, max_viewing_angle, max_altitude_viewpoint, max_altitude_observed)
        counts = self.sightline_gdf['index_right'].value_counts(sort=False)
        counts.name = column_name
        self.opening_gdf = self.opening_gdf.join(counts)
        self.opening_gdf[counts.name] = self.opening_gdf[counts.name].fillna(0).astype(int)

    @staticmethod
    def _drop_blocked_sightlines(sightlines: GeoDataFrame, buildings: GeoDataFrame) -> GeoDataFrame:
        """
        Drops all sightlines that are blocked by buildings.
        :param sightlines: GeoDataFrame containing the sightlines
        :param buildings: GeoDataFrame containing the buildings.
        :return: GeoDataFrame with all sightlines that are not blocked by buildings
        """
        log.info("Removing sightlines that are blocked by buildings")
        # Generate spacial index for the buildings
        spacial_index = buildings.sindex

        # Define function for to apply to each row
        def sightline_obstructed(row):
            sightline: LineString = row['sightline']
            intersects_building = spacial_index.query_bulk([sightline], 'intersects')[0].size > 0
            return True if intersects_building else False

        # Silence false positive warning on chained assignment using pandas context manager
        with geopandas.pd.option_context('mode.chained_assignment', None):
            sightlines['is_obstructed'] = sightlines.swifter.apply(sightline_obstructed, axis=1)

        sightlines = sightlines[sightlines['is_obstructed'] == False]
        sightlines = sightlines.drop(columns='is_obstructed')
        log.info("Remaining sightlines: %s", len(sightlines))
        return sightlines

    @staticmethod
    def _calculate_sightline_angles(sightlines: GeoDataFrame) -> GeoDataFrame:
        """
        Calculates the angles of all sightlines relative to the building segment that the opening is located in.
        :param sightlines: GeoDataFrame containing the sightlines to calculate the angles for
        :return: The sightlines GeoDataFrame with the 'angle' column appended.
        """
        def line_string_to_vector(line_string):
            vector = [line_string.coords[0][0] - line_string.coords[1][0],
                      line_string.coords[0][1] - line_string.coords[1][1]]
            norm = np.linalg.norm(vector)
            return vector / norm if norm != 0 else vector

        def calculate_viewing_angle(row):
            building_segment: LineString = row['building_segment']
            sightline: LineString = row['sightline']

            building_vector = line_string_to_vector(building_segment)
            sightline_vector = line_string_to_vector(sightline)

            # If any of the vectors is a 0 vector, the angle between them is not defined
            if not np.any(building_vector) or not np.any(sightline_vector):
                return np.nan

            dot_product = np.dot(building_vector, sightline_vector)

            # Fix small numeric errors that make the dot product fall outside of the range of arc cos,
            # e.g. dot_product = 1.0000000000000002 becomes dot_product = 1
            if dot_product > 1:
                dot_product = 1
            if dot_product < -1:
                dot_product = -1

            angle = np.rad2deg(np.arccos(dot_product))
            return angle

        sightlines['angle'] = sightlines.swifter.apply(calculate_viewing_angle, axis=1)
        return sightlines

    @staticmethod
    def _drop_sightlines_out_of_viewing_angle(sightlines: GeoDataFrame, view_angle: int) -> GeoDataFrame:
        """
        Remove sightlines that are outside of the defined viewing angle of the building opening serving as viewpoint.
        :param sightlines: GeoDataFrame containing the computed sightlines.
        :param view_angle: Viewing angle of observing building openings.
        :return: GeoDataframe containing only the sightlines within the defined viewing angle
        """
        log.info("Removing sightlines outside the viewing angle of %s degrees", view_angle)
        min_angle = 90 - view_angle
        max_angle = 90 + view_angle
        sightlines = sightlines[sightlines['angle'].between(min_angle, max_angle)]
        log.info("Remaining sightlines: %s", len(sightlines))
        return sightlines

    @staticmethod
    def _convert_to_crs(gdf: GeoDataFrame, origin_epsg: int, target_epsg: int, geometry_column_names: List[str]) \
            -> GeoDataFrame:
        """
        Converts multiple geometry columns to the desired coordinate reference system (CRS)
        :param gdf: GeoDataFrame to change the CRS for
        :param origin_epsg: EPSG code of the CRS the columns are currently in.
        :param target_epsg: EPSG code to convert to
        :param geometry_column_names: List with names of the geometry columns to convert.
        The last column in the list will be the active geometry column of the returned GeoDataFrame
        :return: GeoDataFrame with converted geometry columns
        """
        for column_name in geometry_column_names:
            gdf = gdf.set_geometry(column_name)
            gdf = gdf.set_crs(epsg=origin_epsg, allow_override=True)
            gdf = gdf.to_crs(epsg=target_epsg)
        return gdf
