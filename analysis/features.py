#  Copyright (c) 2021, Timo van Asten
from pathlib import Path
from geopandas import GeoDataFrame
from config import settings
import logging
from analysis.sightlines import Sightlines
from datasources.neighborhoods import Neighborhoods
log = logging.getLogger(__name__)


class Features:
    """
    Class that calculates the natural surveillance feature values for both occupant and road surveillability at both
    street and neighborhood level. This class produces two GeoDataFrames with results: one for the street level, and one
    for the neighborhood level.

    These two contain the feature values for the included configurations (maximum sightline length and opening height)
    as columns, and the geometry of the streets/neighborhood.

    the feature values can be accessed by neighborhood_features_gdf['feature name']
    or street_segment_features_gdf['feature_name']. Feature names are:

    - 'road_surveillability[_per_meter]_1f_reliable'
    - 'road_surveillability[_per_meter]_2f_reliable'
    - 'road_surveillability[_per_meter]_3f_reliable'
    - 'road_surveillability[_per_meter]_1f_dependable'
    - 'road_surveillability[_per_meter]_2f_dependable'
    - 'road_surveillability[_per_meter]_3f_dependable'
    - 'road_surveillability[_per_meter]_no_filter'

    - 'occupant_surveillability[_per_meter]_1f_reliable'
    - 'occupant_surveillability[_per_meter]_2f_reliable'
    - 'occupant_surveillability[_per_meter]_3f_reliable'
    - 'occupant_surveillability[_per_meter]_1f_dependable'
    - 'occupant_surveillability[_per_meter]_2f_dependable'
    - 'occupant_surveillability[_per_meter]_3f_dependable'

    Where the _per_meter is optional: without returns the absolute feature value and including contains the
    value normalized for street length

    The neighborhood_feature_gdf only contains normalized features, and does not require the inclusion of
    '_per_meter' in the feature name.

    The feature GeoDataFrames are stored as a csv file in the output directory of the pipeline.
    """
    def __init__(self, street_network_gdf: GeoDataFrame, sightlines: Sightlines, config: 'PipelineConfig'):
        # Postfixes for the dataframe column names for the different feature configurations
        self.feature_names = ["_1f_reliable",
                              "_2f_reliable",
                              "_3f_reliable",
                              "_1f_dependable",
                              "_2f_dependable",
                              "_3f_dependable"]
        self.config = config
        # Create folder to save features in
        self.save_dir = features_dir = Path(self.config.pipeline_path, settings.features.SUB_DIR)
        self.save_dir.mkdir(exist_ok=True)
        # Drop unused columns
        self.sightlines = sightlines
        self.street_segment_features_gdf = street_network_gdf.copy()\
            .drop(columns=['street_segments', 'bearings', 'sample_points_with_bearing'])
        # Create GeoDataFrame with the feature values for this neighborhood.
        self.neighborhood_features_gdf = None
        self.segment_count = len(street_network_gdf)
        self._compute_street_level_features()
        self._compute_neighborhood_level_features()
        self.to_csv()

    def _add_occupant_surveillability_columns_opening_level(self) -> None:
        """
        Calculate the absolute feature values for the occupant surveillability features at the opening level. That is,
        add a column to the opening GeoDataFrame which says by how many openings this opening is observed given the
        specific parameters (maximum angle, length and opening height).
        These are added to the opening_gdf field of the Sightlines object
        :return: None
        """
        self.sightlines.append_incoming_sightlines_count(self.feature_names[0], 90, 15, 3, 3)
        self.sightlines.reset_filters()
        self.sightlines.append_incoming_sightlines_count(self.feature_names[1], 90, 15, 6, 6)
        self.sightlines.reset_filters()
        self.sightlines.append_incoming_sightlines_count(self.feature_names[2], 90, 15, 9, 9)
        self.sightlines.reset_filters()
        self.sightlines.append_incoming_sightlines_count(self.feature_names[3], 90, 43, 3, 3)
        self.sightlines.reset_filters()
        self.sightlines.append_incoming_sightlines_count(self.feature_names[4], 90, 43, 6, 6)
        self.sightlines.reset_filters()
        self.sightlines.append_incoming_sightlines_count(self.feature_names[5], 90, 43, 9, 9)

    def _filter_road_sightlines(self, max_sightline_length: float, max_altitude: float) -> GeoDataFrame:
        """
        Filter the road surveillability sightlines on the maximum length and altitude of the openings
        :param max_sightline_length: Maximum length of the sightlines to be included.
        :param max_altitude: Maximum altitude of the openings to be included.
        :return: GeoDataFrame with the filtered openings and their sightlines as shapely LineString.
        """
        return self.sightlines.opening_gdf.query(
            f"altitude<={max_altitude} & road_sightline_length<={max_sightline_length}")

    def _add_road_surveillability_column(self, max_sightline_length: float, max_altitude: float, feature_postfix) \
            -> None:
        """
        Calculates the road surveillability feature at the street segment level, stored in the street_gdf field.
        :param max_sightline_length: Maximum length of the sightlines to be included.
        :param max_altitude: Maximum altitude of the openings to be included.
        :param feature_postfix: Postfix for this configuration (e.g. _1f_reliable).
        :return: None
        """
        feature_name = 'road_surveillability' + feature_postfix
        filtered_df = self._filter_road_sightlines(max_sightline_length, max_altitude)
        grouped_by_street = filtered_df.groupby('street_index')
        # 'size' here is and arbitrary, just selecting a column to count the rows
        road_surveillability = grouped_by_street['size'].count()
        road_surveillability.name = feature_name
        # Normalize feature
        street_length = self.street_segment_features_gdf['length']
        normalized_feature_name = 'road_surveillability_per_meter' + feature_postfix
        self.street_segment_features_gdf = self.street_segment_features_gdf.join([road_surveillability])
        self.street_segment_features_gdf[normalized_feature_name] = \
            self.street_segment_features_gdf[feature_name] / street_length

    def _add_road_surveillability_columns(self) -> None:
        """
        Calculates the different configurations for the road surveillability feature and ads the values as columns in
        the street_gdf attribute.
        :return: None
        """
        # TODO: This is here now to prevent having to recalculate the sightlines
        self.sightlines.calculate_road_sightline_length()
        self._add_road_surveillability_column(15, 3, self.feature_names[0])
        self._add_road_surveillability_column(15, 6, self.feature_names[1])
        self._add_road_surveillability_column(15, 9, self.feature_names[2])
        self._add_road_surveillability_column(43, 3, self.feature_names[3])
        self._add_road_surveillability_column(43, 6, self.feature_names[4])
        self._add_road_surveillability_column(43, 9, self.feature_names[5])
        # Add a feature where all detected openings are included (10000 is meant as large value)
        self._add_road_surveillability_column(10000, 100000, "_no_filter")

    def _add_occupant_surveillability_columns_street_level(self) -> None:
        """
        Add the columns for the occupant surveillability features to the street_segment_features_gdf. That is,
        calculate for each street segment the number of occupant sightlines normalized for street length.
        :return: None
        """
        grouped_df = self.sightlines.opening_gdf.groupby('street_index')
        street_length = self.street_segment_features_gdf['length']
        for feature_name in self.feature_names:
            column_name = 'incoming_sightlines' + feature_name
            # Occupant surveillability: the number of sightlines to other building openings
            occupant_surveillability = grouped_df[column_name].sum()
            occupant_surveillability.name = column_name
            self.street_segment_features_gdf = self.street_segment_features_gdf.join([occupant_surveillability])
            normalized_column_name = 'occupant_surveillability_per_meter' + feature_name
            # Normalize features for street length
            self.street_segment_features_gdf[normalized_column_name] = \
                self.street_segment_features_gdf[column_name] / street_length

    def _compute_street_level_features(self) -> None:
        """
        Compute the occupant and road surveillability features at the street segment level and store them in the
        street_segment_features_gdf field
        :return: None
        """
        log.info("Computing street level features...")
        # Group data on openings by street segment
        self._add_occupant_surveillability_columns_opening_level()
        self._add_occupant_surveillability_columns_street_level()
        self._add_road_surveillability_columns()
        # Use 0 for missing values, which could be due to the fact that imagery is not present for that segment,
        # or no building openings were detected in the available imagery
        self.street_segment_features_gdf.fillna(0, inplace=True)

    def _compute_neighborhood_level_features(self) -> None:
        """
        Compute the occupant and road surveillability features at the neighborhood level and store them in the
        neighborhood_features_gdf field
        :return: None
        """
        log.info("Computing neighborhood level features...")
        street_network_length = self.street_segment_features_gdf['length'].sum()

        # Calculate occupant surveillability
        occupant_feature_values = []
        for feature_name in self.feature_names:
            column_name = 'incoming_sightlines' + feature_name
            occupant_surveillability = self.street_segment_features_gdf[column_name].sum() / street_network_length
            occupant_feature_values.append(occupant_surveillability)

        column_names_occupant_surveillability  = ['occupant_surveillability' + feature_name
                                                  for feature_name in self.feature_names]

        # Road feature also has _no_filter feature
        self.feature_names.append("_no_filter")
        column_names_road_surveillability = ['road_surveillability' + feature_name
                                             for feature_name in self.feature_names]

        # Calculate road surveillability
        road_feature_values = []
        for feature_name in self.feature_names:
            column_name = 'road_surveillability' + feature_name
            road_surveillability = self.street_segment_features_gdf[column_name].sum() / street_network_length
            road_feature_values.append(road_surveillability)

        geometry = Neighborhoods().get_geom_by_name(self.config.neighborhood_name)
        # Create GeoDataFrame with the neighborhood level features. This GeoDataFrame will only have one row:
        # the one for this neighborhood. Later these rows are aggregated into a bigger dataframe with all neighborhoods.
        columns = ["neighborhood_name", "geometry"] \
                  + column_names_road_surveillability + column_names_occupant_surveillability
        neighborhood_features_gdf = GeoDataFrame(columns=columns, geometry='geometry')
        # Add the data for this neighborhood
        neighborhood_features_gdf.loc[0] = [self.config.neighborhood_name, geometry] \
                                            + road_feature_values + occupant_feature_values
        self.neighborhood_features_gdf = neighborhood_features_gdf

    def to_csv(self):
        """
        Save the neighborhood and street level features as CSV file in the output directory of the pipeline.
        :return: None
        """
        save_path_street_segments = Path(self.save_dir, settings.features.STREET_FEATURES_FILENAME)
        save_path_neighborhood = Path(self.save_dir, settings.features.NEIGHBORHOOD_FEATURES_FILENAME)
        log.info("Saved street level features to %s", save_path_street_segments)
        log.info("Saved neighborhood level features to %s", save_path_neighborhood)
        self.street_segment_features_gdf.to_csv(save_path_street_segments, index=False)
        self.neighborhood_features_gdf.to_csv(save_path_neighborhood, index=False)
