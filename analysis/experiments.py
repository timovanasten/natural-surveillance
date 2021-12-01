#  Copyright (c) 2021, Timo van Asten
import geopandas
from pipeline import PipelineStep, Pipeline, PipelineConfig
from config import settings
from pathlib import Path
import logging
from scipy.stats import pearsonr
import numpy as np
from itertools import product
from geopandas import GeoDataFrame
import plotly.express as px
import plotly.subplots as sp
log = logging.getLogger(__name__)


class Experiment:
    def __init__(self, neighbourhood_names):
        """
        Class for conducting the experiment for correlating the natural surveillance scores with the
        Amsterdam Safety Index.
        :param neighbourhood_names: List of names of neighborhoods in Amsterdam to include in the experiment.
        """
        self.neighbourhood_names = neighbourhood_names
        self.pipeline_names = [name.replace(' ', '-').replace('/', "-").lower() for name in self.neighbourhood_names]
        self.neighbourhood_gdf = None
        self.street_gdf = None
        self.opening_gdf = None

    def collect_data(self, load_method: str = 'step', step_modifier: str = 'up_to_and_including', step: PipelineStep = PipelineStep.CALCULATE_FEATURES,
                     overwrite=False, fov=90, sightline_distance=43, viewing_angle=90) -> None:
        """
        Collect data for the specified neighborhoods.
        :param load_method: Specifies how the existing data is loaded.
        Use 'step' for loading individual step results, which is less prone to changes in the source code, but slower.
        Use 'state' to load the pickled state of the pipeline. Faster but often breaks when changing the source code.
        Use 'none' to not load previous data'
        :param step_modifier: String that specifies to execute the from, up to and including or only the specified step.
        Should be any of 'only', 'from' or 'up_to_and_including' (default)
        :param step: Species start or end point for the data pipeline, depending on the operation setting.
        :param overwrite: Overwrite all previous saved data.
        :param fov: Field of view, only used if a new pipeline needs to be created. See PipelineConfig for more.
        :param sightline_distance: Maximum sightline distance, only used if a new pipeline needs to be created.
        See PipelineConfig for more.
        :param viewing_angle: Maximum viewing angle, only used if a new pipeline needs to be created.
        See PipelineConfig for more.
        :return: None
        """
        for pl_name, neighborhood_name in zip(self.pipeline_names, self.neighbourhood_names):
            try:
                if load_method == 'step':
                    pipeline = Pipeline.load_from_step_results(pl_name)
                elif load_method == 'state':
                    pipeline = Pipeline.load_from_state_file(pl_name)
                elif load_method == 'none':
                    pipeline = None
                else:
                    log.info("Unknown load method %s. Choose any of 'step', 'state' or 'none'",
                             load_method)
                    pipeline = None
                if pipeline is None:
                    log.info("No pipeline data loaded for %s. Creating a new pipeline.", neighborhood_name)
                    config = PipelineConfig(pl_name, fov, sightline_distance, viewing_angle, neighborhood_name)
                    pipeline = Pipeline(config)
                # pipeline.execute_up_to_and_including(up_to_and_including)
                if step_modifier == 'only':
                    pipeline.execute_step(step, overwrite=overwrite)
                elif step_modifier == 'from':
                    pipeline.execute_from(step, overwrite=overwrite)
                elif step_modifier == 'up_to_and_including':
                    pipeline.execute_up_to_and_including(step, overwrite=overwrite)
                else:
                    log.info("Unknown step modifier %s. Choose any of 'only', 'from' or 'up_to_and_including'",
                             step_modifier)
            except Exception as e:
                log.warning("Could not data collection execute for pipeline %s: %s", pl_name, e)

    def merge_data_files(self) -> None:
        """
        Merges the street scores, neighborhood scores and opening location data of the individual areas into 3 files
        which are then saved in the output directory
        :return: None
        """
        # Define all path relative to the pipeline paths where all feature files are located
        street_features_path = Path(settings.features.SUB_DIR, settings.features.STREET_FEATURES_FILENAME)
        neighborhood_features_path = Path(settings.features.SUB_DIR, settings.features.NEIGHBORHOOD_FEATURES_FILENAME)
        opening_location_path = Path(settings.localization.SUB_DIR, settings.localization.CSV_FILENAME)
        file_paths = [street_features_path, neighborhood_features_path, opening_location_path]
        merged_file_names = [filename.replace('.csv', '_merged.csv') for filename in
                             [settings.features.STREET_FEATURES_FILENAME,
                              settings.features.NEIGHBORHOOD_FEATURES_FILENAME,
                              settings.localization.CSV_FILENAME]]
        for feature_path, output_filename in zip(file_paths, merged_file_names):
            file_list = [Path(settings.OUTPUT_DIR, pl_name, feature_path) for pl_name in self.pipeline_names]
            concatenated_df = geopandas.pd.concat([geopandas.pd.read_csv(features_path) for features_path in file_list])
            concatenated_df.to_csv(str(Path(settings.OUTPUT_DIR, output_filename)))

    def load_neighborhood_gdf(self, year) -> GeoDataFrame:
        """
        Loads the GeoDataFrame containing the neighborhood level natural surveillance and safety index scores into the
        neighbourhood_gdf field, given it has been created using the merge_data_files method.
        :param year: Year of the Amsterdam Safety Index to load the GeoDataFrame for.
        :return: The loaded GeoDataframe.
        """
        neighborhood_gdf = geopandas.pd.read_csv(str(Path(settings.OUTPUT_DIR,
                                                          settings.features.NEIGHBORHOOD_FEATURES_FILENAME.replace(
                                                              '.csv', '_merged.csv'))))
        safety_index_gdf = geopandas.pd.read_csv(f'data/safety_index/safety_index{year}.csv')
        neighborhood_gdf = geopandas.pd.merge(neighborhood_gdf, safety_index_gdf, how='left', on='neighborhood_name')
        # Select the relevant fields from the dataframe
        neighborhood_gdf = neighborhood_gdf[['neighborhood_name',
                                             'road_surveillability_1f_reliable',
                                             'road_surveillability_2f_reliable',
                                             'road_surveillability_3f_reliable',
                                             'road_surveillability_1f_dependable',
                                             'road_surveillability_2f_dependable',
                                             'road_surveillability_3f_dependable',
                                             'road_surveillability_no_filter',
                                             'occupant_surveillability_1f_reliable',
                                             'occupant_surveillability_2f_reliable',
                                             'occupant_surveillability_3f_reliable',
                                             'occupant_surveillability_1f_dependable',
                                             'occupant_surveillability_2f_dependable',
                                             'occupant_surveillability_3f_dependable',
                                             'safety_index',
                                             'crime_index',
                                             'hic_subindex',
                                             'hvc_subindex',
                                             'nuisance_index',
                                             'nuisance_by_people_subindex',
                                             'deterioration_subindex',
                                             'perceived_safety_index',
                                             'risk_perception_subindex',
                                             'feelings_of_unsafety_subindex',
                                             'avoidance_subindex']]
        self.neighbourhood_gdf = neighborhood_gdf
        return neighborhood_gdf

    def load_street_gdf(self) -> GeoDataFrame:
        """
        Loads the GeoDataFrame containing the street level natural surveillance and safety index scores into the
        street_gdf field, given it has been created using the merge_data_files method.
        :return: The loaded GeoDataframe.
        """
        street_gdf = geopandas.pd.read_csv(str(Path(settings.OUTPUT_DIR,
                                                    settings.features.STREET_FEATURES_FILENAME.replace(
                                                              '.csv', '_merged.csv'))))
        self.street_gdf = street_gdf
        return street_gdf

    def calculate_correlations(self, year):
        """
        Calculates the Pearson correlations with the Amsterdam Safety Index and saves it to file
        :param year: Year of the safety index to correlate with, either 2019 or 2020
        :return: None
        """
        neighborhood_gdf = self.load_neighborhood_gdf(year)
        rho = neighborhood_gdf.corr(method='pearson')
        p_val = neighborhood_gdf.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = p_val.applymap(lambda x: ''.join(['*' for t in [0.001, 0.01, 0.05] if x <= t]))
        rho = rho.round(2).astype(str) + p
        print(rho.to_string())
        rho.to_csv(f'data/experiments/correlations{year}.csv')

    def scatter_plot(self, year, x, y, trendline="lowess", trendline_options=None) -> None:
        """
        Creates a scatter plot with the neighborhood safety data and natural surveillance scores of interest.
        Run this only after merge_data_files has been executed.
        :param year: Year of the Amsterdam Safety Index to load the data for.
        :param x: column of the neighborhood data to display on the x axis e.g. 'road_surveillability_1f_dependable'
        :param y: column of the neighborhood data to display on the y axis 'perceived_safety_index'
        :param trendline: type of trendline, either 'lowess' or 'ols'. See Plotly Express docs for more details.
        :param trendline_options: See Plotly Express docs for more details.
        :return: None
        """
        if self.neighbourhood_gdf is None:
            self.load_neighborhood_gdf(year)
        fig = px.scatter(self.neighbourhood_gdf, x=x, y=y, trendline=trendline, hover_data=['neighborhood_name'],
                         trendline_options=trendline_options)
        fig.show()

    def scatter_plot_matrix(self, year, x_cols, y_cols, x_names, y_names, title,
                            trendline='lowess', trendline_options=None) -> None:
        """
        Creates a scatter plot matrix for the neighborhood safety data and natural surveillance scores of interest.
        Run this only after merge_data_files has been executed.
        :param year: Year of the Amsterdam Safety Index to load the data for.
        :param x_cols: List of neighborhood data columns to display on the x axis e.g.
        ['road_surveillability_1f_reliable', 'road_surveillability_2f_reliable', 'road_surveillability_3f_reliable',...]
        :param y_cols: List of neighborhood data columns to display on the y axis e.g.
        ['safety_index', 'crime_index', 'hic_subindex', 'hvc_subindex', 'nuisance_index', ...]
        :param x_names: List of names to display for the x columns, e.g. ['Road Surveillability 1F Reliable, ...]
        :param y_names: List of names to display for the x columns, e.g. ['Safety Index, Crime Index, ...]
        :param title: Title to display above the scatter plot matrix.
        :param trendline: Type of trendline. See Plotly Express docs for more details.
        :param trendline_options: See Plotly Express docs for more details.
        :return: None
        """
        if self.neighbourhood_gdf is None:
            self.load_neighborhood_gdf(year)

        # Create scatter plots
        fig = sp.make_subplots(cols=len(x_cols), rows=len(y_cols), shared_xaxes=True, shared_yaxes=True)
        for (sp_col, x_name), (sp_row, y_name) in product(enumerate(x_cols, 1), enumerate(y_cols, 1)):
            scatter = px.scatter(self.neighbourhood_gdf, x=x_name, y=y_name,
                                 trendline=trendline, trendline_options=trendline_options,
                                 hover_data=['neighborhood_name'],
                                 trendline_color_override="red")
            fig.append_trace(scatter["data"][0], col=sp_col, row=sp_row)
            fig.append_trace(scatter["data"][1], col=sp_col, row=sp_row)

        # Add titles on the x axis
        for col, axis_name in enumerate(x_names, 1):
            fig.update_xaxes(title_text=axis_name, col=col, row=len(y_cols))

        # Add titles on the y axis
        for row, axis_name in enumerate(y_names, 1):
            fig.update_yaxes(title_text=axis_name, col=1, row=row)

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5
            ),
            autosize=False,
            width=1240,
            height=1600,
            font=dict(
                family="Utopia",
                size=18)
        )
        fig.show()
