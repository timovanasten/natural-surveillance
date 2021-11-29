#  Copyright (c) 2021, Timo van Asten
from __future__ import annotations

import os
import time
from datetime import timedelta
from enum import Enum, auto
from typing import Optional, Union
from labeling.labelers import FacadeLabeler
from localization.building_manager import BuildingManager, BuildingCollection
from datasources.gsv import GSVImageMetaCollection, GSVQueryCollection
from labeling.openings import FacadeLabelingResultCollection
from datasources.osm import StreetNetwork, BuildingFootprints
from localization.estimator import LocationEstimator, OpeningLocationEstimateCollection
import pickle
import jsons
import json
from config import settings
from pathlib import Path
from analysis.features import Features
from analysis.sightlines import Sightlines
import logging
log = logging.getLogger(__name__)


class PipelineStep(Enum):
    RETRIEVE_STREET_NETWORK = auto()
    GENERATE_GSV_QUERIES = auto()
    RETRIEVE_IMAGE_METADATA = auto()
    RETRIEVE_IMAGES = auto()
    LABEL_FACADES = auto()
    RETRIEVE_BUILDING_FOOTPRINTS = auto()
    LOCALIZE_OPENINGS = auto()
    CALCULATE_SIGHTLINES = auto()
    CALCULATE_FEATURES = auto()


class PipelineConfig:
    def __init__(self, name: str, fov: int, max_sightline_distance: int, viewing_angle: int,
                 neighborhood_name: Optional[str] = None,
                 address: Optional[str] = None, distance_around_address: Optional[int] = None,
                 polygon_wkt_string: Optional[str] = None):
        """
        Configuration of the pipeline. You must either provide a neighborhood name or an address and distance.
        :param name: Name of the folder to store the pipeline data
        :param address: The address to geocode and use as the central point around which to execute the pipeline
        :param distance_around_address: Distance around the address in meters for which to execute the pipeline
        :param neighborhood_name: Name of a neighborhood to run this pipeline for
        :param polygon_wkt_string: WKT string for a polygon or multi polygon defining the area to run this pipeline on
        :param fov: Field of view for the Street View images. See GSVRequest for more details.
        :param max_sightline_distance: Maximum distance of sightlines to be included in the analysis
        :param viewing_angle: Width of the view from the openings in degrees to be used to calculate sightlines
        """
        self.name = name
        # Set paths to save files to based on general settings and pipeline name
        self._set_paths()
        self.fov = fov
        self.max_sightline_distance = max_sightline_distance
        self.viewing_angle = viewing_angle
        self.address = address
        self.distance_around_address = distance_around_address
        self.neighborhood_name = neighborhood_name
        self.polygon_wkt_string = polygon_wkt_string
        self.area_definition_method, is_valid = \
            self._validate_config(self.address,
                                  self.distance_around_address,
                                  self.neighborhood_name,
                                  self.polygon_wkt_string)

        if not is_valid:
            log.error("Configuration is invalid. Provide either an address and distance, "
                      "a neighborhood name or a polygon")
            exit(-1)

    @staticmethod
    def _validate_config(address, distance_around_address, neighborhood_name, polygon_wtk_string) -> tuple[str, bool]:
        if address is not None:
            is_valid = distance_around_address is not None \
                       and polygon_wtk_string is None \
                       and neighborhood_name is None
            return 'address', is_valid
        if neighborhood_name is not None:
            is_valid = address is None \
                       and distance_around_address is None \
                       and polygon_wtk_string is None
            return 'neighborhood', is_valid
        if polygon_wtk_string is not None:
            is_valid = address is None \
                       and distance_around_address is None \
                       and neighborhood_name is None
            return 'polygon', is_valid

    def _set_paths(self):
        self.pipeline_path: Path = Path(settings.OUTPUT_DIR, self.name)
        self.config_path: Path = Path(self.pipeline_path, settings.pipeline.CONFIG_FILENAME)

    def rename(self, name):
        self.name = name
        self._set_paths()

    def export_to_json(self):
        """
        Export the configuration to a JSON file.
        :return: None
        """
        json_path = Path(self.pipeline_path, settings.pipeline.CONFIG_FILENAME)
        with open(json_path, 'w') as outfile:
            json_dict = jsons.dump(self)
            json.dump(json_dict, outfile, indent=settings.JSON_INDENT)
        log.info("Saved pipeline configuration to %s.", json_path)

    @classmethod
    def load_from_json(cls, name) -> Optional[PipelineConfig]:
        """
        Load a previously created config from a JSON file.
        :return: Collection object with loaded data if data loaded successfully, None otherwise
        """
        json_path = Path(settings.OUTPUT_DIR, name, settings.pipeline.CONFIG_FILENAME)
        if not json_path.exists():
            log.info("Could not load configuration for '%s'. File does not exist: %s", name, str(json_path))
            return None
        try:
            with open(json_path) as json_file:
                json_dict = json.load(json_file)
            config = jsons.load(json_dict, cls)
            config._set_paths()  # jsons.load initializes paths as strings. This turns them into Path objects again
            log.info("Loaded pipeline configuration for '%s'", name)
            return config
        except jsons.exceptions.DeserializationError as e:
            log.info("Error loading pipeline config for '%s'. File is most likely outdated: %s", name, e)
            return None


class Pipeline:
    def __init__(self, config: PipelineConfig):
        """
        Creates a Pipeline object.
        :param : Name that will be used to save and retrieve all execution steps of the pipeline, e.g. 'barcelona'
        """
        self.config = config
        self.building_manager: Optional[BuildingManager] = None
        self.building_footprints: Optional[BuildingFootprints] = None
        self.street_network: Optional[StreetNetwork] = None
        self.gsv_queries: Optional[GSVQueryCollection] = None
        self.gsv_metadata: Optional[GSVImageMetaCollection] = None
        self.images_are_retrieved: bool = False
        self.labeling_result: Optional[FacadeLabelingResultCollection] = None
        self.location_estimates: Optional[OpeningLocationEstimateCollection] = None
        self.sightlines: Optional[Sightlines] = None
        self.features: Optional[Features] = None

        # Create directory
        self.config.pipeline_path.mkdir(parents=True, exist_ok=True)
        # Save configuration to a JSON file
        self.config.export_to_json()

    def execute_step(self, pipeline_step: PipelineStep, overwrite=False) -> Pipeline:
        """
        Executes the provided step of the pipeline.
        :param pipeline_step: Step to execute.
        :param overwrite: If set to True, previously collected data for this step will be overwritten.
        :return: Reference to this Pipeline object
        """
        # Retrieve street network
        start_time = time.time()
        if pipeline_step == PipelineStep.RETRIEVE_STREET_NETWORK:
            if self.street_network is None or overwrite:
                self.retrieve_street_network()
                self.save()
            else:
                log.info("Street network already retrieved, skipping this step")

        # Generate Google Street View queries
        if pipeline_step == PipelineStep.GENERATE_GSV_QUERIES:
            if self.gsv_queries is None or overwrite:
                self.generate_gsv_queries()
                self.save()
            else:
                log.info("Queries already generated, skipping this step")

        # Obtain image metadata from Google Street View API
        if pipeline_step == PipelineStep.RETRIEVE_IMAGE_METADATA:
            if self.gsv_metadata is None or overwrite:
                self.retrieve_metadata()
                self.save()
            else:
                log.info("Metadata already retrieved, skipping this step")

        # Retrieve Google Street View images
        if pipeline_step == PipelineStep.RETRIEVE_IMAGES:
            if self.images_are_retrieved is False or overwrite:
                self.retrieve_images()
                self.save()
            else:
                log.info("Images already retrieved, skipping this step")

        # Label facades
        if pipeline_step == PipelineStep.LABEL_FACADES:
            if self.labeling_result is None or overwrite:
                self.label_facades()
                self.save()
            else:
                log.info("Facades already labeled, skipping this step")

        # Retrieve building footprints
        if pipeline_step == PipelineStep.RETRIEVE_BUILDING_FOOTPRINTS:
            if self.building_manager is None or self.building_footprints is None or overwrite:
                self.retrieve_building_footprints()
                self.save()
            else:
                log.info("Buildings already retrieved, skipping this step")

        # Localize openings
        if pipeline_step == PipelineStep.LOCALIZE_OPENINGS:
            if self.location_estimates is None or overwrite:
                self.localize_openings()
                self.save()
            else:
                log.info("Openings already localized, skipping this step")

        # Calculate sightlines
        if pipeline_step == PipelineStep.CALCULATE_SIGHTLINES:
            if self.sightlines is None or overwrite:
                self.calculate_sightlines()
                self.save()
            else:
                log.info("Sightlines already calculated, skipping this step")

        # Calculate features
        if pipeline_step == PipelineStep.CALCULATE_FEATURES:
            if self.features is None or overwrite:
                self.calculate_features()
                self.save()
            else:
                log.info("Features already calculated, skipping this step")

        end_time = time.time()
        step_time_seconds = round(end_time - start_time)
        if step_time_seconds > 0:
            log.info("Step time: %s", str(timedelta(seconds=step_time_seconds)))
        return self

    def execute_all(self, overwrite=False) -> Pipeline:
        """
        Executes all steps of the pipeline
        :param overwrite: If set to True, previously collected data will be overwritten.
        :return: Reference to this Pipeline object
        """
        for step in PipelineStep:
            self.execute_step(step, overwrite)
        return self

    def execute_up_to_and_including(self, pipeline_step: PipelineStep, overwrite=False) -> Pipeline:
        """
        Execute all pipeline steps from the beginning up to and including the provided step.
        :param pipeline_step: Step to execute up to
        :param overwrite: If set to True, previously collected data will be overwritten.
        :return: Reference to this Pipeline object
        """
        for step in PipelineStep:
            if step.value <= pipeline_step.value:
                self.execute_step(step, overwrite)
        return self

    def execute_from(self, pipeline_step: PipelineStep, overwrite=False) -> Pipeline:
        """
        Executes the provided pipeline step and all following steps, up to the end.
        :param pipeline_step: Step to execute up to
        :param overwrite: If set to True, previously collected data will be overwritten.
        :return: Reference to this Pipeline object
        """
        for step in PipelineStep:
            if step.value >= pipeline_step.value:
                self.execute_step(step, overwrite)
        return self

    def retrieve_building_footprints(self, plot=settings.osm.SHOW_PLOTS) -> Pipeline:
        """
        Execute the retrieval of building footprints.
        :param plot: If set to True, a plot will be displayed after data retrieval
        :return: Reference to this Pipeline object
        """
        footprints = BuildingFootprints(self.config).download()
        if plot:
            footprints.plot()
        collection = BuildingCollection(self.config.name).from_building_footprints(footprints)
        self.building_footprints = footprints
        self.building_manager = BuildingManager(collection)
        return self

    def retrieve_street_network(self, plot=settings.osm.SHOW_PLOTS) -> Pipeline:
        """
        Execute the retrieval of the street network.
        :param plot: If set to True, a plot will be displayed after data retrieval
        :return: Reference to this Pipeline object
        """
        network = StreetNetwork(self.config).download()
        if plot:
            network.plot_street_network()
        self.street_network = network
        return self

    def generate_gsv_queries(self) -> Pipeline:
        """
        Generate queries for the Google Street View API by sampling the street network.
        :return: Reference to this Pipeline object
        """
        if self.street_network is not None:
            self.gsv_queries = GSVQueryCollection(self.config.name)\
                .create_from_street_network(self.street_network, self.config.fov)
        else:
            log.error("Step failed. Retrieve street network first")
        return self

    def retrieve_metadata(self) -> Pipeline:
        """
        Retrieve the metadata for Google Street View images (free of charge) for the generated queries.
        :return: Reference to this Pipeline object
        """
        if self.gsv_queries is not None:
            self.gsv_metadata = GSVImageMetaCollection(self.config.name).create_from_api(self.gsv_queries)
        else:
            log.error("Step failed. Generate Street View queries first")
        return self

    def retrieve_images(self) -> Pipeline:
        """
        Retrieve the images from Google Street View API ($7 per 1000 images as of 2021)
        belonging to the retrieved metadata.
        :return: Reference to this Pipeline object
        """
        if self.gsv_metadata is not None:
            self.gsv_metadata.request_images()
            self.images_are_retrieved = True
        else:
            log.error("Step failed. Retrieve Street View metadata first")
        return self

    def label_facades(self) -> Pipeline:
        """
        Detect building openings within the retrieved street level imagery.
        :return: Reference to this Pipeline object
        """
        if self.gsv_metadata is not None:
            self.labeling_result = FacadeLabeler().label(self.gsv_metadata)
        else:
            log.error("Step failed. Retrieve Street View images first")
        return self

    def localize_openings(self) -> Pipeline:
        """
        Calculate 3D geolocations for the detected building openings.
        :return: Reference to this Pipeline object
        """
        if self.labeling_result is None:
            log.error("Cannot localize openings. Label images first")
        if self.building_manager is None:
            log.error("Cannot localize openings. Missing BuildingManager")
        else:
            self.location_estimates = LocationEstimator().estimate_collection(self.building_manager,
                                                                              self.labeling_result)
        return self

    def calculate_sightlines(self):
        """
        Calculate sightlines between building openings.
        :return: Reference to this Pipeline object
        """
        if self.location_estimates is None:
            log.error("Cannot calculate visibility graph. Missing opening location estimates")
        if self.building_footprints is None:
            log.error("Cannot calculate visibility graph. Missing building footprints")
        else:
            self.sightlines = Sightlines(self.config, self.building_footprints, self.location_estimates).calculate()

    def calculate_features(self) -> Pipeline:
        """
        Calculate feature values for road and occupant surveillability at both the street and neighborhood level.
        :return: Reference to this Pipeline object
        """
        if self.street_network is None:
            log.error("Cannot calculate features. Missing street network")
        if self.sightlines is None:
            log.error("Cannot calculate features. Missing sightlines")
        else:
            self.features = Features(self.street_network.street_network_gdf,
                                     self.sightlines,
                                     self.config)
        return self

    def save(self) -> None:
        """
        Saves the whole pipeline state as pickle file in its directory.
        :return: None
        """
        log.info("Saving pipeline state...")
        save_path = Path(self.config.pipeline_path, settings.pipeline.STATE_FILENAME)
        with open(save_path, 'wb') as out_file:
            pickle.dump(self, out_file)
        log.info("Saved pipeline state to %s", str(save_path))

    def print_state(self) -> None:
        """
        Prints an overview of the completed steps in the pipeline.
        :return: None
        """
        log.info("Pipeline state:")
        log.info("  %s Street network", "✅" if self.street_network is not None else "❌")
        log.info("  %s GSV queries", "✅" if self.gsv_queries is not None else "❌")
        log.info("  %s GSV images", "✅" if self.gsv_metadata is not None else "❌")
        log.info("  %s Labeling result", "✅" if self.labeling_result is not None else "❌")
        log.info("  %s Building footprints", "✅" if self.building_footprints is not None else "❌")
        log.info("  %s Opening locations", "✅" if self.location_estimates is not None else "❌")
        log.info("  %s Sightlines", "✅" if self.sightlines is not None else "❌")
        log.info("  %s Features", "✅" if self.features is not None else "❌")

    def rename(self, new_name):
        """
        Change the name of the pipeline. Deals with the changes in folder structure automatically.
        :param new_name: The new name of the pipeline
        :return: None
        """
        old_path = self.config.pipeline_path
        self.config.rename(new_name)
        os.rename(old_path, self.config.pipeline_path)
        self.config.export_to_json()
        self.save()

    @classmethod
    def load_from_state_file(cls, name) -> Optional[Pipeline]:
        """
        Load the pipeline state from a pickle file.
        :param name: Name of the pipeline
        :return: Pipeline object if load was successful, None otherwise
        """
        state_path = Path(settings.OUTPUT_DIR, name, settings.pipeline.STATE_FILENAME)
        if not state_path.exists():
            log.info("Could not load state for this pipeline. File does not exist: %s", str(state_path))
            return None
        with open(state_path, 'rb') as state_file:
            pipeline: Pipeline = pickle.load(state_file)
        pipeline.print_state()
        return pipeline

    @classmethod
    def load_from_step_results(cls, name, up_to_and_including: PipelineStep = PipelineStep.CALCULATE_FEATURES) \
            -> Optional[Pipeline]:
        """
        Load the pipeline state from the JSON files produced in each step
        :param name: Name of the pipeline
        :param up_to_and_including: PipelineStep up to which the pipeline should be loaded
        :return: Pipeline object if load was successful, None otherwise
        """
        log.info("Loading saved pipeline data...")
        pl_config = PipelineConfig.load_from_json(name)
        # If no config is found, return None
        if pl_config is None:
            return None
        # Else try to load the step files and return a pipeline object
        pipeline = Pipeline(pl_config)
        if up_to_and_including.value >= PipelineStep.RETRIEVE_STREET_NETWORK.value:
            pipeline.street_network = StreetNetwork(pipeline.config).load()
        if up_to_and_including.value >= PipelineStep.GENERATE_GSV_QUERIES.value:
            pipeline.gsv_queries = GSVQueryCollection(name).load_from_json()
        if up_to_and_including.value >= PipelineStep.RETRIEVE_IMAGE_METADATA.value:
            pipeline.gsv_metadata = GSVImageMetaCollection(name).load_from_json()
        if up_to_and_including.value >= PipelineStep.LABEL_FACADES.value:
            pipeline.labeling_result = FacadeLabelingResultCollection(name).load_from_json()
        if up_to_and_including.value >= PipelineStep.RETRIEVE_BUILDING_FOOTPRINTS.value:
            pipeline.building_footprints = BuildingFootprints(pipeline.config).load()
            building_collection = BuildingCollection(name).load_from_json()
            if building_collection is not None:
                pipeline.building_manager = BuildingManager(building_collection)
        if up_to_and_including.value >= PipelineStep.LOCALIZE_OPENINGS.value:
            pipeline.location_estimates = OpeningLocationEstimateCollection(name).load_from_json()
        if up_to_and_including.value >= PipelineStep.CALCULATE_SIGHTLINES.value:
            if pipeline.building_footprints is not None and pipeline.location_estimates is not None:
                pipeline.sightlines = Sightlines(pipeline.config,
                                                 pipeline.building_footprints,
                                                 pipeline.location_estimates).load()
        else:
            log.warning("Did not load sightlines: missing building footprints or opening location estimates")

        pipeline.print_state()
        # Save state file for faster loading in the futere
        pipeline.save()
        return pipeline
