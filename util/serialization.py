#  Copyright (c) 2021, Timo van Asten
from __future__ import annotations
from typing import TypeVar, Generic, List, Optional
import gzip
import jsons
import json
from pathlib import Path
from config import settings
import logging
from geopandas import GeoDataFrame
import geopandas

log = logging.getLogger(__name__)

T = TypeVar('T')


def get_file_path(pipeline_name: str, sub_dir: str = "", file_name: str = "") -> Path:
    """
    Returns a path according to the directory structure used for the project.
    Used to save and load step results to and from the appropriate location.
    :param pipeline_name: Name of the pipeline, will be used to create a folder structure and automatically
    load files.
    :param sub_dir: Subdirectory to save files to, for example 'gsv' or 'labeling'
    :param file_name: Name of the file
    :return: Path object containing the appropriate path
    """
    return Path(settings.OUTPUT_DIR, pipeline_name, sub_dir, file_name)


class Collection(Generic[T]):
    """Class used to export and load data for different stages of the pipeline"""
    def __init__(self, collection_name: str, cls: T, sub_dir: str, json_filename: str, object_list: Optional[List[T]],
                 output_dir: str = settings.OUTPUT_DIR, batch_index: int = 0, strip_privates=False):
        """
        Collection of a list of objects that can be serialized an loaded
        :param collection_name: Name of the collection, will be used to create a folder structure and automatically
        load files.
        :param cls: The class that will be contained in the object list. Used to reconstruct the objects from JSON.
        :param sub_dir: Subdirectory to save files for this collection to, for example 'gsv' or 'labeling'
        :param json_filename: Filename of the JSON file to save the collection to, e.g. 'queries.json'
        :param object_list: List of objects for this collection.
        :param output_dir: Top level directory to save output files to, e.g. './data/'
        :param batch_index: Index used to split large areas into smaller batches.
        :param strip_privates: Boolean indicating if private variables should also be serialized.
        """
        self.collection_name = collection_name
        # Final directory the files are saved to
        self.collection_dir = Path(output_dir, collection_name, sub_dir)
        self._json_filename = json_filename
        self._json_path = Path(self.collection_dir, self._json_filename)
        self.object_list = object_list
        self._cls = cls
        self._strip_privates = strip_privates
        # Create directory if it does not exist
        self.collection_dir.mkdir(parents=True, exist_ok=True)

    def load_from_json(self):
        """
        Load a previously created collection from a JSON file.
        :return: Collection object with loaded data if data loaded successfully, None otherwise
        """
        if not self._json_path.exists():
            log.info("Could not load %s for '%s'. File does not exist: %s", 
                     self._cls.__name__, 
                     self.collection_name, 
                     str(self._json_path))
            return None
        try:
            with gzip.open(self._json_path, 'rt', encoding='UTF-8') as json_file:
                json_dict = json.load(json_file)
            collection = [jsons.load(collection_element, cls=self._cls) for collection_element in json_dict]
            log.info("Loaded %s for '%s'", self._cls.__name__ + "Collection", self.collection_name)
            self.object_list = collection
            return self
        except jsons.exceptions.DeserializationError as e:
            log.info("Error loading %s for '%s'. File is most likely outdated: %s",
                     self._cls.__name__ + "Collection",
                     self.collection_name, e)
            return None

    def export_to_json(self):
        """
        Export the collection to a compressed JSON file.
        :return: None
        """
        log.info("Saving %s...", self._cls.__name__ + "Collection")
        with gzip.open(self._json_path, 'wt', encoding='UTF-8') as outfile:
            json_dict = jsons.dump(self.object_list, strip_privates=self._strip_privates)
            json.dump(json_dict, outfile, indent=settings.JSON_INDENT)
        log.info("Saved %s to %s", self._cls.__name__ + "Collection", self._json_path)


class PickledDataFrame:
    def __init__(self, pipeline_name: str, sub_dir: str, file_name: str):
        self.save_path = get_file_path(pipeline_name, sub_dir, file_name)

    def save_gdf(self, gdf: GeoDataFrame):
        """
        Saves building GeoDataFrame to the path specified during object initialization.
        :return: None
        """
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        log.info("Saving GeoDataFrame...")
        gdf.to_pickle(str(self.save_path))
        log.info("Saved GeoDataFrame to %s", self.save_path)

    def load_gdf(self) -> Optional[GeoDataFrame]:
        """
        Loads GeoDataFrame from file
        :return: GeoDataFrame if successful, None otherwise
        """
        try:
            gdf = GeoDataFrame(geopandas.pd.read_pickle(str(self.save_path)))
            log.info("Loaded GeoDataFrame from %s", self.save_path)
            return gdf
        except Exception as e:
            log.info("Failed to load GeoDataFrame: %s", e)
            return None
