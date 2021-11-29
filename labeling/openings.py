#  Copyright (c) 2021, Timo van Asten
from shapely.geometry import Polygon
from datasources.gsv import GSVImageMeta
from typing import List
from util.serialization import Collection
from config import settings


class BuildingOpening:
    """
    Class that represents a building opening, i.e. either a door or a window.
    """
    def __init__(self, keypoints, confidence):
        """
        Creates a building opening object.
        :param keypoints: 4x2 array of [[x,y],...] where the origin (0,0) is in the top right of the image. The
        keypoints are ordered: top left, bottom left, bottom right, top right.
        :param confidence: Confidence of the detected window.
        """
        self.keypoints = keypoints
        self.confidence = float(confidence)

    def area_px(self):
        """Calculates the area in pixels
        :return Integer containing the area
        """

        polygon = Polygon(self.keypoints)
        return polygon.area

    def centroid(self):
        """
        Calculate the centroid of the opening.
        :return: (x,y) tuple containing the centroid of the window within the image.
        """
        polygon = Polygon(self.keypoints)
        return polygon.centroid.coords[0]


class FacadeLabelingResult:
    """
    Object that contains the result of feeding a StreetView image trough a facade labeling algorithm.
    Contains the StreetView image metadata and a list with the detected building openings within the image
    """
    def __init__(self, street_view_meta: GSVImageMeta, opening_list: List[BuildingOpening]):
        self.street_view_meta = street_view_meta
        self.opening_list = opening_list


class FacadeLabelingResultCollection(Collection[FacadeLabelingResult]):
    def __init__(self, collection_name: str, object_list: List[FacadeLabelingResult] = None,
                 sub_dir=settings.labeling.SUB_DIR, json_filename=settings.labeling.JSON_FILENAME):
        super().__init__(collection_name, FacadeLabelingResult, sub_dir, json_filename, object_list)