from __future__ import annotations
from .building_manager import *
import math
from .latlng import LatLng
from labeling.openings import FacadeLabelingResult, BuildingOpening, FacadeLabelingResultCollection
from typing import List, Union
from datasources.gsv import GSVImageMeta
from pathlib import Path
from config import settings
from tqdm import tqdm
from shapely.geometry import Polygon, Point, LineString
from geopandas import GeoDataFrame, points_from_xy
from util.serialization import Collection
import logging
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt


class Segment:
    def __init__(self, end1: LatLng, end2: LatLng):
        self.end1 = end1
        self.end2 = end2


class LocationEstimate3D:
    def __init__(self,
                 coordinates: LatLng,
                 facade_x: float,
                 facade_y: float,
                 building: Building,
                 segment: Segment):
        """
        Location estimate in 3 dimensions determined by calculating the intersection of a ray with the closest building.
        :param coordinates: Coordinates (latitude and longitude)
        :param facade_x: Position among the x axis of the facade element in meters
        :param facade_y: Height of the window along the y axis of the facade, i.e. the height of the opening
        :param building: Building object that this window belongs to
        :param segment: Segment of the building that this window belongs to
        """
        self.coordinates = coordinates
        self.facade_x = facade_x
        self.facade_y = facade_y
        self.building = building
        self.segment = segment


class OpeningLocationEstimate:
    def __init__(self, building_opening: BuildingOpening, street_view_meta: GSVImageMeta,
                 top_left: LocationEstimate3D,
                 bottom_left: LocationEstimate3D,
                 bottom_right: LocationEstimate3D,
                 top_right: LocationEstimate3D,
                 center_point: LocationEstimate3D):
        """
        Create a OpeningLocationEstimate objects representing the position of the opening in 3D space
        along with the Street View metadata used to compute it.
        :param building_opening: BuildingOpening object that this estimation was calculated for
        :param street_view_meta: Street View metadata that was used to calculate this position
        :param top_left, bottom_left, bottom_right, top_right, center_point:
        Estimated location (lat, long, alt) of all 4 keypoints and center point of the
        opening in 3D space.
        """
        self.building_opening = building_opening
        self.street_view_meta = street_view_meta
        self.top_left: LocationEstimate3D = top_left
        self.bottom_left: LocationEstimate3D = bottom_left
        self.bottom_right: LocationEstimate3D = bottom_right
        self.top_right: LocationEstimate3D = top_right
        self.center_point: LocationEstimate3D = center_point
        self._polygon = self.opening_polygon2d()
        self.opening_size = self._polygon.area
        self.altitude = self._polygon.centroid.coords[0][1]

    def to_dict(self):
        """
        Method used for easy conversion to GeoDataFrame using the DataFrame.from_records() method
        :return: Dict representing the object
        """
        return {
            "latitude": self.center_point.coordinates.lat,
            "longitude": self.center_point.coordinates.lng,
            "opening_location": Point(self.center_point.coordinates.lng, self.center_point.coordinates.lat),
            "building_segment": LineString(
                [Point(self.center_point.segment.end1.lng, self.center_point.segment.end1.lat),
                 Point(self.center_point.segment.end2.lng, self.center_point.segment.end2.lat)]
            ),
            "altitude": self.center_point.facade_y,
            "size": self.opening_size,
            "origin_image": self.street_view_meta.filename,
            "origin_image_latitude": self.street_view_meta.coordinates.lat,
            "origin_image_longitude": self.street_view_meta.coordinates.lng,
            "camera_location": Point(self.street_view_meta.coordinates.lng, self.street_view_meta.coordinates.lat),
            "road_sightline": LineString(
                [Point(self.street_view_meta.coordinates.lng, self.street_view_meta.coordinates.lat),
                 Point(self.center_point.coordinates.lng, self.center_point.coordinates.lat)]
            ),
            "street_index": self.street_view_meta.street_gdf_index,
            "street_segment_index": self.street_view_meta.street_segment_index
        }

    def opening_polygon2d(self):
        """
        :return: 2D Polygon of the opening in the plane of the building facade
        """
        points = []
        # Calculate distance between top left and top right corner
        # Order of the list is: top left, bottom left, bottom right, top right
        x_distance_top = self.top_left.coordinates.get_distance(self.top_right.coordinates)
        points.append(Point(0, self.top_left.facade_y))
        points.append(Point(0, self.bottom_left.facade_y))
        points.append(Point(x_distance_top, self.bottom_right.facade_y))
        points.append(Point(x_distance_top, self.top_right.facade_y))
        return Polygon(points)


class OpeningLocationEstimateCollection(Collection[OpeningLocationEstimate]):
    def __init__(self, collection_name: str, object_list: List[OpeningLocationEstimate] = None,
                 sub_dir=settings.localization.SUB_DIR, json_filename=settings.localization.JSON_FILENAME):
        super().__init__(collection_name, OpeningLocationEstimate, sub_dir, json_filename, object_list,
                         strip_privates=True)

    def to_gdf(self):
        gdf = GeoDataFrame.from_records([location.to_dict() for location in self.object_list])
        gdf = gdf.set_geometry("opening_location")
        return gdf

    def export_to_csv(self):
        csv_file_path = Path(self.collection_dir, settings.localization.CSV_FILENAME)
        self.to_gdf().to_csv(csv_file_path)


class Raycast:
    def __init__(self, camera_latlng, image_width, image_height, target_x, target_y, heading,
                 camera_height=settings.localization.GSV_CAMERA_HEIGHT, pitch=0, fov=90):
        """
        Creates a Raycast object.
        :param camera_latlng:
        :param image_width:
        :param image_height:
        :param target_x: X coordinate of the pixel to target the ray at
        :param target_y: Y coordinate of the pixel to target the ray at
        :param heading: Heading of the street view camera in degrees [0, 360] where 0 is north
        :param camera_height: Height in meters of the camera
        :param pitch: Pitch of the street view camera in degrees [-90, 90]
        :param fov: Field of view in degrees
        """
        self.camera_latlng = camera_latlng
        self.heading = heading
        self.pitch = pitch
        self.fov = fov
        self.screen_width = image_width
        self.screen_height = image_height
        self.screen_x = 2.0 * target_x / image_width - 1.0
        self.screen_y = 1.0 - 2.0 * target_y / image_height
        self.aspect = image_width / image_height
        self.camera_height = camera_height

    def get_raycast(self):
        return {"pitch": self.pitch + 0.5 * self.screen_y * self.fov / self.aspect,
                "heading": self.heading + 0.5 * self.screen_x * self.fov}

    def get_latlng(self): # a point at raycast 50 meters away from the camera
        heading = ((360 - self.get_raycast()["heading"]) + 90) % 360
        x = 50 * math.cos(heading/180.0*math.pi)
        y = 50 * math.sin(heading/180.0*math.pi)
        return self.camera_latlng.get_latlng(x, y)

    def get_range(self, segment):
        x3 = self.camera_latlng.get_xy(segment.end1).x
        y3 = self.camera_latlng.get_xy(segment.end1).y
        x4 = self.camera_latlng.get_xy(segment.end2).x
        y4 = self.camera_latlng.get_xy(segment.end2).y
        heading1 = (math.atan2(x3, y3) / math.pi * 180 + 360) % 360
        heading2 = (math.atan2(x4, y4) / math.pi * 180 + 360) % 360
        heading1 = min(heading1, self.heading + 0.5 * self.fov)
        heading1 = max(heading1, self.heading - 0.5 * self.fov)
        heading2 = min(heading2, self.heading + 0.5 * self.fov)
        heading2 = max(heading2, self.heading - 0.5 * self.fov)
        x1 = (heading1 - self.heading + 0.5 * self.fov) / self.fov * self.screen_width
        x2 = (heading2 - self.heading + 0.5 * self.fov) / self.fov * self.screen_width
        return min(x1,x2), max(x1,x2)

    def intersection2d(self, segment):
        p2 = self.get_latlng()
        if self.camera_latlng == None or p2 == None: return None
        x2 = self.camera_latlng.get_xy(p2).x
        y2 = self.camera_latlng.get_xy(p2).y
        x3 = self.camera_latlng.get_xy(segment.end1).x
        y3 = self.camera_latlng.get_xy(segment.end1).y
        x4 = self.camera_latlng.get_xy(segment.end2).x
        y4 = self.camera_latlng.get_xy(segment.end2).y
        if ( x4 * y2 - x3 * y2 - x2 * y4 + x2 * y3 ) == 0 or ( y4 * x2 - y3 * x2 - y2 * x4 + y2 * x3 ) == 0:
            return None
        x = ( y3 * x4 * x2 - y4 * x3 * x2 ) / ( x4 * y2 - x3 * y2 - x2 * y4 + x2 * y3 )
        y = ( -y3 * x4 * y2 + y4 * x3 * y2) / ( y4 * x2 - y3 * x2 - y2 * x4 + y2 * x3 )
        if min(0, x2) <= x <= max(x3, x4) and max(0, x2) >= x >= min(x3, x4) and \
                min(0, y2) <= y <= max(0, y2) and min(y3, y4) <= y <= max(y3, y4):
            return self.camera_latlng.get_latlng(x, y)
        return None

    def intersection(self, building_manager: BuildingManager) -> Union[LocationEstimate3D, None]:
        """
        Calculates a intersection between a ray and a building
        :param building_manager: BuildingManager object containing the buildings to check for an intersection.
        :return: LocationEstimate3D object if an intersection is found, None otherwise.
        """
        nearby_buildings = building_manager.find_buildings(self.camera_latlng)
        min_dis = 1e99
        inter_latlng = None
        inter_building = None
        inter_x = None
        inter_seg = None

        for building in nearby_buildings:
            x = 0
            for i in range(len(building.coordinates) - 1):
                # Create building segment from consecutive coordinates
                seg = Segment(building.coordinates[i], building.coordinates[i + 1])
                # Check if the ray intersects this building segment
                intersection_2d = self.intersection2d(seg)
                if intersection_2d is not None:
                    distance_to_intersection = self.camera_latlng.get_distance(intersection_2d)
                    # If this intersection is the closest so far, update return value
                    if distance_to_intersection < min_dis:
                        min_dis = distance_to_intersection
                        inter_latlng = intersection_2d
                        inter_building = building
                        inter_seg = seg
                        # Get the distance between the edge of the building segment and the intersection
                        inter_x = x + building.coordinates[i].get_distance(inter_latlng)
                x += building.coordinates[i].get_distance(building.coordinates[i + 1])
        # Calculate the height of the intersection
        inter_height = min_dis * math.tan(self.get_raycast()["pitch"] / 180.0 * math.pi) + self.camera_height

        if inter_latlng is not None:
            # Return the intersection values
            return LocationEstimate3D(
                LatLng(inter_latlng.lat, inter_latlng.lng),
                inter_x,
                inter_height,
                inter_building,
                inter_seg)
        else:
            log.debug("Did not find intersection for Raycast")
            return None

    def plot(self):
        p2 = self.get_latlng()
        plt.scatter(self.camera_latlng.lng, self.camera_latlng.lat)
        plt.plot([self.camera_latlng.lng,p2.lng],[self.camera_latlng.lat,p2.lat])


class LocationEstimator:
    @staticmethod
    def estimate(building_manager: BuildingManager,
                 labeling_result: FacadeLabelingResult) -> List[OpeningLocationEstimate]:
        """
        Calculate the estimated 3D location of windows detected in a image.
        :param labeling_result: FacadeLabelingResult object containing the openings detected in the Street View image
        :param building_manager: BuildingManager object containing the buildings the windows should belong to
        :return: List of WindowLocationEstimate objects containing the estimated location of the openings within the
        image, along with the data used to compute it
        """
        opening_location_estimations: List[OpeningLocationEstimate] = []
        street_view_meta = labeling_result.street_view_meta
        for opening in labeling_result.opening_list:
            # Create an array to hold the locations of all four keypoints
            keypoint_locations = []
            # Calculate the location of the four corner points and the centroid
            keypoints = opening.keypoints + [opening.centroid()]
            for keypoint in keypoints:
                # Create the ray from the Street View camera to the opening keypoints
                target_x, target_y = keypoint
                ray = Raycast(
                    LatLng(street_view_meta.coordinates.lat, street_view_meta.coordinates.lng),
                    street_view_meta.width,
                    street_view_meta.height,
                    target_x, target_y,
                    street_view_meta.heading,
                    fov=street_view_meta.fov,
                    pitch=street_view_meta.pitch)

                # Calculate the intersection between the ray and the buildings
                position_estimation = ray.intersection(building_manager)
                if position_estimation is not None:
                    keypoint_locations.append(position_estimation)
                else:
                    log.debug("Could not find intersection for opening keypoint in image %s, with image coords %s",
                                street_view_meta.filename, opening.centroid())
                    # Don't calculate the positions of the other keypoints
                    break
            # If the locations of all four keypoints and opening center have been have been found,
            # create the OpeningLocationEstimate
            if len(keypoint_locations) == 5:
                opening_location = OpeningLocationEstimate(opening, street_view_meta,
                                                           keypoint_locations[0],
                                                           keypoint_locations[1],
                                                           keypoint_locations[2],
                                                           keypoint_locations[3],
                                                           keypoint_locations[4])
                # Append the estimated position for this window to the list
                opening_location_estimations.append(opening_location)

        # Return a list with the estimated position for each window in the input image
        return opening_location_estimations

    @classmethod
    def estimate_collection(cls,
                            building_manager: BuildingManager,
                            labeling_results: FacadeLabelingResultCollection,
                            export_to_json=True,
                            export_to_csv=True) -> OpeningLocationEstimateCollection:
        """
        Calculate the estimated 3D location for windows detected in a collection of images.
        :param labeling_results: FacadeLabelingResultCollection object containing the windows detected in
        the StreetView images
        :param building_manager: BuildingManager object containing the buildings the windows should belong to
        :param export_to_json: If set to True, a json file with the output will be created
        provided amount of meters will be removed from the results. Set to None to not filter the results.
        :param export_to_csv: If set to True, a csv file with the output will be created
        :return: WindowLocationEstimateCollection containing the estimated location of the windows within the
        images, along with the data used to compute it
        """
        log.info("Localizing windows...")
        window_location_estimates = []
        for labeling_result in tqdm(labeling_results.object_list):
            window_location_estimates += cls.estimate(building_manager, labeling_result)

        # Create output collection and export to file if enabled
        output_collection = OpeningLocationEstimateCollection(labeling_results.collection_name,
                                                              window_location_estimates)
        if export_to_json:
            output_collection.export_to_json()
        if export_to_csv:
            output_collection.export_to_csv()
        return output_collection

