from __future__ import annotations
import requests
from util.coordinates import Coordinates
from util.serialization import Collection
from typing import List, Optional
from config import settings
from pathlib import Path
from datasources.osm import StreetNetwork
import logging
from tqdm import tqdm
from itertools import product, compress
log = logging.getLogger(__name__)


class GSVImageMeta:
    def __init__(self, filename: str,
                 panorama_id: str,
                 coordinates: Coordinates,
                 heading: int, pitch: int, fov: int,
                 height: int, width: int,
                 date: str,
                 street_gdf_index: Optional[tuple],
                 street_segment_index: Optional[int]):
        self.filename = filename
        self.panorama_id = panorama_id
        self.coordinates = coordinates
        self.heading = heading
        self.pitch = pitch
        self.fov = fov
        self.height = height
        self.width = width
        self.date = date
        self.street_gdf_index = street_gdf_index
        self.street_segment_index = street_segment_index

    # Define methods to check for equality:
    def __key(self):
        return self.filename

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, GSVImageMeta):
            return self.__key() == other.__key()
        return NotImplemented


class GoogleStreetViewAPI:
    def __init__(self, output_dir, api_key=settings.GSV_API_KEY, width=640, height=640):
        """
        This class handles API requests to the Google Static Street View API
        :param output_dir: output directory of the images
        :param api_key: obtain it from your Google Cloud Platform console
        :param width: width of the requested images in pixels
        :param height height of the requested images in pixels
        """
        # input params are saved as attributes for later reference
        self.api_key = api_key
        self.output_directory = output_dir
        self.width = width
        self.height = height

    def request_and_store_image(self, image_meta: GSVImageMeta) -> bool:
        """
        Request and store an image from Google Street View Static API
        :param image_meta: Previously requested GSVImageMeta object to request the image for.
        :return: True if the image was successfully retrieved from the API, or was already retrieved previously,
        False otherwise.
        """
        # Set location and heading parameters for the request
        image_request_params = {
            "key": self.api_key,
            "location": f"{image_meta.coordinates.lat},{image_meta.coordinates.lng}",
            "heading": str(image_meta.heading),
            "fov": str(image_meta.fov),
            "pitch": str(image_meta.pitch),
            "size": f"{self.width}x{self.height}",
            "source": "outdoor"  # Only request outdoor images
        }

        # Define path to save the image
        image_path = Path(self.output_directory, image_meta.filename)

        log.debug("Picture available, requesting now...")
        request_response = requests.get(
            'https://maps.googleapis.com/maps/api/streetview?',
            params=image_request_params)
        if request_response.ok:
            self.store_image(image_path, request_response.content)
            return True
        else:
            log.error("Request for image failed with response: %s", request_response.status_code)
            return False

    def is_previously_requested(self, image_meta: GSVImageMeta):
        # Define path to save the image
        image_path = Path(self.output_directory, image_meta.filename)
        # Check if the image is previously requested
        if image_path.exists():
            return True
        else:
            return False

    def request_and_store_images(self, image_metas: List[GSVImageMeta]) -> None:
        """
        Get pictures and metadata for a given set of coordinates and headings.
        :param image_metas: Set with GSVImageMeta objects to request the images for.
        :return: List with StreetViewImageMeta objects
        """
        log.info("Retrieving Street View images")
        # Check if the images where previously downloaded
        previously_downloaded = [self.is_previously_requested(image_meta) for image_meta in image_metas]
        previously_downloaded_count = sum(previously_downloaded)
        if previously_downloaded_count > 0:
            log.info("%s/%s images have been previously downloaded", previously_downloaded_count, len(image_metas))
        to_request = list(compress(image_metas, [not downloaded for downloaded in previously_downloaded]))

        # If there are images to be requested, check the budget and request them from the api
        if len(to_request) > 0 and GSVBudget().can_download(len(to_request)):
            failed = 0
            log.info("Requesting the corresponding images for %s GSVImageMeta objects from the Street View API",
                     len(to_request))
            for image_meta in tqdm(to_request):
                success = self.request_and_store_image(image_meta)
                if not success:
                    failed += 1
            log.info("Retrieved %s images from the Street View API, %s request(s) failed",
                     len(to_request) - failed, failed)

    def request_metas(self, query_collection: GSVQueryCollection) -> List[GSVImageMeta]:
        log.info("Requesting image metadata for %s GSVRequest objects from the Street View API",
                 len(query_collection.object_list))
        failed = 0
        meta_list = []
        for query in tqdm(query_collection.object_list):
            meta = self.request_meta(query)
            if meta is None:
                failed += 1
            else:
                meta_list.append(meta)

        nr_retrieved_metas = len(meta_list)
        log.info("Retrieved metadata for %s queries from the Street View API, %s request(s) failed",
                 nr_retrieved_metas, failed)

        # Remove duplicate queries:
        meta_list = list(set(meta_list))
        duplicate_count = nr_retrieved_metas - len(meta_list)
        log.info("Removed %s duplicate results metadata results",
                 duplicate_count)
        return meta_list

    def request_meta(self, query: GSVRequest) -> Optional[GSVImageMeta]:
        """
        Method to query the Street View metadata for the provided location.
        :param query GSVRequest object to retrieve the metadata for
        :return GSVImageMeta object if the request to the server was successful
        and an image is available for the request, None otherwise
        """
        # Set location parameter for the request
        # Set location and heading parameters for the request
        meta_request_params = {
            "key": self.api_key,
            "location": f"{query.coordinates.lat},{query.coordinates.lng}",
            "source": "outdoor"  # Only request outdoor images
        }

        # Request the metadata from the API
        request_response = requests.get(
            'https://maps.googleapis.com/maps/api/streetview/metadata?',
            params=meta_request_params)

        # Check if request was successful
        if not request_response.ok:
            log.warning("Failed to obtain metadata from StreetView API with status code %s!",
                        request_response.status_code)
            return None

        # If request was successful, store JSON metadata response
        meta_data = request_response.json()

        # Check if an image is available for the request
        if not self.image_available(meta_data):
            log.info("No Street View images available from the API for lat: %s, lng: %s",
                     query.coordinates.lat, query.coordinates.lng)
            return None

        # If the image is available return the GSVImageMeta object
        file_name = f"{meta_data['pano_id']}_h{query.heading}_p{query.pitch}.png"
        return GSVImageMeta(
            file_name,
            meta_data['pano_id'],
            Coordinates(meta_data["location"]["lat"], meta_data["location"]["lng"]),
            query.heading,
            query.pitch,
            query.fov,
            self.height,
            self.width,
            meta_data["date"],
            query.street_gdf_index,
            query.street_segment_index
        )

    @staticmethod
    def image_available(meta_json_response: dict):
        """
        :param meta_json_response: JSON response from the Street View Image Metadata API.
        :return True if the metadata response from the Google Street View server indicates an image is available."""
        is_available = meta_json_response["status"] == 'OK'
        return is_available

    def store_image(self, image_path: Path, image):
        """Stores a image to file"""
        image_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Saving image to %s", self.output_directory)
        with open(image_path, 'wb') as file:
            file.write(image)


class GSVRequest:
    def __init__(self, coordinates: Coordinates,
                 heading: int,
                 fov: int = 90,
                 pitch: int = 0,
                 street_gdf_index: Optional[tuple] = None,
                 street_segment_index: Optional[int] = None):
        """
        Create a GSVRequest object with the required parameters to request a Google Street View image
        :param coordinates: Coordinate object with latitude and longitude
        :param heading: Indicates the compass heading of the camera. Accepted values are from 0 to 360
        (both values indicating North, with 90 indicating East, and 180 South).
        :param fov: Field of view (default is 90) determines the horizontal field of view of the image.
        The field of view is expressed in degrees, with a maximum allowed value of 120.
        When dealing with a fixed-size viewport, as with a Street View image of a set size, field of view in essence
        represents zoom, with smaller numbers indicating a higher level of zoom.
        :param pitch: Pitch (default is 0) specifies the up or down angle of the camera relative to the Street View
        vehicle. This is often, but not always, flat horizontal. Positive values angle the camera up (with 90 degrees
        indicating straight up); negative values angle the camera down (with -90 indicating straight down).
        :param street_gdf_index: Index (u, v, key) of the street network GeoDataFrame.
        Used to aggregate results back to the street level later in the pipeline.
        :param street_segment_index: Index indicating which street segment of the street this query belongs to.
        Can be used to index the 'street_segments' field of the GeoDataFrame.
        Used to aggregate results back to the street level later in the pipeline.
        """
        self.coordinates = coordinates
        self.fov = fov
        self.heading = heading
        self.pitch = pitch
        self.street_gdf_index = street_gdf_index
        self.street_segment_index = street_segment_index

    # Define methods to check for equality:
    def __key(self):
        return self.coordinates.lat, self.coordinates.lng, self.fov, self.heading, self.pitch

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, GSVRequest):
            return self.__key() == other.__key()
        return NotImplemented


class GSVQueryCollection(Collection[GSVRequest]):
    """Collection of GSVRequest objects to be used to query the Google Street View API"""
    def __init__(self, collection_name: str,
                 object_list: List[GSVRequest] = None,
                 sub_dir=settings.gsv.SUB_DIR, json_filename=settings.gsv.QUERY_JSON_FILENAME):
        super().__init__(collection_name, GSVRequest, sub_dir, json_filename, object_list)

    def create_from_street_network(self, network: StreetNetwork, fov=90, export_to_json=True) \
            -> Optional[GSVQueryCollection]:
        """
        Populates the collection with queries generated from a StreetNetwork object
        :param network: StreetNetwork object to be used to generate the queries
        :param fov: Field of view to set for each query object
        :param export_to_json: If set to True the collection will be saved to a JSON file
        :return: The populated GSVQueryCollection object
        """
        if network.street_network_gdf is None:
            log.warning("Street network is not set yet, returning unaltered collection")
            return self

        log.info("Creating new GSVQueryCollection '%s' from street network", self.collection_name)
        HEADING_OFFSETS = [90, 270]  # Degrees offset in heading to get the direction perpendicular to to the street
        query_list = []

        # Generate queries from GeoDataFrame
        for offset, street in product(HEADING_OFFSETS, network.street_network_gdf.itertuples(index=True)):
            for segment_index, (sample_point, bearing) in enumerate(street.sample_points_with_bearing):
                heading = int(round(bearing + offset) % 360)
                query = GSVRequest(Coordinates(sample_point.y, sample_point.x),
                                   heading,
                                   fov,
                                   street_gdf_index=street.Index,
                                   street_segment_index=segment_index)
                query_list.append(query)

        # Remove duplicate queries:
        query_list = list(set(query_list))
        # Populate collection object
        self.object_list = query_list
        # Export if configured
        if export_to_json:
            self.export_to_json()
        return self

    def set_fov(self, fov):
        """
        Sets the field of view for all GSVRequest objects in the collection to the specified value.
        :param fov: field of view to be set. Maximum is 120.
        :return: The updated collection.
        """
        log.info("Set field of view for collection '%s' to %s.", self.collection_name, fov)
        updated_query_list = []
        for query in self.object_list:
            query.fov = fov
            updated_query_list.append(query)
        self.object_list = updated_query_list
        return self


class GSVImageMetaCollection(Collection[GSVImageMeta]):
    """Helper class to store and load metadata of a collection of Street View images"""
    def __init__(self, collection_name: str,
                 object_list: List[GSVImageMeta] = None,
                 sub_dir=settings.gsv.SUB_DIR, json_filename=settings.gsv.META_JSON_FILENAME):
        super().__init__(collection_name, GSVImageMeta, sub_dir, json_filename, object_list)
        self.image_dir = Path(self.collection_dir, settings.gsv.image_dir)
        self.api = GoogleStreetViewAPI(self.image_dir)

    def create_from_api(self,
                        query_collection: GSVQueryCollection,
                        export_to_json=True) -> GSVImageMetaCollection:
        """
        Creates an street view image collection and saves it to a JSON file if specified.
        :param query_collection: List with GSVRequest objects to request from the api
        :param export_to_json: If set to True the collection will be saved to a JSON file.
        :return: Reference to this StreetViewImageCollection object with populated data
        """
        log.info("Creating new GSVImageCollection collection: '%s' from GSV API.", self.collection_name)
        meta_list = self.api.request_metas(query_collection)
        self.object_list = meta_list
        if export_to_json:
            self.export_to_json()
        return self

    def request_images(self):
        self.api.request_and_store_images(self.object_list)

    def to_file_list(self) -> List[str]:
        """Returns a list with strings containing the file paths of the images in the collection."""
        file_list = []
        for meta in self.object_list:
            file_list.append(str(Path(self.image_dir, meta.filename)))
        return file_list


class GSVBudget:
    def __init__(self):
        self.budget: float = self.load()

    def can_download(self, nr_of_images, image_price=settings.gsv.DOLLAR_PRICE_PER_IMAGE) -> bool:
        """
        Returns True if there is enough budget available to download the request number of
        Street View images from the Google Static Street View API.
        If enough budget is available, the estimated price will be deducted from the available budget.
        :param nr_of_images: amount of images to check the budget for
        :param image_price: price per image downloaded from the API
        :return: True if enough budget is available, False otherwise
        """
        estimated_price = nr_of_images * image_price
        log.info("Estimated price for the requested Google Street View images will be $%s", estimated_price)
        log.info("Currently available budget: $%s", self.budget)
        prospected_budget = self.budget - estimated_price
        if prospected_budget < 0:
            log.warning("This download cannot proceed as it would exceed the available budget by $%s",
                        -prospected_budget)
            return False
        else:
            log.info("Enough budget available. Download can proceed.")
            self._deduct(estimated_price)
            return True

    def _deduct(self, amount):
        """
        Deducts the amount from the budget and saves the new budget to file.
        :param amount: Amount in dollar to deduct
        :return: None
        """
        old_budget = self.budget
        self.budget -= amount
        self.save()
        log.info("Deducted $%s from the remaining $%s Google Street View API budget.", amount, old_budget)
        log.info("New budget: $%s", self.budget)

    def reset_budget(self, reset_to=settings.gsv.DEFAULT_BUDGET_DOLLAR):
        """
        Resets the budget to the default budget or the budget provided and saves the new budget to file.
        :param reset_to: Dollar amount to reset the budget to.
        :return: None
        """
        self.budget = reset_to
        self.save()

    def save(self):
        with open(settings.gsv.BUDGET_FILE, 'w') as budget_file:
            budget_file.write(str(self.budget))

    @staticmethod
    def load() -> float:
        """
        Load the remaining budget from the budget file specified in the settings.
        :return: The loaded remaining budget.
        """
        try:
            with open(settings.gsv.BUDGET_FILE, 'r') as budget_file:
                budget = float(budget_file.readline())
                return budget
        except:
            log.error("Could not load Google Street View Budget from %s. "
                      "Terminating the program to prevent unwanted expenses", settings.gsv.BUDGET_FILE)
            exit(-1)



