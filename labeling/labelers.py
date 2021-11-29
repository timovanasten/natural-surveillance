#  Copyright (c) 2021, Timo van Asten
from .heatmap_fusion import infer
from .heatmap_fusion.common.utility.visualization import vis_eval_result
import os
from tqdm import tqdm
import numpy as np
from .openings import BuildingOpening, FacadeLabelingResult, FacadeLabelingResultCollection
from datasources.gsv import GSVImageMetaCollection, GSVImageMeta
from typing import List
from config import settings
import logging
log = logging.getLogger(__name__)


class FacadeLabeler:
    @classmethod
    def label(cls, image_collection: GSVImageMetaCollection, export_to_json=True,
              export_visualizations=settings.labeling.OUTPUT_VISUALIZATIONS) -> FacadeLabelingResultCollection:
        """
        Label facades present in the StreetView images.
        :param image_collection: StreetViewImageCollection to detect windows within.
        :param export_to_json: Boolean indicating whether or not to export the data to a json file
        :param export_visualizations: If set to True, a folder containing the visualzations of
        the detections will be created.
        :return: An FacadeLabelingResultCollection containing the results for each input image
        """
        output_collection = FacadeLabelingResultCollection(image_collection.collection_name)
        detected_windows = HeatmapFusion(output_collection.collection_dir).detect_openings(image_collection,
                                                                                           export_visualizations)
        output_collection.object_list = detected_windows
        if export_to_json:
            output_collection.export_to_json()
        return output_collection


class HeatmapFusion:
    """Class that transforms the the input and output to interface with the pre-existing heatmap fusion
    facade labeling algorithm"""
    def __init__(self, output_path,
                 config_path=settings.labeling.MODEL_CONFIG_PATH,
                 model_path=settings.labeling.TRAINED_MODEL_PATH):
        """
        :param output_path: Path to export the visualizations of the results to.
        :param config_path: Path to the .yaml file containing the configuration of the network.
        :param model_path: Path to the .pth.tar file containing the trained model.
        """
        self.output_path = output_path
        self.config_path = config_path
        self.model_path = model_path

    def detect_openings(self, image_collection: GSVImageMetaCollection,
                        output_visualizations) -> List[FacadeLabelingResult]:
        """
        Detects openings within the images in the provide Google Street View images.
        :param image_collection: GSVImageMetaCollection with the images to label.
        :param output_visualizations: If set to True, a set of images will be creates showing where in the image
        openings have been detected.
        :return: List of FacadeLabelingResult objects, one for each image.
        """
        file_list = image_collection.to_file_list()
        BATCH_SIZE = settings.labeling.BATCH_SIZE
        batches = [file_list[x:x + BATCH_SIZE] for x in range(0, len(file_list), BATCH_SIZE)]
        log.info("Split %s images in %s batches with a maximum size of %s images",
                 len(file_list), len(batches), BATCH_SIZE)
        log.info("Be patient, this could take a while...")
        opening_list = []
        image_info = []
        for batch_file_list in tqdm(batches):
            opening_list_batch, image_info_batch = infer.main(
                data_path=str(image_collection.collection_dir),
                output_path=self.output_path,
                config_path=self.config_path,
                model_path=self.model_path,
                image_file_list=batch_file_list)
            if output_visualizations:
                self.visualize(opening_list_batch, image_info_batch)
            opening_list.extend(opening_list_batch)
            image_info.extend(image_info_batch)
        return self._process_output(opening_list, image_info, image_collection.object_list)

    @staticmethod
    def _process_output(window_list, imdb_list, street_view_metas: List[GSVImageMeta]) -> List[FacadeLabelingResult]:
        """
        Removes keypoint scores and adds image name to output
        :param window_list: Window predictions of the heatmap fusion algorithm
        :param imdb_list: Image database output of the algorithm containing relative image paths and image sizes
        :param street_view_metas List of StreetViewMeta objects that correspond to the input of the
        window detection algorithm.
        :return: List of FacadeLabelingResult objects, one for each image.
        """
        pairings: List[FacadeLabelingResult] = []
        for img_index, detected_windows in enumerate(window_list):
            # Match output to the correct Street View metadata
            matched = False
            output_filename = os.path.basename(imdb_list[img_index]['image'])
            for image_meta in street_view_metas:
                gsv_meta_filename = image_meta.filename
                if gsv_meta_filename != output_filename:
                    # File names don't match, continue to check next meta
                    continue

                # File names match, create the window object
                matched = True
                # Construct window objects
                window_list: List[BuildingOpening] = []
                for window in detected_windows:
                    # Remove keypoint confidence score to turn it into a 4x2 array of [x,y] coordinates
                    keypoints = np.array(window['position'])[:, :2].copy().tolist()
                    window = BuildingOpening(keypoints, window['score'])
                    window_list.append(window)
                # Construct and return the list of ImageWindowSet
                pairings.append(FacadeLabelingResult(image_meta, window_list))
                # Correct meta was matched to this window detection output, move on to the next image
                break
            if not matched:
                log.warning("Could not find matching Street View metadata for %s", output_filename)
        return pairings

    def visualize(self, opening_list, imdb_list) -> None:
        """
        Creates visualizations displaying the detected building openings.
        :param opening_list: List of openings
        :param imdb_list: List containing information about every image (filename, size etc.)
        :return: None
        """
        log.info("Creating visualizations")
        # pre-processing
        ap_pred = []
        # For every image, get the keypoints
        for s_idx in range(len(opening_list)):
            im = imdb_list[s_idx]['image']

            # aggregate predictions into list
            win_pred = opening_list[s_idx]
            for i in range(len(win_pred)):
                temp = {}
                temp['position'] = np.array(win_pred[i]['position'])[:, :2].copy()  # 4x2 array
                temp['img_id'] = s_idx  # index of image
                temp['score'] = win_pred[i]['score']  # confident
                ap_pred.append(temp)
            filename = os.path.basename(im)
            filename = os.path.join(self.output_path, settings.labeling.VISUALIZATION_DIR, filename)
            vis_eval_result(im, win_pred, plot_line=True, save_filename=filename)