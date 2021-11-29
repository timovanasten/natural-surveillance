import os
import copy
import time
import glob
import torch
import logging

from torch.utils.data import DataLoader

# define project dependency

# project dependence
from .common_pytorch.dataset.all_dataset import *
from .common_pytorch.config_pytorch import update_config_from_file, s_config
from .common_pytorch.common_loss.balanced_parallel import DataParallelModel
from .common_pytorch.net_modules import infer_net

from .common_pytorch.blocks.resnet_pose import get_default_network_config, get_pose_net
from .common_pytorch.loss.heatmap import get_default_loss_config, get_merge_func

from .core.loader import InferFacadeDataset
exec('from .common_pytorch.blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')


def main(data_path, config_path, model_path, output_path, image_file_list=None):
    """
        Infers the window locations in the images
        :param output_path: Path to export the visualizations of the results to.
        :param config_path: Path to the .yaml file containing the configuration of the network.
        :param model_path: Path to the .pth.tar file containing the trained model.
        :param data_path: Path to the directory containing the images to be labeled
        :param image_file_list: List of image path file strings.
        If provided only the files in this list will be labeled instead of the entire data directory
        :return: tuple (windows_list_with_score, imdb_list)
        windows_list_with_score - list of arrays (one per image) of dicts (one per window) containing:
            position: (array of keypoints [4x [x,y, certainty]])
            img_id: index of the image in imdb_list
            score: average of the scores of the keypoints
        img_list - list of image names
        """
    # parsing specific config
    s_config_file = config_path
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config.dataset.path = data_path
    if not os.path.exists(data_path):
        print("invalid infer path")
        exit(-1)

    # create log and path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = logging.getLogger()

    # define devices create multi-GPU context
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    # devices = [int(i) for i in config.pytorch.gpus.split(',')]
    # logger.info("Using Devices: {}".format(str(devices)))

    # label, loss, metric and result
    merge_hm_flip_func, merge_tag_flip_func = get_merge_func(config.loss)

    # dataset, basic imdb
    if image_file_list is not None:
        infer_imdb = image_file_list
    else:
        infer_imdb = glob.glob(data_path + '/*.jpg')
        infer_imdb += glob.glob(data_path + '/*.png')
    logger.debug("Labeling %s images", len(infer_imdb))
    infer_imdb.sort()

    dataset_infer = InferFacadeDataset(infer_imdb, config.train.patch_width, config.train.patch_height, config.aug)

    # here disable multi-process num_workers, because limit of GPU server
    batch_size = config.dataiter.batch_images_per_ctx
    infer_data_loader = DataLoader(dataset=dataset_infer, batch_size=batch_size)

    # prepare network
    assert os.path.exists(model_path), 'Cannot find model!'
    logger.debug("Loading model from %s", model_path)
    net = get_pose_net(config.network, config.loss.ae_feat_dim,
                       num_corners if not config.loss.useCenterNet else num_corners + 1)
    net = DataParallelModel(net).cpu()
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))  # or other path/to/model
    net.load_state_dict(ckpt['network'])
    logger.debug("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # train and valid
    logger.debug("Test DB size: %s.", len(infer_imdb))

    begin_t = time.time()
    windows_list_with_score, imdb_list = \
        infer_net(infer_data_loader, net, merge_hm_flip_func, merge_tag_flip_func, flip_pairs,
              config.train.patch_width, config.train.patch_height, config.loss, config.test, output_path)
    end_t = time.time() - begin_t
    logger.debug('Speed: %.3f second per image', (end_t / len(infer_imdb)))
    return windows_list_with_score, imdb_list