import os.path as osp
import os

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from torch.utils.data import Dataset

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import LoadNPYAnnotations, Compose
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class WaymoDataset(CustomDataset):
    """Waymo dataset.

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_semantic_label.npy' for Waymo dataset.
    """

    CLASSES = ('UNDEFINED', 'EGO_VEHICLE', 'CAR', 'TRUCK', 'BUS',
               'OTHER_LARGE_VEHICLE', 'BICYCLE', 'MOTORCYCLE', 'TRAILER',
               'PEDESTRIAN', 'CYCLIST', 'MOTORCYCLIST', 'BIRD',
               'GROUND_ANIMAL', 'CONSTRUCTION_CONE_POLE', 'POLE',
               'PEDESTRIAN_OBJECT', 'SIGN', 'TRAFFIC_LIGHT', 'BUILDING',
               'ROAD', 'LANE_MARKER', 'ROAD_MARKER', 'SIDEWALK', 'VEGETATION',
               'SKY', 'GROUND', 'DYNAMIC', 'STATIC')

    PALETTE = [[0, 0, 0], [102, 102, 102], [0, 0, 142], [0, 0, 70],
               [0, 60, 100], [61, 133, 198], [119, 11, 32], [0, 0, 230],
               [111, 168, 220], [220, 20, 60], [255, 0, 0], [180, 0, 0],
               [127, 96, 0], [91, 15, 0], [230, 145, 56], [153, 153, 153],
               [234, 153, 153], [246, 178, 107], [250, 170, 30], [70, 70, 70],
               [128, 64, 128], [234, 209, 220], [217, 210, 233],
               [244, 35, 232], [107, 142, 35], [70, 130, 180], [102, 102, 102],
               [102, 102, 102], [102, 102, 102]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_semantic_label.npy',
                 **kwargs):
        super(WaymoDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.gt_seg_map_loader = LoadNPYAnnotations(
            reduce_zero_label=self.reduce_zero_label)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            if result.shape[-1] == 1:
                out = np.squeeze(result, axis=-1)

            palette = np.zeros((len(self.CLASSES), 3), dtype=np.uint8)
            palette[:] = self.PALETTE

            out = palette[out]
            output = Image.fromarray(out.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def get_cat_ids(self, idx, partial_classes):
        """Get category distribution of single image.

        Args:
            idx (int): Index of the data_info.
            partial_classes (tuple): Classes requiring repetition.

        Returns:
            dict[list]: for each category, if the current image
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        seg_map = self.get_gt_seg_map_by_idx(idx)
        seg_classes = np.unique(seg_map)
        cat_ids = []
        for seg_class in seg_classes:
            if self.CLASSES[seg_class] in partial_classes:
                cat_ids.append(seg_class)
        return cat_ids


@DATASETS.register_module()
class MultiViewDataset(Dataset):
    """Multi-view dataset. An example of file structure is as followed.
    .. code-block:: none

        ├── data
        │   ├── waymo
        │   │   ├── kitti_format
        │   │   │   ├── training
        │   │   │   │   ├── image_0
        │   │   │   │   │   ├── 0000000.png
        │   │   │   │   │   ├── {image_idx}.png
        │   │   │   │   ├── image_1
        │   │   │   │   ├── image_2
        │   │   │   │   ├── image_3
        │   │   │   │   ├── image_4
        │   │   │   │   ├── velodyne
        │   │   │   │   │   ├── 0000000.bin
        │   │   │   │   │   ├── {image_idx}.bin
        │   │   │   │   ├── velodyne_projection
        │   │   │   │   │   ├── 0000000.bin
        │   │   │   │   │   ├── {image_idx}.bin

    Args:
        pipeline (list[dict]): Processing pipeline
        data_root (str): Data root for images and velodyne files.
        img_suffix (str): Suffix of images. Default: '.png'
        velodyne_suffix (str): Suffix of velodyne and velodyne_projection
        files. Default: '.bin'
    """

    CLASSES = ('UNDEFINED', 'EGO_VEHICLE', 'CAR', 'TRUCK', 'BUS',
               'OTHER_LARGE_VEHICLE', 'BICYCLE', 'MOTORCYCLE', 'TRAILER',
               'PEDESTRIAN', 'CYCLIST', 'MOTORCYCLIST', 'BIRD',
               'GROUND_ANIMAL', 'CONSTRUCTION_CONE_POLE', 'POLE',
               'PEDESTRIAN_OBJECT', 'SIGN', 'TRAFFIC_LIGHT', 'BUILDING',
               'ROAD', 'LANE_MARKER', 'ROAD_MARKER', 'SIDEWALK', 'VEGETATION',
               'SKY', 'GROUND', 'DYNAMIC', 'STATIC')

    PALETTE = [[0, 0, 0], [102, 102, 102], [0, 0, 142], [0, 0, 70],
               [0, 60, 100], [61, 133, 198], [119, 11, 32], [0, 0, 230],
               [111, 168, 220], [220, 20, 60], [255, 0, 0], [180, 0, 0],
               [127, 96, 0], [91, 15, 0], [230, 145, 56], [153, 153, 153],
               [234, 153, 153], [246, 178, 107], [250, 170, 30], [70, 70, 70],
               [128, 64, 128], [234, 209, 220], [217, 210, 233],
               [244, 35, 232], [107, 142, 35], [70, 130, 180], [102, 102, 102],
               [102, 102, 102], [102, 102, 102]]

    def __init__(self,
                 pipeline,
                 data_root,
                 img_suffix='.png',
                 velodyne_suffix='.bin'):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.velodyne_suffix = velodyne_suffix

        # load infos
        self.infos = self.load_infos(self.data_root, self.img_suffix,
                                     self.velodyne_suffix)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.infos)

    def load_infos(self, data_root, img_suffix, velodyne_suffix):
        """Load annotation from directory.

        Args:
            data_root (str): Data root for images and velodyne files.
            img_suffix (str): Suffix of images.
            velodyne_suffix (str): Suffix of velodyne and
            velodyne_projection files.

        Returns:
            list[dict]: All infos of dataset.
        """
        infos = []
        img_dir = osp.join(data_root, 'image_0')
        infos = os.listdir(img_dir)
        infos = sorted(infos)

        print_log(f'Loaded {len(infos)} infos', logger=get_root_logger())
        return infos

    def pre_pipeline(self, info):
        """Prepare results dict for pipeline."""
        out = []
        for cam in range(5):
            out.append(
                dict(img_info={'filename': info},
                     img_prefix=osp.join(self.data_root, f'image_{cam}')))
        return out

    def __getitem__(self, idx):
        """Get inference data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Inference data after pipeline with new keys introduced by
                pipeline.
        """

        info = self.infos[idx]
        results = self.pre_pipeline(info)
        data = []
        for result in results:
            data.append(self.pipeline(result))
        return data

    def format_results(self, results, imgfile_prefix, img_metas=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            img_metas (list[DC]): Meta information.


        """
        img_name = img_metas[0].data[0][0]['ori_filename']
        velodyne_name = img_name.replace(self.img_suffix, self.velodyne_suffix)
        velodyne_path = osp.join(self.data_root, 'velodyne', velodyne_name)
        velodyne_cp_path = osp.join(self.data_root, 'velodyne_projection',
                                    velodyne_name)
        points = np.fromfile(
            velodyne_path, dtype=np.float32, count=-1).reshape([-1, 6])
        cp_points = np.fromfile(
            velodyne_cp_path, dtype=np.uint16, count=-1).reshape((-1, 6))
        assert points.shape[0] == cp_points.shape[0],\
            f"{velodyne_name} don't match!"
        labels = np.zeros(points.shape[0], dtype=np.uint8) + len(self.CLASSES)
        valid_idx = np.where(cp_points[:, 1:3].all(axis=-1))[0]
        for cam in range(5):
            cam_idx = np.where((cp_points[valid_idx][:, 0] - 1 == cam))[0]
            abs_idx = valid_idx[cam_idx]
            assert (cp_points[abs_idx, 0] - 1 == cam).all()
            temp_x = cp_points[abs_idx, 1] - 1
            temp_y = cp_points[abs_idx, 2] - 1
            labels[abs_idx] = results[cam][temp_y, temp_x]
        labels.tofile(osp.join(imgfile_prefix, velodyne_name))
