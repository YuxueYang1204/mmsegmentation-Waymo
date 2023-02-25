import argparse
import mmcv
from mmcv.utils import Config
from mmseg.datasets import build_dataset


CLASSES = ('UNDEFINED', 'EGO_VEHICLE', 'CAR', 'TRUCK', 'BUS',
            'OTHER_LARGE_VEHICLE', 'BICYCLE', 'MOTORCYCLE', 'TRAILER',
            'PEDESTRIAN', 'CYCLIST', 'MOTORCYCLIST', 'BIRD',
            'GROUND_ANIMAL', 'CONSTRUCTION_CONE_POLE', 'POLE',
            'PEDESTRIAN_OBJECT', 'SIGN', 'TRAFFIC_LIGHT', 'BUILDING',
            'ROAD', 'LANE_MARKER', 'ROAD_MARKER', 'SIDEWALK', 'VEGETATION',
            'SKY', 'GROUND', 'DYNAMIC', 'STATIC')


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument('class_sample_idxs_path', help='class_sample_idxs path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    partial_classes = ('BICYCLE', 'MOTORCYCLE', 'CYCLIST', 'MOTORCYCLIST')
    print(f'Config:\n{cfg.pretty_text}')
    dataset = build_dataset(cfg.data)
    class_sample_idxs = {
        CLASSES.index(class_name): []
        for class_name in partial_classes
    }
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        sample_cat_ids = dataset.get_cat_ids(idx, partial_classes)
        for cat_id in sample_cat_ids:
            class_sample_idxs[cat_id].append(idx)
        prog_bar.update()
    mmcv.dump(class_sample_idxs, args.class_sample_idxs_path)


if __name__ == '__main__':
    main()