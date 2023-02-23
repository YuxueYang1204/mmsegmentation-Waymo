import numpy as np
import open3d as o3d
import pickle

palette = np.array([[0, 0, 0], [102, 102, 102], [0, 0, 142], [0, 0, 70],
                    [0, 60, 100], [61, 133, 198], [119, 11, 32], [0, 0, 230],
                    [111, 168, 220], [220, 20, 60], [255, 0, 0], [180, 0, 0],
                    [127, 96, 0], [91, 15, 0], [230, 145, 56], [153, 153, 153],
                    [234, 153, 153], [246, 178, 107], [250, 170, 30],
                    [70, 70, 70], [128, 64, 128], [234, 209, 220],
                    [217, 210, 233], [244, 35, 232], [107, 142, 35],
                    [70, 130, 180], [102, 102, 102], [0, 255, 0], [0, 0, 255],
                    [255, 255, 255]])
classes = np.array([
    'UNDEFINED', 'EGO_VEHICLE', 'CAR', 'TRUCK', 'BUS', 'OTHER_LARGE_VEHICLE',
    'BICYCLE', 'MOTORCYCLE', 'TRAILER', 'PEDESTRIAN', 'CYCLIST',
    'MOTORCYCLIST', 'BIRD', 'GROUND_ANIMAL', 'CONSTRUCTION_CONE_POLE', 'POLE',
    'PEDESTRIAN_OBJECT', 'SIGN', 'TRAFFIC_LIGHT', 'BUILDING', 'ROAD',
    'LANE_MARKER', 'ROAD_MARKER', 'SIDEWALK', 'VEGETATION', 'SKY', 'GROUND',
    'DYNAMIC', 'STATIC', 'NOTREGISTER'])
box_palette = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

with open('demo.pkl', 'rb') as f:
    demo = pickle.load(f)
points = demo['points']
points_label = points[:, -1].astype(np.uint8)
bbox3d = demo['box']
bbox3d_label = demo['label']
colors = np.array(palette[points_label], dtype=np.float32) / 255

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
point_cloud.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option()
opt = vis.get_render_option()
opt.point_size = 1.0
opt.background_color = np.array([0, 0, 0])
vis.add_geometry(point_cloud)
for i in range(len(bbox3d)):
    center = bbox3d[i, 0:3]
    dim = bbox3d[i, 3:6]
    yaw = np.zeros(3)
    yaw[2] = -bbox3d[i, 6]
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
    center[2] += dim[2] / 2  # bottom center to gravity center
    box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    line_set.paint_uniform_color(box_palette[bbox3d_label[i]])
    # draw bboxes on visualizer
    vis.add_geometry(line_set)
vis.run()
