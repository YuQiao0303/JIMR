import pandas as pd
import numpy as np

MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673], dtype=np.float32)

RFS_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']
CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
CAD_cnts = [555, 1093, 212, 113, 232, 260, 191, 121]
CAD_weights = np.sum(CAD_cnts) / np.array(CAD_cnts)

CAD2ShapeNetID = ['4379243', '3001627', '2871439', '4256520', '2747177', '2933112', '3211117', '2808440']
#################################
not_synthetic_CAD_label_id = [1,2,3,6,7]
synthetic_CAD_label_id = [0,4,5]
not_synthetic_mask = [0,1,1,1,0,0,1,1] # these catogories are not synthetic
synthetic_mask =     [1,0,0,0,1,1,0,0] # these catogories are not synthetic
#################################
CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])} # selected 8 categories from SHAPENETCLASSES
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

# cabinet, display, and bathtub (sink) may fly.
CADNotFly = [0, 1, 2, 3, 4]

# assert exist, label map file.
raw_label_map_file = 'datasets/scannet/rfs_label_map.csv'
raw_label_map = pd.read_csv(raw_label_map_file)

RFS2CAD = {} # RFS --> cad
for i in range(len(raw_label_map)):
    row = raw_label_map.iloc[i]
    RFS2CAD[int(row['rfs_ids'])] = row['cad_ids']

RFS2CAD_arr = np.ones(30) * -1
for k, v in RFS2CAD.items():
    RFS2CAD_arr[k] = v
# RFS2CAD_arr [-1. -1.  5. -1.  1.  3.  0. -1. -1.  2. -1.  5.  0. -1.  5. -1.  1.  7.
#   7. -1. -1.  6.  4.  2.  0. -1. -1. -1. -1. -1.]