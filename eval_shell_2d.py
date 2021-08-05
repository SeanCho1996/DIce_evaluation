import numpy as np
from PIL import Image
import os
from collections import OrderedDict
import pandas as pd
import argparse

from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient


join = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", "-p", type=str, help="predicted mask folder path")
parser.add_argument("--gt_path", "-g", type=str, help="ground truth mask folder path")
parser.add_argument("--report_path", "-r", type=str , help='where to save evaluation csv')
parser.add_argument("--num_classes", "-n", type=int, help="total number of segmentation classes(bg included)")
args = parser.parse_args()

seg_path = args.pred_path
gt_path = args.gt_path
save_path = args.report_path
num_classes = args.num_classes

save_name = 'eval.csv'
assert (os.path.exists(seg_path) and os.path.exists(gt_path))
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png')]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
for i in range(1, num_classes):
    seg_metrics['DSC_{}'.format(i)] = list()
    # seg_metrics['NSD-1mm_{}'.format(i)] = list()


for name in filenames:
    seg_metrics['Name'].append(name)

    # load grond truth and segmentation
    gt_img = Image.open(join(gt_path, name))
    gt_np = np.array(gt_img).astype(int)
    seg_img = Image.open(join(seg_path, name))
    seg_np = np.array(seg_img).astype(int)

    for i in range(1, num_classes):
        if np.sum(gt_np==i)==0 and np.sum(seg_np==i)==0:
            DSC_i = 1
            # NSD_i = 1
        elif np.sum(gt_np==i)==0 and np.sum(seg_np==i)>0:
            DSC_i = 0
            # NSD_i = 0
        else:
            # surface_distances = compute_surface_distances(gt_np==i, seg_np==i, case_spacing)
            DSC_i = compute_dice_coefficient(gt_np==i, seg_np==i)
            # NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
        seg_metrics['DSC_{}'.format(i)].append(DSC_i)
        # seg_metrics['NSD-1mm_{}'.format(i)].append(NSD_i)  

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(join(save_path, save_name), index=False)
