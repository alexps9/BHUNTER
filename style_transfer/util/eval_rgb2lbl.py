import numpy as np
from PIL import Image
import os
import glob
from util import AverageMeter, intersectionAndUnion
import cv2
import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', action="store", type=str)
parser.add_argument('--target_path', action="store", type=str, default="../data/Viper2Label_single/val/B/")
parser.add_argument('--map_cache_dir', action="store", type=str, default="seg_lbl_ind_map")
parser.add_argument('--lbl_cache_dir', action="store", type=str, default="gt_lbl_ind_map")
parser.add_argument('--eval_h', action="store", type=int, default="256")
parser.add_argument('--eval_w', action="store", type=int, default="256")
parser.add_argument('--details', action="store", type=bool, default="True")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

fns = os.path.join(args.exp_name)
fs = glob.glob(fns)
print("V2L experiment, Looking files from:", fns)
rgb2id = {
    (0, 0, 0): (0, "unlabeled"),
    (111, 74, 0): (1, "ambiguous"),	
    (70, 130, 180): (2, "sky"), 
    (128, 64, 128): (3, "road"), 
    (244, 35, 232): (4, "sidewalk"), 
    (230, 150, 140): (5, "railtrack"),
    (152, 251, 152): (6, "terrain"), 
    (87, 182, 35): (7, "tree"), 
    (35, 142, 35): (8, "vegetation"), 
    (70, 70, 70): (9, "building"), 
    (153, 153, 153): (10, "infrastructure"), 
    (190, 153, 153): (11, "fence"), 
    (150, 20, 20): (12, "billboard"), 
    (250, 170, 30): (13, "traffic light"), 
    (220, 220, 0): (14, "traffic sign"), 
    (180, 180, 100): (15, "mobilebarrier"), 
    (173, 153, 153): (16, "firehydrant"),
    (168, 153, 153): (17, "chair"),
    (81, 0, 21): (18, "trash"),
    (81, 0, 81): (19, "trashcan"),
    (220, 20, 60): (20, "person"),
    (255, 0, 0): (21, "animal"),
    (119, 11, 32): (22, "bicycle"),
    (0, 0, 230): (23, "motorcycle"),
    (0, 0, 142): (24, "car"),
    (0, 80, 100): (25, "van"),
    (0, 60, 100): (26, "bus"),
    (0, 0, 70): (27, "truck"),
    (0, 0, 90): (28, "trailer"),
    (0, 80, 100): (29, "train"),
    (0, 100, 100): (30, "plane"),
    (50, 0, 90): (31, "boat"),
}

colors = torch.from_numpy(np.array(list(rgb2id.keys()))).to(device).float()
ids = [rgb2id[i][0] for i in rgb2id.keys()]
class_names = [rgb2id[i][1] for i in rgb2id.keys()]

if not os.path.exists(args.map_cache_dir):
    os.makedirs(args.map_cache_dir)
if not os.path.exists(args.lbl_cache_dir):
    os.makedirs(args.lbl_cache_dir)

def cal_acc(data_list, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if len(data_list) == 0:
        print("Empty list.")
        return

    for i, (image_path, target_path) in enumerate(data_list):
        pred = np.array(Image.open(image_path))
        target = np.array(Image.open(target_path))
        eval_size = (args.eval_w, args.eval_h)
        if pred.shape != eval_size:
            pred = cv2.resize(pred, eval_size, cv2.INTER_NEAREST)
        if target.shape != eval_size:
            target = cv2.resize(target, eval_size, cv2.INTER_NEAREST)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(classes):
        print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

data_list = []
day_list = []
night_list = []
sunset_list = []
snow_list = []
rain_list = []

day = ['068', '066', '069', '065', '067']
night = ['076', '073', '077', '074', '075']
sunset = ['028', '025', '029', '027', '026']
snow = ['040', '042', '041', '043']
rain = ['063', '064']

for f in (fs):
    img = torch.from_numpy(np.array(Image.open(f).convert('RGB'))).to(device) # (h, w, 3)
    img = img.unsqueeze(2).float()
    diff = torch.norm(img - colors, dim=3)
    ind_map = torch.argmin(diff, dim=2)
    lbl = 255 * torch.ones_like(ind_map)
    for i in range(len(colors)):
        lbl[ind_map == i] = ids[i]
    # save the map
    save_dir = os.path.join(args.map_cache_dir, os.path.basename(f))
    im = Image.fromarray(np.uint8(lbl.cpu().numpy()))
    im.save(save_dir)

    vid_ind = os.path.basename(f).split("_")[0]
    img_ind = os.path.basename(f).split("_")[1]
    target_path = os.path.join(args.target_path, vid_ind + "_" + img_ind + "_centered.png")
    gt_cache_dir = os.path.join(args.lbl_cache_dir, vid_ind + "_" + img_ind + "_centered.png")
    if not os.path.exists(gt_cache_dir):

        gt_rgb = torch.from_numpy(np.array(Image.open(target_path).convert('RGB'))).to(device) # (h, w, 3)
        gt_rgb = gt_rgb.unsqueeze(2).float()
        tgt_diff = torch.norm(gt_rgb - colors, dim=3)
        tgt_ind_map = torch.argmin(tgt_diff, dim=2)
        gt = 255 * torch.ones_like(tgt_ind_map)
        for i in range(len(colors)):
            gt[tgt_ind_map == i] = ids[i]
        gt = Image.fromarray(np.uint8(gt.cpu().numpy()))
        gt.save(gt_cache_dir)

    data_list.append((save_dir, gt_cache_dir))
    if args.details:
        if vid_ind in day:
            day_list.append((save_dir, gt_cache_dir))
        elif vid_ind in night:
            night_list.append((save_dir, gt_cache_dir))
        elif vid_ind in sunset:
            sunset_list.append((save_dir, gt_cache_dir))
        elif vid_ind in snow:
            snow_list.append((save_dir, gt_cache_dir))
        elif vid_ind in rain:
            rain_list.append((save_dir, gt_cache_dir))
        else:
            raise ValueError

print("All:")
cal_acc(data_list, len(class_names), class_names)
if args.details:
    print("Day:")
    cal_acc(day_list, len(class_names), class_names)
    print("Night:")
    cal_acc(night_list, len(class_names), class_names)
    print("Sunset:")
    cal_acc(sunset_list, len(class_names), class_names)
    print("Snow:")
    cal_acc(snow_list, len(class_names), class_names)
    print("Rain:")
    cal_acc(rain_list, len(class_names), class_names)
