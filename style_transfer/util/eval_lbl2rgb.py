import numpy as np
from PIL import Image
import os
import glob
from util import AverageMeter, intersectionAndUnion
import cv2
import tqdm
import torch
import argparse
import fcn


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', action="store", type=str)
parser.add_argument('--target_path', action="store", type=str, default="../data/viper/val/resize_cls/")
parser.add_argument('--pred_cache_dir', action="store", type=str, default="pred_lbl_ind_map")
parser.add_argument('--lbl_cache_dir', action="store", type=str, default="gt_lbl_ind_map_l2v")
parser.add_argument('--eval_h', action="store", type=int, default="256")
parser.add_argument('--eval_w', action="store", type=int, default="256")
parser.add_argument('--model_path', action="store", type=str, default=None)
parser.add_argument('--n_class', action="store", type=int, default="32")
parser.add_argument('--mean', action="store", type=bool, default="True")
parser.add_argument('--verbose', action="store", type=bool, default="False")
parser.add_argument('--output_name', action="store", type=str, default="")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = fcn.FCN8sAtOnce(n_class=args.n_class).to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

fns = os.path.join(args.exp_name)
fs = glob.glob(fns)

print("L2V experiment, Looking files from:", fns, ", with model:", args.model_path)

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

if not os.path.exists(args.pred_cache_dir):
    os.makedirs(args.pred_cache_dir)
if not os.path.exists(args.lbl_cache_dir):
    os.makedirs(args.lbl_cache_dir)

def cal_acc(data_list, classes, names):
    print(classes, names)
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

img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

day = ['068', '066', '069', '065', '067']
night = ['076', '073', '077', '074', '075']
sunset = ['028', '025', '029', '027', '026']
snow = ['040', '042', '041', '043']
rain = ['063', '064']

for f in (fs):

    im = Image.open(f).convert('RGB')
    im = np.array(im.resize((args.eval_w, args.eval_h), Image.LANCZOS), dtype=np.uint8)
    im = im[:, :, ::-1]  # RGB -> BGR
    im = im.astype(np.float64)
    if args.mean:
        im -= img_mean
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().cuda().unsqueeze(0).to(device)
    with torch.no_grad():
        lbl = model(im)[0]

    lbl = torch.argmax(lbl[0], 0)
    # save the map
    im = Image.fromarray(np.uint8(lbl.cpu().numpy()))
    save_dir = os.path.join(args.pred_cache_dir, os.path.basename(f))
    im.save(save_dir)

    vid_ind = os.path.basename(f).split("_")[0]
    img_ind = os.path.basename(f).split("_")[1]
    target_path = os.path.join(args.target_path, vid_ind, vid_ind + "_" + img_ind + ".png")

    data_list.append((save_dir, target_path))
    if vid_ind in day:
        day_list.append((save_dir, target_path))
    elif vid_ind in night:
        night_list.append((save_dir, target_path))
    elif vid_ind in sunset:
        sunset_list.append((save_dir, target_path))
    elif vid_ind in snow:
        snow_list.append((save_dir, target_path))
    elif vid_ind in rain:
        rain_list.append((save_dir, target_path))
    else:
        raise ValueError

print("All:")
cal_acc(data_list, len(class_names), class_names)
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
