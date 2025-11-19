import os
import tqdm
import torch
import argparse
import numpy as np
import cv2

from models.architecture import SpikeFPN_GAD
from utils.spikefpn_config import spikefpn_cfg
from utils import create_labels
from utils.tools import ori_target_frame_collate
from utils.tools import get_box_score, resized_box_to_original
from utils.datasets import Resize_frame, GAD_SBT
from utils.gad_evaluate import coco_eval


def parse_args():
    parser = argparse.ArgumentParser(description="GAD Object Detection")
    
    parser.add_argument("--data_path", type=str, default="/dvs_dataset/GAD")
    parser.add_argument("--device", default="0", help="CUDA device, i.e. 0 or cpu")
    
    # Basic setting
    parser.add_argument("-size", "--img_size", default=256, type=int, help="img_size")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    
    # Model setting
    parser.add_argument("--weight", default="./weights/gad_weight.pth", type=str)
    parser.add_argument("--conf_thresh", default=0.3, type=float, help="NMS threshold")
    parser.add_argument("--nms_thresh", default=0.5, type=float, help="NMS threshold")
    parser.add_argument("-t", "--time_steps", default=3, type=int, help="SpikeFPN time steps") 
    parser.add_argument("-tf", "--time_per_frame", default=20, type=int, help="SpikeFPN time per frame")
    parser.add_argument("-fs", "--frame_per_stack", default=3, type=int, help="SpikeFPN frame per stack")

    return parser.parse_args()


def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split(".")[:-1] + ["conv1", "weight"]
    elif is_b:
        this_str = this_str.split(".")[:-1] + ["snn_optimal", "b"]
    elif is_cell:
        this_str = this_str.split(".")[:3]
    else:
        this_str = this_str.split(".")
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append(f"[{value:s}]")
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append("." + value)
    return "".join(new_index)


if __name__ == "__main__":
    args = parse_args()

    # Set device
    if args.device != "cpu":
        print(f"Using CUDA:{args.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Using weight: {args.weight}.")

    # Define dataset and data loader
    test_dataset = GAD_SBT(
        root_dir = args.data_path, 
        object_classes = "all", 
        height = 240, 
        width = 304, 
        mode = "test", 
        ms_per_frame = args.time_per_frame, 
        frame_per_sequence = args.frame_per_stack, 
        T = args.time_steps, 
        transform = Resize_frame(args.img_size), 
        sbt_method = "before"
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        shuffle = False,
        batch_size = args.batch_size, 
        collate_fn = ori_target_frame_collate,
        num_workers = 0,
        pin_memory = True
    )
    classes_name = test_dataset.object_classes
    num_classes = len(classes_name)

    # Build model
    model = SpikeFPN_GAD(
        device = device, 
        input_size = args.img_size, 
        num_classes = num_classes, 
        cfg = spikefpn_cfg, 
        center_sample = False,
        time_steps = args.time_steps,
        init_channels = args.frame_per_stack,
        args = args
    )
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if "_ops" in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval(f"model.{key:s}.mem")
            mem_keys.append(key)
        except:
            print(key)
    print("mem", mem_keys)
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    gt_label_list = []
    pred_label_list = []
    with torch.no_grad():
        img_list = []
        for id_, data in enumerate(tqdm.tqdm(test_dataloader)):
            for key in mem_keys:
                exec(f"model.{key:s}.mem=None")
            image, targets, original_label, original_frame, file = data
            

            for label in original_label:
                gt_label_list.append(label)
                
            targets = [label.tolist() for label in targets]
            size = np.array([[
                image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]
            ]])
            targets = create_labels.gt_creator(
                img_size = args.img_size, 
                strides = model.stride, 
                label_lists = targets, 
                anchor_size = anchor_size, 
                multi_anchor = True,
                center_sample = False
            )

            image = image.float().to(device)

            """
            for B_index in range(len(targets)):
                np_image = image[B_index][0] + image[B_index][1] + image[B_index][2]
                np_image = np_image.cpu().numpy()
                np_image = np.transpose(np_image, (1,2,0))
                np_image = np.sign(np_image)  
                np_image = ((np_image * 127.5) + 127.5).astype(np.uint8)
                cv2.imwrite(f"./test_output/gad_test_image_B{B_index}_sum.png", np_image)

                                
                for S_index in range(3):
                    np_image = image[B_index][S_index].cpu().numpy()
                    np_image = np.transpose(np_image, (1,2,0))
                    np_image = ((np_image * 127.5) + 127.5).astype(np.uint8)
                    cv2.imwrite(f"./test_output/gad_test_image_B{B_index}_S{S_index}.png", np_image)
                    #print(torch.max(image[B_index][S_index]), torch.min(image[B_index][S_index]))
                
            break
            """
            conf_pred, cls_pred, reg_pred, box_pred = model(image)

            bboxes, scores, cls_inds = get_box_score(
                conf_pred, cls_pred, box_pred, 
                num_classes = num_classes, 
                conf_thresh = args.conf_thresh, 
                nms_thresh = args.nms_thresh
            )

            """
            bboxes = [box * size for box in bboxes]
            bboxes = [
                resized_box_to_original(box, args.img_size, test_dataset.height, test_dataset.width)
                for box in bboxes
            ]
            """
            #size = [[304, 240, 304, 240]]
            #bboxes = [box * size for box in bboxes]

            
            for i in range(len(bboxes)):

                #bboxes[i] = bboxes[i].cpu().numpy()
                pred_label = []
                np_image = image[i][0].cpu().numpy() + image[i][1].cpu().numpy() + image[i][2].cpu().numpy()
                np_image = np.transpose(np_image, (1,2,0))
                np_image = np.sign(np_image)  
                np_image = ((np_image * 127.5) + 127.5).astype(np.uint8)
                np_image = cv2.resize(np_image, (304, 240))
                #gt_bboxes = gt_label_list[i-(32*(id_ + 1))]
                gt_bboxes = gt_label_list[id_ * args.batch_size + i]
                
                for gt_bbox in gt_bboxes:
                    #resized_box = resized_box_to_original(gt_bbox[1:], args.img_size, test_dataset.height, test_dataset.width)
                    
                    x1 = int(gt_bbox[1])
                    y1 = int(gt_bbox[2])
                    x2 = int(gt_bbox[1] + gt_bbox[3])
                    y2 = int(gt_bbox[2] + gt_bbox[4])
                    #print(gt_bbox)
                    cv2.rectangle(np_image, (x1, y1), (x2, y2), (255,0,0), 2)
                for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                    x1 = float(box[0]*304)
                    y1 = float(box[1]*240)
                    x2 = float(box[2]*304)
                    y2 = float(box[3]*240)

                    if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and score > 0.9):
                        cv2.rectangle(np_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(score) # object_score * class_score
                    A = {"image_id": id_ * args.batch_size + i, "category_id": cls_ind, "bbox": bbox, "score": score} # COCO JSON format
                    pred_label.append(A)
                pred_label_list.append(pred_label)
                cv2.imwrite(f"./test_output/pred_event_batch_{id_}_image_{i}.png", np_image)
        
    map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=240, width=304, labelmap=classes_name)
    print(f"test mAP(0.5:0.95):{map50_95}, mAP(0.5):{map50}")
