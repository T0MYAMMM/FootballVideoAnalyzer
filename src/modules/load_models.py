import yaml
import torch

from ultralytics import YOLO
from modules.model.cls_hrnet import get_cls_net
from modules.model.cls_hrnet_l import get_cls_net as get_cls_net_l

device = "cuda:0"

def load_model_obj(PATH):
    # Load the YOLOv8 objects detection model 
    model_obj = YOLO(PATH)
    return model_obj

def load_model_kp(PATH_CONFIG, PATH_WEIGHTS):
    # Load the Keypoints config and model
    cfg = yaml.safe_load(open(PATH_CONFIG, 'r'))
    loaded_state = torch.load(PATH_WEIGHTS, map_location = device)
    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(loaded_state)
    model_kp.to(device)
    model_kp.eval()
    return model_kp

def load_model_line(PATH_CONFIG, PATH_WEIGHTS):
    # Load the Line segmenatiton config and model
    cfg_l = yaml.safe_load(open(PATH_CONFIG, 'r'))
    loaded_state_l = torch.load(PATH_WEIGHTS, map_location = device)
    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(loaded_state_l)
    model_line.to(device)
    model_line.eval()
    return model_line