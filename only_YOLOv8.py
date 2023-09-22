import os
import numpy as np
import cv2

from ultralytics import YOLO
import torch


__all__ = ['predict_segmentation_yolo']



## check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_frame(frame, yolo_model):
    results = yolo_model(frame, conf=0.25, classes=[0], retina_masks=True)
    for r in results:
        masks = r.masks
    return masks



def predict_segmentation_yolo(source_image):
    yolo_model = YOLO('yolov8x-seg.pt').to(device)
    
    ## predict segmentation
    masks = process_frame(source_image, yolo_model)
    combined_mask = masks.data
    
    ## combnine them to one mask
    combined_mask = torch.max(combined_mask, dim=0)[0]
  
    return combined_mask
    
