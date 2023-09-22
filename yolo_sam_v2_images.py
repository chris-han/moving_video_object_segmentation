import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from multiprocessing import Pool

from ultralytics import YOLO
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


__all__ = ['predict_segmentation']


## checkpoints for sam
sam_checkpoints = "checkpoints"
vit_h = "sam_vit_h_4b8939.pth"
vit_b = "sam_vit_b_01ec64.pth"
vit_l = "sam_vit_l_0b3195.pth"

## check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_frame(frame, yolo_model, predictor):
    
    results = yolo_model(frame, conf=0.25, classes=[0])
    
    ## Process results
    for result in results:
        boxes = result.boxes
        
    bbox = boxes.xyxy
    #confidences = boxes.conf
    #classes = boxes.cls 
    #predictor = SamPredictor(sam)
    predictor.set_image(frame)
    
    input_boxes = bbox.to(predictor.device)
    if len(input_boxes) == 0:
        return None
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    
    masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
    )
    
    return masks



def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    
    ## convert result_masks to torch tensor
    result_masks = torch.from_numpy(np.array(result_masks)).to(device)
    return result_masks


def process_frame2(frame, yolo=None, sam_predictor=None):
    # detect objects
    if yolo:
        # detections = grounding_dino_model.predict_with_classes(
        #     image=frame,
        #     classes= enhance_class_name(class_names=CLASSES),
        #     box_threshold=BOX_TRESHOLD,
        #     text_threshold=TEXT_TRESHOLD
        # )
        # if detections is None:
        #     return None
        # masks = segment(
        #     sam_predictor=sam_predictor,
        #     image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        #     xyxy=detections.xyxy
        # )
    
        detections = yolo(frame, conf=0.25, classes=[0])
        for result in detections:
            boxes = result.boxes
        # convert detections to masks
        if len(boxes) == 0:
            return None
        masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            xyxy= (boxes.xyxy).detach().cpu().numpy()
        )
    return masks #detections.mask

def predict_segmentation(source_image):
    yolo_model = YOLO('yolov8x.pt').to(device)
    
    ## sam model
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=os.path.join(sam_checkpoints, vit_h))
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    
    ## predict segmentation
    masks = process_frame2(source_image, yolo_model, predictor)
    
    ## take the maximum value from all the mask (torch tensor)
    combined_mask = torch.max(masks, dim=0)[0]
    
    ## convert the [True, False] to [1, 0] (torch tensor)
    combined_mask = combined_mask.type(torch.float32)
    
    return combined_mask
    
