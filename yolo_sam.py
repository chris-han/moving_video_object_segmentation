import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from multiprocessing import Pool

from ultralytics import YOLO
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

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


def optimized_mask2img(mask):
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    items = mask.shape[0]
    rows = mask.shape[1]
    cols = mask.shape[2]
    image = np.zeros((items, rows, cols, 3), dtype=np.uint8)
    #print('palette: ', palette[1], palette[1][0])
    image[:, :, :, 0] = mask * palette[1][0]
    image[:, :, :, 1] = mask * palette[1][1]
    image[:, :, :, 2] = mask * palette[1][2]
    return image


def optimized_show_mask(masks):
    print('pre squeeze shape: ', masks.shape)
    masks = np.squeeze(masks, axis = 1)
    print('masks: ', masks.shape)
    separate_rgb_masks = optimized_mask2img(masks)
    print(separate_rgb_masks.shape)
    combined_mask = np.sum(separate_rgb_masks, axis = 0)
    return combined_mask

def process_mask(dim):
    mask = masks[dim, :, :]
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours


if __name__ == '__main__':
    ## Load YOLO
    ## yolo model
    yolo_model = YOLO('yolov8x.pt').to(device)

    ## sam model
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=os.path.join(sam_checkpoints, vit_h))
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    
    
    
    ## Load video
    video_path  = os.path.join(os.getcwd(), 'src','output1024_crop.mp4')
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)
    
    if cap.isOpened() == False:
        print("Error in loading the video")
    
    i = 0
    
    # # Get the video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # # Define the output video path
    output_path_contours = os.path.join(os.getcwd(), 'output', 'classroom_c.mp4')
    output_path_segmentation = os.path.join(os.getcwd(), 'output', 'classroom_s.mp4')

    # # Create a VideoWriter object to save the processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_c = cv2.VideoWriter(output_path_contours, fourcc, fps, (frame_width, frame_height))
    out_s = cv2.VideoWriter(output_path_segmentation, fourcc, fps, (frame_width, frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
           # frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #frame2 = cv2.cvtColor((cv2.GaussianBlur(frame2, (3, 3), 0)), cv2.COLOR_GRAY2RGB)
            masks = process_frame(frame, yolo_model, predictor)
            #dispaly frame and colour mask in same window
            
         
            if masks is not None:        
                frame = ((frame/np.max(frame))*255).astype(np.uint8)
                colour_mask = optimized_show_mask(masks.detach().cpu().numpy())
       
                colour_mask = cv2.addWeighted(colour_mask.astype(np.uint8), 0.3, frame, 0.7, 0, dtype=cv2.CV_8U)#colour_mask.astype(np.uint8))
                
                #-----------for contours -------
                masks = np.squeeze(masks.detach().cpu().numpy(), axis = 1).astype(np.uint8)
                #print('masks shape: ', masks.shape, masks.shape[0], np.unique(masks))
                for dim in range(masks.shape[0]):
                    #print('in shape: ', masks[dim, :, :].shape)
                    contours, hierarchy = cv2.findContours(image = masks[dim, :, :], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(image = frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
          
                       
                # Write the combined frame to the output video
                out_c.write(frame)
                out_s.write(colour_mask)
                cv2.imshow('frame', frame)
                cv2.imshow('frame', colour_mask)    
                # add cv2.waitKey()   


            else:
                out_c.write(frame)
                out_s.write(frame)
                cv2.imshow('frame', frame)
                
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
            i = i + 1
        ## save frame and make video
        except:
            i = i + 1
            out_c.release()
            out_s.release()
            break
       

    cap.release()
    cv2.destroyAllWindows()