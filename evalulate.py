import os

import cv2
import torch
import torch.nn as nn


from yolo_sam_v2_images import predict_segmentation_v2, predict_segmentation_v1
from only_YOLOv8 import predict_segmentation_yolo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(input, target, smooth=1.):
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))
    



if __name__ == '__main__':
    main_data_path = "D:\dataset\DAVIS"
    selected_videos = [ 'parkour', 'kid-football']
    
    
    mean_dice_score = 0
    
    for video in selected_videos:
        images_path = os.path.join(main_data_path, 'JPEGImages', 'Full-Resolution')
        segmentation_path = os.path.join(main_data_path, 'Annotations_unsupervised', 'Full-Resolution')
        full_images_path = os.path.join(images_path, video)
        full_segmentation_path = os.path.join(segmentation_path, video)
        
        mean_dice_Score_per_video = 0
        
        for image, segmentation in zip(os.listdir(full_images_path), os.listdir(full_segmentation_path)):
            image_path = os.path.join(full_images_path, image)
            segmentation_path = os.path.join(full_segmentation_path, segmentation)
            
            image = cv2.imread(image_path)
            segmentation = cv2.imread(segmentation_path)
            
            
            predicted_segmentation = predict_segmentation_yolo(image)
            target_segmentation_torch = torch.from_numpy(segmentation).to(DEVICE)
            
            ## compress three channels into one, by taking the max value and assign 1 to the max value otherwise 0
            target_segmentation_torch = torch.max(target_segmentation_torch, dim=2)[0]
            ## assign max value to 1
            target_segmentation_torch[target_segmentation_torch > 0] = 1
            
            dice = dice_score(predicted_segmentation, target_segmentation_torch.unsqueeze(0))
            
            mean_dice_Score_per_video += dice.detach().cpu().numpy()
        
        mean_dice_score += mean_dice_Score_per_video / len(os.listdir(full_images_path))
    
          
    print('mean dice: ', mean_dice_score / len(selected_videos))
    
    