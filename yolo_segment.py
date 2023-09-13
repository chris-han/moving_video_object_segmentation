# import os

# import cv2 
# from ultralytics import YOLO
# import torch


# ## check for device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def main(video_path):
#     ## yolo model
#     yolo_model = YOLO('yolov8x-seg.pt').to(device)
    
#     ## perform segmentation
#     output_video = os.path.join("output", "classroom_yolo.mp4")
    
#     results = yolo_model.predict(source=video_path, device=device, segment=True, segment_conf=0.5, classes=[0], save=True)
    



# if __name__ == "__main__":
#     video_path = os.path.join("src", "classroom.mp4")