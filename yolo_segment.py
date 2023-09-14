# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image

# import torchvision.transforms as transforms
# import torch
# from ultralytics import YOLO
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ## checkpoints for sam
# sam_checkpoints = "checkpoints"
# vit_h = "sam_vit_h_4b8939.pth"
# vit_b = "sam_vit_b_01ec64.pth"
# vit_l = "sam_vit_l_0b3195.pth"

# def optimized_mask2img(mask):
#     palette = {
#         0: (0, 0, 0),
#         1: (255, 0, 0),
#         2: (0, 255, 0),
#         3: (0, 0, 255),
#         4: (0, 255, 255),
#     }
#     items = mask.shape[0]
#     rows = mask.shape[1]
#     cols = mask.shape[2]
#     image = np.zeros((items, rows, cols, 3), dtype=np.uint8)
#     image[:, :, :, 0] = mask * palette[1][0]
#     image[:, :, :, 1] = mask * palette[1][1]
#     image[:, :, :, 2] = mask * palette[1][2]
#     return image

# def optimized_show_mask(masks):
#     masks = np.squeeze(masks, axis = 1)
#     separate_rgb_masks = optimized_mask2img(masks)
#     combined_mask = np.sum(separate_rgb_masks, axis = 0)
#     return combined_mask


# def visualize_batch(img_batch):
#     """
#     Input: 
#         img_batch: a batch of images as a PyTorch tensor
#     Details:
#         This function visualizes a batch of images using Matplotlib.
#     """
#     num_images = img_batch.size(0)
    
#     # Create a grid for visualization
#     grid_size = int(num_images**0.5)
    
#     fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
#     axs = axs.ravel()
    
#     for i in range(min(num_images, grid_size * grid_size)):  # Ensure we don't go out of bounds
#         image = img_batch[i, 0].permute(1, 2, 0)  # Select the image, remove extra dimension and convert to (H, W, C)
#         axs[i].imshow(image.cpu().numpy(), cmap='gray')
#         axs[i].axis('off')
    
#     plt.show()

# # def augment_image(img):
# #     """
# #     Input: 
# #         img: input image
# #     Output:
# #         img_batch: a batch of augmented images
    
# #     Details:
# #         This function takes in an image and performs multiple data augmentations using pytorch. 
# #         The augmentation techniques used are changing saturation, brightness, contrast, hue, sharpness, etc.
# #     """
# #     augmented_images = [
# #         img,
# #         F.adjust_saturation(img, 2),
# #         F.adjust_saturation(img, 0.5),
# #         F.adjust_contrast(img, 2),
# #         F.adjust_contrast(img, 0.5),
# #         F.adjust_sharpness(img, 2),
# #         F.adjust_sharpness(img, 0.5)
# #     ]
    
# #     img_batch = torch.stack(augmented_images)
    
#     # return img_batch
    
# def augment_image(img):
#     ## set the same seed for all transformations
#     torch.manual_seed(5)
#     augmented_images = [
#         img,
#         #sharp the image, then gaussian blur
#         cv2.GaussianBlur(cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])), (7, 7), 0),
#         ## invert the color of the image
#         cv2.bitwise_not(img),
#         cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
#         # img,
#         # img,
#         # img,
#         #transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))(img),
#     ]
    
#     #augmented_images = [np.array(img.permute(2,1,0)) for img in augmented_images]
#     return augmented_images

# def display_image_cv2(img):
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image_path = os.path.join(os.getcwd(), 'src','multi.jpg')
#     img2 = cv2.imread(image_path)


#     ## resize image to 416x416
#     #img = cv2.resize(img, (1024, 2336), interpolation=cv2.INTER_LINEAR)
#     ## convert image to tensor
#     #img = torch.from_numpy(img2).permute(2, 1, 0).float()

#     image_list = augment_image(img2)
#    # display_image_cv2(image_list[2].astype(np.uint8))
    
#     yolo_model = YOLO('yolov8x.pt').to(device)
#     resutls = yolo_model(image_list, conf=0.25, classes=[0])
#     #result = yolo_model(img, conf=0.25, classes=[0])
    
#     bbox_list = []
#     for result in resutls:
#         bbox_list.append(result.boxes.xyxy)
    
#     model_type = "vit_l"
#     sam = sam_model_registry[model_type](checkpoint=os.path.join(sam_checkpoints, vit_l))
#     sam = sam.to(device)
#     predictor = SamPredictor(sam)
    
#     frame = image_list[1]
#     frame = ((frame/np.max(frame))*255).astype(np.uint8)
#     predictor.set_image(frame)
    
#     ## take the average of the boxes all the boxes list
#     #bbox_average = torch.mean(torch.stack(bbox_list), dim = 0)
    
    
    
  
    
#     input_boxes = bbox_list[1].to(predictor.device)
#     transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    
#     masks, _, _ = predictor.predict_torch(
#     point_coords=None,
#     point_labels=None,
#     boxes=transformed_boxes,
#     multimask_output=False,
#     )
    
#     ## apply text prompt to SAM model to get the masks
#     #masks = sam(prompt = "a photo of a person", image = frame)
    
#     colour_mask = optimized_show_mask(masks.detach().cpu().numpy())

#     #frame = frame + colour_mask*0.3
    
    
#     colour_mask = cv2.addWeighted(colour_mask.astype(np.uint8), 0.3, frame, 0.7, 0, colour_mask.astype(np.uint8))
#     ## add the bounding box as well to the image
#     for dim in range(masks.shape[0]):
#         cv2.rectangle(colour_mask, (int(bbox_list[1][dim][0]), int(bbox_list[1][dim][1])), (int(bbox_list[1][dim][2]), int(bbox_list[1][dim][3])), (0, 0, 255), 2)
    
    
    
#     #-----------for contours -------
#     # masks = np.squeeze(masks.detach().cpu().numpy(), axis = 1).astype(np.uint8)
#     # print('masks shape: ', masks.shape, masks.shape[0], np.unique(masks))
#     # for dim in range(masks.shape[0]):
#     #     print('in shape: ', masks[dim, :, :].shape)
#     #     contours, hierarchy = cv2.findContours(image = masks[dim, :, :], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
#     #     cv2.drawContours(image = img2, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
#     #     ## add the bounding box as well to the image
#     #     cv2.rectangle(img2, (int(bbox_list[1][dim][0]), int(bbox_list[1][dim][1])), (int(bbox_list[1][dim][2]), int(bbox_list[1][dim][3])), (0, 0, 255), 2)
    
#     # #cv2.imshow('frame', colour_mask)
#     # #display_image_cv2(colour_mask.astype(np.uint8))
#     display_image_cv2(colour_mask)
    
 
#     #for image in image_list:
#        # display_image_cv2(image.astype(np.uint8))
    

    
