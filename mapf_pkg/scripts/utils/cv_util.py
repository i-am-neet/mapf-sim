import math
import numpy as np
import cv2
import torch

def tensor_to_cv(T, W, H):
    return T.reshape(H, W).cpu().numpy().astype('uint8')

def vconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
      # take minimum width
    w_min = min(img.shape[1] 
                for img in img_list)
      
    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation = interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)
  
def hconcat_resize(img_list, 
                   interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv2.hconcat(im_list_resize)

def concat_vh(list_2d):
    
      # return final image
    return cv2.vconcat([cv2.hconcat(list_h) 
                        for list_h in list_2d])

def concat_tile_resize(list_2d, 
                       interpolation = cv2.INTER_CUBIC,
                       border=True,
                       colorful=True,
                       text=None):

    if border:
        for _i in range(0, len(list_2d), 1):
            for _j in range(0, len(list_2d[0]), 1):
                list_2d[_i][_j] = cv2.cvtColor(list_2d[_i][_j], cv2.COLOR_GRAY2RGB) if colorful else list_2d[_i][_j]
                if text is not None:
                    if len(text) == len(list_2d) and len(text[0]) == len(list_2d[0]):
                        cv2.putText(list_2d[_i][_j], text[_i][_j], (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        raise ValueError("text's shape must equal to images' list.")

                list_2d[_i][_j] = cv2.copyMakeBorder(list_2d[_i][_j], 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(50, 168, 82))

    img_list_v = [hconcat_resize(list_h, 
                                 interpolation = cv2.INTER_CUBIC) 
                  for list_h in list_2d]
    
    # img_list_v = [hconcat_resize(cv2.copyMakeBorder(list_h, 5, 5, 5, 5, cv2.BORDER_CONSTANT), 
    #                              interpolation = cv2.INTER_CUBIC) 
    #               for list_h in list_2d]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)

