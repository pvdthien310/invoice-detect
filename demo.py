import os
import cv2
import re
import pandas as pd
from modules import Preprocess, Detection, OCR, Retrieval, Correction
from tool.utils import natural_keys, visualize
import time
import matplotlib.pyplot as plt

def find_highest_score_each_class(labels, probs):
        best_score = [0] * (len(class_mapping.keys())-1)
        best_idx = [-1] * (len(class_mapping.keys())-1)
        for i, (label, prob) in enumerate(zip(labels, probs)):
            label_idx = class_mapping[label]
            if label_idx != class_mapping["NONE"]:
                if prob > best_score[label_idx]:
                    best_score[label_idx] = prob
                    best_idx[label_idx] = i
        return best_idx

def find_total_cost_value(total_cost_idx, boxes):
    total_cost_box = boxes[total_cost_idx]
    x1,y1 = total_cost_box[0]
    for i in range(total_cost_idx+1, len(boxes)):
        x1_,y1_ = boxes[i][0]

        if abs(x1-x1_) < 2:
          return i-1


def extract_timestamp(text):
    x = re.findall(r'\d{2}:\d{2}|\d{2}:\d{2}:\d{2}|\d{2}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{2}|\d+/\d+/\d+', text)
    return ' '.join(x)

img_id = "z4083144499175"
class_mapping = {"SELLER":0, "ADDRESS":1, "TIMESTAMP":2, "TOTAL_COST":3, "NONE":4}
idx_mapping = {0:"SELLER", 1:"ADDRESS", 2:"TIMESTAMP", 3:"TOTAL_COST", 4:"NONE"}

det_weight = "/weights/PANNet_best_map.pth"
ocr_weight = "/weights/transformerocr.pth"

img = cv2.imread(f"/{img_id}.jpg")

plt.imshow(img)
plt.show()

det_model = Detection(weight_path=det_weight)
ocr_model = OCR(weight_path=ocr_weight)
preproc = Preprocess(
    det_model=det_model,
    ocr_model=ocr_model,
    find_best_rotation=False)
# retrieval = Retrieval(class_mapping, mode = 'all')
correction = Correction()

img1 = preproc(img)

plt.imshow(img1)
plt.show()

boxes, img2  = det_model(
    img1,
    crop_region=True,                               #Crop detected regions for OCR
    return_result=True,                             # Return plotted result
    output_path=f"/results/{img_id}"   #Path to save cropped regions
)

plt.imshow(img2)
plt.show()

img_paths=os.listdir(f"/results/{img_id}/crops") # Cropped regions
img_paths.sort(key=natural_keys)
img_paths = [os.path.join(f"/results/{img_id}/crops", i) for i in img_paths]

texts, probs = ocr_model.predict_folder(img_paths, return_probs=True) # OCR
# texts = correction(texts)   # Word correction

for i in texts:
    print(i)