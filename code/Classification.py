"""
Create training and test folders and divide the datasets to age groups (Baby, Child, Young, Adult, Senior)
"""


import pandas as pd
import shutil
import os
import datetime
import math
import time
import cv2
from PIL import Image
import random
import sys

# dataset = pd.read_csv("selfie_dataset.txt", delimiter=" ")

# rows = dataset.iloc[:, [0, 4, 5, 6, 7, 8, 9]].values


# for row in rows:
#     print(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
#     if row[1] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'baby/'+row[0]+'.jpg')
#     if row[2] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'child/'+row[0]+'.jpg')
#     if row[3] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'teenager/'+row[0]+'.jpg')
#     if row[4] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'youth/'+row[0]+'.jpg')
#     if row[5] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'middle_age/'+row[0]+'.jpg')
#     if row[6] == 1:
#         if os.path.isfile('images/' + row[0] + '.jpg'):
#             shutil.move('images/'+row[0]+'.jpg', 'senior/'+row[0]+'.jpg')


# For Wiki Images Classification..

# path = 'e:/Selfie-dataset/test_dataset1/part3'
# # folders = os.listdir('../datasets/wiki_crop')
# missingImages = []
# # for folder in folders:
# gray = None
# for file in os.listdir(path):
#     try:
#         filename = file.split('_')
#         age = int(filename[0])
#         filename = str(filename[3])
#         print(file)
#         print(age,filename)
#         # time.sleep(5000)
#         if 0 <= age <= 3:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/baby/' + str(filename.split('.')[0]) + '.png',gray)
#             # img = Image.open(path + '/' + str(file)).convert('LA')
#             # img.save('c:/Facial-based-age-estimator/The Project/datasets/ages/training/baby/' + str(filename.split('.')[0]) + '.png')
#             # shutil.copy(path + '/' + str(file), '../datasets/ages/baby/' + str(filename.split('.')[0]) + '.png')
#         if 6 <= age <= 16:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/child/' + str(filename.split('.')[0]) + '.png',gray)
#         if 17 <= age <= 20:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/teenager/' + str(filename.split('.')[0]) + '.png',gray)
#         if 21 <= age <= 32:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/youth/' + str(filename.split('.')[0]) + '.png',gray)
#         if 33 <= age <= 65:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/middle_age/' + str(filename.split('.')[0]) + '.png',gray)
#         if 66 <= age:
#             gray = cv2.cvtColor(cv2.imread(path + '/' + str(file)), cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('c:/Facial-based-age-estimator/The Project/datasets/ages/training/senior/' + str(filename.split('.')[0]) + '.png',gray)
#     except Exception as e:
#         print(e)
#         # missingImages.append([folder, file])
# print(len(missingImages))


for folder in os.listdir('../datasets/ages/training'):
    files_len = len(os.listdir('..//datasets/ages/training/' + folder))
    if not os.path.exists('..//datasets/ages/test/' + folder):
        os.mkdir('The Project/datasets/ages/test/' + folder)
    for file in random.sample(os.listdir('..//datasets/ages/training/' + folder), int(files_len * 0.20)):
        shutil.move('..//datasets/ages/training/' + folder + '/' + str(file),
                    '..//datasets/ages/test/' + folder + '/' + file)
    sys.stdout.write('\r Moveing')
    sys.stdout.flush()
