"""
Create training and test folders and divide the datasets to age groups (Baby, Child, Young, Adult, Senior)
"""


import pandas as pd
import shutil
import os
import sys
import datetime
import math
import time
import cv2
import os
import random
from PIL import Image
from FacialBasedAgeEstimator import FacialBasedAgeEstimator
from matplotlib import pyplot
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import time


if __name__ == '__main__':
    path = 'e:/imdb_crop'
    # path = 'E:/Selfie-dataset/crop_part1'
    total_files = 0
    for folder in os.listdir(path):
        if folder != 'imdb.mat':
            total_files += len(os.listdir(path+'/'+folder))
    # total_files = len(os.listdir(path))
    face_cascade = cv2.CascadeClassifier('code/haarcascade_frontalface_default.xml')
    fbae = FacialBasedAgeEstimator(face_cascade, 1.15)

    missingImages = 0
    current_files = 0
    missed_files = 0
    num = 0

    if not os.path.exists('datasets/ages/training/baby'):
        os.mkdir('datasets/ages/training/baby')

    if not os.path.exists('datasets/ages/training/child'):
        os.mkdir('datasets/ages/training/child')

    if not os.path.exists('datasets/ages/training/young'):
        os.mkdir('datasets/ages/training/young')

    if not os.path.exists('datasets/ages/training/adult'):
        os.mkdir('datasets/ages/training/adult')

    if not os.path.exists('datasets/ages/training/senior'):
        os.mkdir('datasets/ages/training/senior')

    import scipy.io
    mat = scipy.io.loadmat('e:/imdb_crop/imdb.mat')
    instances = mat['imdb'][0][0][0].shape[1]
    
    columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]
    
    import pandas as pd
    df = pd.DataFrame(index = range(0,instances), columns = columns)
    
    for i in mat:
        if i == "imdb":
            current_array = mat[i][0][0]
            #print(current_array)
            for j in range(len(current_array)):
                try:
                    df[columns[j]] = pd.DataFrame(current_array[j][0])
                except Exception as e:
                    print(e)
    df = df[df['face_score'] >= 3]
    rows = df.iloc[:,[2]].values
    for file in rows:
        try:
            file = file[0]
            folderFileName = str(file[0]).split('/')
            folder = folderFileName[0]
            file = str(file[0]).split('/')[1]
            filename = file.split('_')
            date_birth = filename[2].split('-')
            date_taket = filename[3].split('.')[0]
            if int(date_birth[2]) == 00:
                date_birth[2] = int(date_birth[2]) + 1
            if int(date_birth[1]) >= 12 and int(date_birth[1]) <= 12:
                date_birth[1], date_birth[2] = date_birth[2], date_birth[1]
            date_brith = datetime.datetime(int(date_birth[0]), 6, 15)
            date_taket = datetime.datetime(int(date_taket), 6, 15)
            age = math.ceil((date_taket - date_brith).days / 365)

            file_path = path + '/' + str(folder) + '/' + file 

            if 0 <= age <= 4:

                filepathsave = 'datasets/ages/training/baby/'+ file + '.png'

                if os.path.isfile(file_path) and ( not os.path.exists(filepathsave)) and len(os.listdir('datasets/ages/training/baby')) <= 10000:
                    shutil.copy(file_path,filepathsave)
                    current_files +=1
                    # img = cv2.imread(file_path)
                    # clip_img = fbae.predict_image(img)
                    # if type(clip_img) != type(0):
                    #     gray = cv2.cvtColor(clip_img, cv2.COLOR_BGR2GRAY)
                    #     cv2.imwrite(filepathsave,gray)
                    #     current_files +=1
            if 5 <= age <= 17:

                filepathsave = 'datasets/ages/training/child/'+ file + '.png'

                if os.path.isfile(file_path) and ( not os.path.exists(filepathsave)) and len(os.listdir('datasets/ages/training/child')) <= 10000:
                    shutil.copy(file_path,filepathsave)
                    current_files +=1
                    # img = cv2.imread(file_path)
                    # clip_img = fbae.predict_image(img)
                    # if type(clip_img) != type(0):
                    #     gray = cv2.cvtColor(clip_img, cv2.COLOR_BGR2GRAY)
                    #     cv2.imwrite(filepathsave,gray)
                    #     current_files +=1
                    
            if 18 <= age <= 32:

                filepathsave = 'datasets/ages/training/young/'+ file + '.png'

                if os.path.isfile(file_path) and ( not os.path.exists(filepathsave)) and len(os.listdir('datasets/ages/training/young')) <= 10000:
                    shutil.copy(file_path,filepathsave)
                    current_files +=1
                    # img = cv2.imread(file_path)
                    # clip_img = fbae.predict_image(img)
                    # if type(clip_img) != type(0):
                    #     gray = cv2.cvtColor(clip_img, cv2.COLOR_BGR2GRAY)
                    #     cv2.imwrite(filepathsave,gray)
                    #     current_files +=1

            if 33 <= age <= 55:

                filepathsave = 'datasets/ages/training/adult/'+ file + '.png'

                if os.path.isfile(file_path) and ( not os.path.exists(filepathsave)) and len(os.listdir('datasets/ages/training/adult')) <= 10000:
                    shutil.copy(file_path,filepathsave)
                    current_files +=1
                    # img = cv2.imread(file_path)
                    # clip_img = fbae.predict_image(img)
                    # if type(clip_img) != type(0):
                    #     gray = cv2.cvtColor(clip_img, cv2.COLOR_BGR2GRAY)
                    #     cv2.imwrite(filepathsave,gray)
                    #     current_files +=1

            if 56 <= age:

                filepathsave = 'datasets/ages/training/senior/'+ file + '.png'

                if os.path.isfile(file_path) and ( not os.path.exists(filepathsave))  and len(os.listdir('datasets/ages/training/senior')) <= 10000:
                    shutil.copy(file_path,filepathsave)
                    current_files +=1
                    # img = cv2.imread(file_path)
                    # clip_img = fbae.predict_image(img)
                    # if type(clip_img) != type(0):
                    #     gray = cv2.cvtColor(clip_img, cv2.COLOR_BGR2GRAY)
                    #     cv2.imwrite(filepathsave,gray)
                    #     current_files +=1
            num += 1
            sys.stdout.write('\r{}----{}/{}, MissedFiles : {}'.format(num,current_files,total_files,missed_files))
            sys.stdout.flush()

        except Exception as ex:
            # print(file_path)
            # print(ex)
            # print(date_birth)
            missed_files += 1

    for folder in os.listdir('datasets/ages/training'):
        files_len = len(os.listdir('datasets/ages/training/'+folder))
        if not os.path.exists('datasets/ages/test/'+folder):
            os.mkdir('datasets/ages/test/'+folder)
        for file in random.sample(os.listdir('datasets/ages/training/'+folder),int(files_len * 0.20)):
            shutil.move('datasets/ages/training/'+folder +'/' + str(file), 'datasets/ages/test/'+folder + '/' + file)
        sys.stdout.write('\r Moveing')
        sys.stdout.flush()
