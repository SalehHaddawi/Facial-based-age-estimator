import pandas as pd
import shutil
import os
import datetime
import math

dataset = pd.read_csv("selfie_dataset.txt", delimiter=" ")

rows = dataset.iloc[:, [0, 4, 5, 6, 7, 8, 9]].values


for row in rows:
    print(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
    if row[1] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'baby/'+row[0]+'.jpg')
    if row[2] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'child/'+row[0]+'.jpg')
    if row[3] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'teenager/'+row[0]+'.jpg')
    if row[4] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'youth/'+row[0]+'.jpg')
    if row[5] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'middle_age/'+row[0]+'.jpg')
    if row[6] == 1:
        if os.path.isfile('images/' + row[0] + '.jpg'):
            shutil.move('images/'+row[0]+'.jpg', 'senior/'+row[0]+'.jpg')



# For Wiki Images Classification..


folders = os.listdir('../datasets/wiki_crop')
missingImages = []
for folder in folders:
    if str(folder) == 'wiki.mat':
        continue

    for file in os.listdir('../datasets/wiki_crop/' + folder):
        filename = file.split('_')
        date_birth = filename[1].split('-')
        date_taket = filename[2].split('.')[0]
        print([date_birth, date_taket])

        try:
            if int(date_birth[2]) == 00:
                date_birth[2] = int(date_birth[2]) + 1
            if int(date_birth[1]) > 12:
                date_birth[1], date_birth[2] = date_birth[2], date_birth[1]
            date_brith = datetime.datetime(int(date_birth[0]), int(date_birth[1]), int(date_birth[2]))
            date_taket = datetime.datetime(int(date_taket), 6, 15)
            age = math.ceil((date_taket - date_brith).days / 365)
        except:
            missingImages.append([folder, file])

        if 0 < age < 4:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/baby/' + str(file))
        if 6 < age < 16:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/child/' + str(file))
        if 17 < age < 20:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/teenager/' + str(file))
        if 21 < age < 32:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/youth/' + str(file))
        if 33 < age < 65:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/middle_age/' + str(file))
        if 66 < age:
            shutil.copy('../datasets/wiki_crop/' + str(folder) + '/' + str(file), '../datasets/ages/senior/' + str(file))
        print([folder, file, age])

print(len(missingImages))
