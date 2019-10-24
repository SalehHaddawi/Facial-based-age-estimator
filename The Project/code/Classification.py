import pandas as pd
import shutil

photo_taken_columns = pd.read_csv('photo_taken.csv', sep=",")
photo_taken_rows = photo_taken_columns.transpose()

date_of_birth_columns = pd.read_csv('dob.csv', sep=",")
date_of_birth_rows = date_of_birth_columns.transpose()

full_path_columns = pd.read_csv('full_path1.csv', sep=",")
full_path_rows = full_path_columns.transpose()

ss = pd.DataFrame({'full_path': full_path_rows.index.values,
                   'date_birth': date_of_birth_rows.index.values,
                   'photo_taken_date': photo_taken_rows.index.values})

print('ss')

# rows = dataset.iloc[:, [0, 4, 5, 6, 7, 8, 9]].values
#
#
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
#
