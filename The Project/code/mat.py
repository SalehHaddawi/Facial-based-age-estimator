import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':

    subjects = list(range(1, 24))
    mat_dir = ''
    csv_dir = 'mat/'
    x_data_format = "%1.20e" # 21 digits in IEEE exponential format

    for subject in subjects:
        print("Subject", subject)
        if subject < 17:
            filename_mat = mat_dir + 'train_subject%02d.mat' % subject
            filename_csv = csv_dir + 'train_subject%02d.csv' % subject
        else:
            filename_mat = mat_dir + 'test_subject%02d.mat' % subject
            filename_csv = csv_dir + 'test_subject%02d.csv' % subject

        print("Loading", filename_mat)
        data = loadmat(filename_mat, squeeze_me=True)
        X = data['X']
        if subject < 17:
            y = data['y']
        else:
            y = data['Id']

        trials, channels, timepoints = X.shape
        print("trials, channels, timepoints:", trials, channels, timepoints)

        print("Creating", filename_csv)
        f = open(filename_csv, 'w')
        if subject < 17:
            print("y ,", end=' ', file=f)
        else:
            print("Id ,", end=' ', file=f)

        print("Writing CSV header.")
        for j in range(channels):
            for k in range(timepoints):
                print("X%03d%03d" % (j, k), end=' ', file=f)
                if (j < channels-1) or (k < timepoints-1):
                    print(",", end=' ', file=f)
                else:
                    print(file=f)

        print("Writing trial information.")
        for i in range(trials):
            if (i % 10) == 0:
                print("trial", i)

            print("%d," %  y[i], end=' ', file=f)
            for j in range(channels):
                for k in range(timepoints):
                    print(x_data_format % X[i,j,k], end=' ', file=f)
                    if (j < channels-1) or (k < timepoints-1):
                        print(",", end=' ', file=f)
                    else:
                        print(file=f)

        f.close()
        print("Done.")
        print()