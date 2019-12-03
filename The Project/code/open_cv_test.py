from FacialBasedAgeEstimator import FacialBasedAgeEstimator
import cv2
import numpy
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfile

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

fbae_image = FacialBasedAgeEstimator(face_cascade, 1.05)


fbae_vide_cam = FacialBasedAgeEstimator(face_cascade, 1.15)


def get_image():
    path = askopenfile(filetypes=[("Image", ".jpg .png .jpeg")]).name
    # scaleFactor = 1.05
    #
    # fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)

    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    cv2.waitKey(0)
    result = fbae_image.predict_image(bgrImage)

    cv2.imshow("img", result)

    cv2.waitKey(0)


def get_video():
    video = askopenfilename(filetypes=[("Image", ".mp4 .mkv")])
    # scaleFactor = 1.15
    # fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
    fbae_vide_cam.predict_video(source=video, sync=True)


def open_cam():
    scaleFactor = 1.15
    fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
    fbae.predict_video(source=0, sync=True)


root = Tk()
frame = Frame(root)
frame.pack()
bottomframe = Frame(root)
bottomframe.pack(side=BOTTOM)

open_image_button = Button(frame, text='Load Image', fg='black', command=get_image)
open_image_button.pack(side=TOP)

open_video_button = Button(frame, text='Load Video', fg='black', command=get_video)
open_video_button.pack(side=TOP)

web_cam_button = Button(frame, text='Web Cam', fg='black', command=open_cam)
web_cam_button.pack(side=TOP)

root.mainloop()

# # ------------------------------ MAIN -------------------------------

# load the module
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # # ------------ STATIC IMAGES --------------
# scaleFactor = 1.05
#
# fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
#
# img = cv2.imread("family.jpg")
#
# cv2.waitKey(0)
# result = fbae.predict_image(img)
#
# cv2.imshow("img", result)
#
# cv2.waitKey(0)

# # ------------ VIDEOS & WEB CAM --------------
# # scaleFactor = 1.15

# # fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
