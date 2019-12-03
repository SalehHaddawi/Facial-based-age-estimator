from FacialBasedAgeEstimator import FacialBasedAgeEstimator
import cv2
import numpy
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfile
from PIL import Image, ImageTk


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
root.geometry('400x500')
root.title("Facial Based Age Estimator")
frame = Frame(root)
frame.pack(padx=30, pady=30)

load = Image.open("logoAI.png")
load = load.resize((250, 177), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)

img = Label(frame, image=render)
img.image = render
img.place(x=0, y=0)
img.pack()

middleframe = Frame(frame)
middleframe.pack(pady=55)

label = Label(middleframe, text="Choose a method to estimate age:", font="system 15", fg="#1A3353", pady=10).pack()

loadimgbtn = Button(middleframe, text='Load Image', fg="#801E3A", bg="white", command=get_image, width=15).pack()
loadvidbtn = Button(middleframe, text='Load Video', fg='#801E3A', bg="white", command=get_video, width=15).pack()
webcambtn = Button(middleframe, text='Web Cam', fg='#801E3A', bg="white", command=open_cam, width=15).pack()

bottomframe = Frame(frame)
bottomframe.pack()

dev = Label(bottomframe, text="Developers: Saleh ,Nawaf and Saeed.", font="system 10", fg="#8192A8").pack()

root.mainloop()

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
