from FacialBasedAgeEstimator import FacialBasedAgeEstimator
import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename




face_cascade = cv2.CascadeClassifier('The Project/code/haarcascade_frontalface_default.xml')


def get_image():
    image = askopenfilename(filetypes=[("Image", ".jpg .png .jpeg")])
    print(image)
    scaleFactor = 1.05

    fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)

    img = cv2.imread(image)

    cv2.waitKey(0)
    result = fbae.predict_image(img)

    cv2.imshow("img", result)

    cv2.waitKey(0)

def get_video():
    video = askopenfilename(filetypes=[("Image", ".mp4 .mkv")])
    print(video)
    scaleFactor = 1.15
    fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
    fbae.predict_video(source=video, sync=True)

def open_cam():
    scaleFactor = 1.15
    fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
    fbae.predict_video(source=0, sync=True)


root = Tk() 
frame = Frame(root) 
frame.pack() 
bottomframe = Frame(root) 
bottomframe.pack( side = BOTTOM ) 
redbutton = Button(frame, text = 'Load Image', fg ='black',command=get_image) 
redbutton.pack( side = TOP) 
greenbutton = Button(frame, text = 'Load Video', fg='black',command=get_video) 
greenbutton.pack( side = TOP ) 
bluebutton = Button(frame, text ='Web Cam', fg ='black',command=open_cam) 
bluebutton.pack( side = TOP ) 
root.mainloop() 







# # ------------------------------ MAIN -------------------------------

# # load the module
# face_cascade = cv2.CascadeClassifier('The Project/code/haarcascade_frontalface_default.xml')

# # ------------ STATIC IMAGES --------------
# # scaleFactor = 1.05

# # fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)

# # img = cv2.imread("The Project/code/baby.jpeg")

# # cv2.waitKey(0)
# # result = fbae.predict_image(img)

# # cv2.imshow("img", result)

# # cv2.waitKey(0)

# # ------------ VIDEOS & WEB CAM --------------
# # scaleFactor = 1.15

# # fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)