from FacialBasedAgeEstimator import FacialBasedAgeEstimator
import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

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
root.geometry('400x500')
root.title("Facial Based Age Estimator")
frame = Frame(root)
frame.pack(padx=30, pady=30)

load = Image.open("LogoAI.jpg")
load = load.resize((250, 177), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)

img = Label(frame, image=render)
img.image = render
img.place(x=0, y=0)
img.pack()

middleframe = Frame(frame)
middleframe.pack(pady=55)

label = Label(middleframe, text="Chose a method to detect age:", font="system 15", fg="#1A3353", pady=10).pack()


loadimgbtn = Button(middleframe, text='Load Image', fg="#801E3A", bg="white", command=get_image, width=15).pack()
loadvidbtn = Button(middleframe, text='Load Video', fg='#801E3A', bg="white", command=get_video, width=15).pack()
webcambtn = Button(middleframe, text='Web Cam', fg='#801E3A', bg="white", command=open_cam, width=15).pack()

bottomframe = Frame(frame)
bottomframe.pack()

dev = Label(bottomframe, text="Developers: Saleh ,Nawaf and Saeed.", font="system 10", fg="#8192A8").pack()

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
