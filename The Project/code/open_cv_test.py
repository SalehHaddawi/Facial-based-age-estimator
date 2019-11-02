from FacialBasedAgeEstimator import FacialBasedAgeEstimator
import cv2

# ------------------------------ MAIN -------------------------------

# load the module
face_cascade = cv2.CascadeClassifier('The Project/code/haarcascade_frontalface_default.xml')

# ------------ STATIC IMAGES --------------
# scaleFactor = 1.02    
#
# fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)
#
# img = cv2.imread("family.jpg")
#
# cv2.waitKey(0)
#
# result = fbae.predict_image(img)
#
# cv2.imshow("img", result)
#
# cv2.waitKey(0)

# ------------ VIDEOS & WEB CAM --------------
scaleFactor = 1.15

fbae = FacialBasedAgeEstimator(face_cascade, scaleFactor)

# put 0 for web cam
fbae.predict_video(source="video/aa.mp4", sync=True)
