# import cv2
#
# from faced import FaceDetector
# from faced.utils import annotate_image
#
# face_detector = FaceDetector()
#
# img = cv2.imread("family.jpg")
# rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#
# # Receives RGB numpy image (HxWxC) and
# # returns (x_center, y_center, width, height, prob) tuples.
# bboxes = face_detector.predict(rgb_img, 0.3)
#
# # Use this utils function to annotate the image.
# ann_img = annotate_image(img, bboxes)
#
# # Show the image
# cv2.imshow('image', ann_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import time


def draw_rect_and_text(image, face, text=""):
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)


def crop_face(imgarray, section, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


# extract faces from a given photograph
def extract_faces(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    start = time.time()
    results = detector.detect_faces(pixels)
    print(time.time() - start)

    faces_pixels = []
    for i in results:
        # extract the bounding box from the face
        x1, y1, width, height = i['box']
        x2, y2 = x1 + width, y1 + height
        clipped_image = crop_face(pixels,i['box'], 15)
        clipped_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2GRAY)
        clipped_image = cv2.resize(clipped_image, (128, 128))
        cv2.imshow('anta klb', clipped_image)
        cv2.waitKey(0)
        # extract the face
        # face = pixels[y1:y2, x1:x2]
        # draw_rect_and_text(pixels, i['box'], "hi")
        # resize pixels to the model size
        # image = Image.fromarray(face)
        # image = image.resize(required_size)
        # face_array = asarray(image)
        # faces_pixels.append(face_array)
    return pixels


# load the photo and extract the face
faces_pixels = extract_faces('1.jpg')
cv2.imshow("ss0", faces_pixels)
cv2.waitKey(0)
# print(len(faces_pixels))
# faces_pixels = extract_faces('family.jpg')
# faces_pixels = extract_faces('family.jpg')
# faces_pixels = extract_faces('family.jpg')
# faces_pixels = extract_faces('family.jpg')
# for face in faces_pixels:
#     # plot the extracted face
#     cv2.imshow('img', face)
#     cv2.waitKey(0)
#     # pyplot.imshow(face)
#     # # show the plot
#     # pyplot.show()
#     # pass
