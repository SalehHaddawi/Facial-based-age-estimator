import cv2
import time
from keras.models import load_model
from VideoStream import VideoStream
import numpy as np
from keras.preprocessing import image as keras_img


class FacialBasedAgeEstimator:

    def __init__(self, cascade, scaleFactor=1.2):
        self.cascade = cascade
        self.scaleFactor = scaleFactor
        self.model = load_model('../../model/model00000002_0.11_0.90.hdf5')

    def predict_image(self, image):
        faces = self.detect_faces(image)

        print("num of faces is {}".format(len(faces)))

        for face in faces:
            face_img, clipped_image_cords = self.crop_face(image, face, margin=5)
            
            x, y, w, h = clipped_image_cords

            clipped_image = image[y:y + h, x:x + w]
            clipped_image = cv2.cvtColor(clipped_image, cv2.COLOR_BGR2GRAY)
            clipped_image = cv2.resize(clipped_image, (128, 128))

            # Predict the face in Model and return the label ex:=> (90%,Child)
            label = self.predict(clipped_image)

            self.draw_rect_and_text(image=image, face=face, text=label)

        return image

    # if source is 0 then its a web cam
    def predict_video(self, source=0, sync=False):
        fvs = VideoStream(source=source, fbae=self, sync=sync).start()

        if not sync:

            time.sleep(1.0)

            # loop over frames from the video file stream
            while fvs.more():

                # grab the frame from the threaded video file stream
                frame = fvs.read()

                cv2.imshow("Frame", frame)

                cv2.waitKey(1)

                # if [esc] is pressed
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

    def detect_faces(self, image):
        # convert the test image to gray scale as opencv face detector expects gray images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying the haar classifier to detect faces
        faces = self.cascade.detectMultiScale(gray_image, scaleFactor=self.scaleFactor, minNeighbors=6)

        return faces

    def draw_rect_and_text(self, image, face, text=""):
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    def crop_face(self, imgarray, section, margin=40, size=64):
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
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def getCateogrical(self, num):
        if num == 0:
            return 'Adult'
        if num == 1:
            return 'Child'
        if num == 2:
            return 'Senior'
        if num == 3:
            return 'Young'

        return 'UnKnown'

    def predict(self, face_image):
        try:
            img_array = keras_img.img_to_array(face_image)
            img_array = np.expand_dims(img_array, axis=0)
            img_class = self.model.predict(img_array)
            img_class = list(img_class[0])
            max1 = max(img_class)
            max_index = img_class.index(max1)

            return '{}% ,{} '.format(round(max1 * 100, 2), self.getCateogrical(max_index))
        except Exception as ex:
            print(ex)
