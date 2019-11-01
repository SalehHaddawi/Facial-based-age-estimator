import cv2
import time
from VideoStream import VideoStream


class FacialBasedAgeEstimator:

    def __init__(self, cascade, scaleFactor=1.2):
        self.cascade = cascade
        self.scaleFactor = scaleFactor

    def predict_image(self, image):
        faces = self.detect_faces(image)

        for face in faces:
            x, y, w, h = face

            clipped_image = image[y:y + h, x:x + w]

            label = self.predict(clipped_image)

            self.draw_rect_and_text(image=image, face=face, text=label)

        return image

    # if source is 0 then its a web cam
    def predict_video(self, source=0):
        fvs = VideoStream(source, self).start()

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
        faces = self.cascade.detectMultiScale(gray_image, scaleFactor=self.scaleFactor, minNeighbors=6, minSize=(30, 30))

        return faces

    def draw_rect_and_text(self, image, face, text=""):
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def predict(self, image):
        # TODO: real prediction
        return "baby"
