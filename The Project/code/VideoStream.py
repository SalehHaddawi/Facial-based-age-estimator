# import the necessary packages
from threading import Thread
import sys
import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


class VideoStream:
    def __init__(self, source, fbae, queueSize=128, sync=False):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(source)
        self.stopped = False
        self.fbae = fbae
        self.sync = sync

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        (grabbed, frame) = self.stream.read()
        self.Q.put(frame)

        if not self.sync:
            # start a thread to read frames from the file video stream
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()

            return self
        else:
            self.update()

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                result = self.fbae.predict_image(frame)

                if self.sync:
                    cv2.imshow('Preview', result)

                    # if [esc] is pressed
                    k = cv2.waitKey(50) & 0xff
                    if k == 27:
                        break
                    # if window [X] button is clicked
                    elif cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) < 1:
                        break
                else:
                    # add the frame to the queue
                    self.Q.put(result)

        cv2.destroyAllWindows()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
