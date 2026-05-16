import cv2
import threading

from MongoDBImporter import MongoDBImporter


class ThreadedCamera:
    def __init__(self, name, device_index=0, width=None, height=None):
        self.cap = cv2.VideoCapture(device_index)
        self.mongo = MongoDBImporter()
        self.connected = self.mongo.test_connection()
        self.name = name
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device_index}")

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self.ret, self.frame = self.cap.read()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if self.connected:
                self.mongo.saveImage(frame, self.name)
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()