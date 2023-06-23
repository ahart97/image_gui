import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QTabWidget, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
from Models.YOLO_model import YoloModel

class CameraTab(QWidget):
    def __init__(self):
        super().__init__()

        #Create YOLO model
        self.object_model = YoloModel()

        # Create the video feed label
        self.video_feed_label = QLabel()

        # Create the image and object classifer objects
        self.objects = ['']
        self.object_button = QPushButton('Classify objects in image')
        self.object_button.clicked.connect(self.object_pipeline)
        self.object_label = QLabel('Objects: {}'.format('\n'.join(self.objects)))


        self.image = ''
        self.image_button = QPushButton('Classify image')
        self.image_button.clicked.connect(self.image_pipeline)
        self.image_label = QLabel('Image: {}'.format(self.image))


        # Create the layout and add the widgets
        layout = QVBoxLayout()
        layout.addWidget(self.video_feed_label)
        layout.addWidget(self.object_button)
        layout.addWidget(self.object_label)
        layout.addWidget(self.image_button)
        layout.addWidget(self.image_label)


        self.setLayout(layout)

        # Create a timer to update the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(30)

        # Create the camera object
        self.camera = cv2.VideoCapture(0)

    def image_pipeline(self):
        # Add your processing pipeline here
        pass

    def object_pipeline(self):
        self.object_model.classify_objects(self.rgb_image)
        self.objects = self.object_model.observed_classes
        self.object_label.setText('Objects: {}'.format('\n'.join(self.objects)))

    def update_video_feed(self):
        # Read the image from the camera
        ret, frame = self.camera.read()

        # Convert the image to a QImage
        if ret:
            self.rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = self.rgb_image.shape
            q_image = QImage(self.rgb_image.data, w, h, ch*w, QImage.Format_RGB888)
            self.video_feed_label.setPixmap(QPixmap.fromImage(q_image))