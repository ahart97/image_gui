import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QTabWidget, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
from Camera.Webcam import CameraTab

class ClassifyTab(CameraTab):
    def __init__(self):
        super().__init__()

        # Set the tab name
        self.tab_name = "Image Prediction"


class MainWindow(QDialog):
    def __init__(self):
        super().__init__()

        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(ClassifyTab(), "Classify")

        # Create the layout and add the tab widget
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)

        self.setLayout(layout)
        self.setWindowTitle("Vision APP")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create the main window and show it
    window = MainWindow()
    window.show()

    # Start the application event loop
    sys.exit(app.exec_())
