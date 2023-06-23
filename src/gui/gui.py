from PyQt5.QtWidgets import QDialog, QTabWidget, QVBoxLayout
from src.peripherals.Webcam import CameraTab

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



