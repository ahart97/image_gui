import sys
from PyQt5.QtWidgets import QApplication
from src.gui.gui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create the main window and show it
    window = MainWindow()
    window.show()

    # Start the application event loop
    sys.exit(app.exec_())