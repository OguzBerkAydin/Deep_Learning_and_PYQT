import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from pyqt_test import ImageSelector  # Assuming the test UI is in a file named test_ui.py
from train_metrics import MLTrainerGUI  # Assuming the train UI is in a file named train_ui.py
from data_aug import ImageSelectorDataAu
class MainApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Main Application')

        # Create the tab widget
        tab_widget = QTabWidget(self)

        # Create instances of the Test and Train classes
        test_ui = ImageSelector()
        train_ui = MLTrainerGUI()
        data_aug = ImageSelectorDataAu()

        # Add the Test and Train widgets to the tabs
        tab_widget.addTab(test_ui, "Test")
        tab_widget.addTab(train_ui, "Train")
        tab_widget.addTab(data_aug, "Data Augmentation")


        # Create the main layout
        layout = QVBoxLayout(self)
        layout.addWidget(tab_widget)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
