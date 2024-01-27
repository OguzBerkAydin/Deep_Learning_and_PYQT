import sys
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
 

class ImageSelectorDataAu(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
 
    def init_ui(self):
        self.setWindowTitle('Resim Seçici')

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap(''))

        self.select_button = QPushButton('Resim Seç', self)
        self.select_button.clicked.connect(self.open_image)

        self.augment_button = QPushButton('Resmi Dönüştür', self)
        self.augment_button.clicked.connect(self.augment_image)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.augment_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Resim Seç', '',
                                                   'Image Files (*.png;*.jpg;*.jpeg;*.gif;*.bmp)')

        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaledToWidth(300))
            self.image_path = file_path

    # ...

    def augment_image(self):
        if hasattr(self, 'image_path'):
            img = image.load_img(self.image_path, target_size=(300, 300))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            it = self.datagen.flow(img_array, batch_size=1)
            batch = it.next()
            augmented_image = batch[0].astype('uint8')

            # Convert the PIL Image to a format suitable for QImage
            h, w, ch = augmented_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(augmented_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(300, 300, Qt.KeepAspectRatio)

            self.image_label.setPixmap(QPixmap.fromImage(p))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSelectorDataAu()
    window.show()
    sys.exit(app.exec_())

