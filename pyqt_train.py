import sys
import os
import threading
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox, QComboBox, \
    QFileDialog, QRadioButton, QHBoxLayout, QListWidget, QGridLayout  # QGridLayout ekledik
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception, EfficientNetB0, ResNet50
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization
import traceback
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.99):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.threshold:
            self.model.stop_training = True


class MLTrainerGUI(QWidget):
    training_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None
        self.preprocess_input = None
        self.history = None
        self.train_path = None
        self.valid_path = None
        self.test_path = None
        self.training_completed.connect(self.on_training_completed)

    def init_ui(self):
        self.setWindowTitle('ML Model Trainer')
        layout = QGridLayout()  # QGridLayout kullanÄ±yoruz

        self.metrics_button = QPushButton('Show Metrics', self)

        # Initialize inputs and buttons for paths
        self.confusion_matrix_button = QPushButton('Display Confusion Matrix', self)

        self.train_path_input = QLineEdit(self)
        self.train_path_button = QPushButton('Select Train Path', self)
        self.valid_path_input = QLineEdit(self)
        self.valid_path_button = QPushButton('Select Valid Path', self)
        self.test_path_input = QLineEdit(self)
        self.test_path_button = QPushButton('Select Test Path', self)

        # Initialize buttons for viewing images
        self.view_train_images_button = QPushButton('View Train Images', self)
        self.view_valid_images_button = QPushButton('View Valid Images', self)
        self.view_test_images_button = QPushButton('View Test Images', self)

        # Initialize list widgets and image label
        self.train_list = QListWidget()
        self.valid_list = QListWidget()
        self.test_list = QListWidget()
        self.image_label = QLabel('Selected image will appear here')
        self.image_label.setAlignment(Qt.AlignCenter)

        # Initialize model selector and add model names
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(['Xception', 'EfficientNetB0', 'ResNet50', 'CustomCNN'])

        # Initialize other inputs and buttons
        self.epoch_input = QLineEdit(self)
        self.batch_input = QLineEdit(self)
        self.patience_input = QLineEdit(self)
        self.checkpoint_radio = QRadioButton("Enable Model Checkpoint", self)
        self.train_button = QPushButton('Train Model', self)
        self.plot_button = QPushButton('Plot Training History', self)

        # Layout setup and other initializations
        self.setup_layout(layout)
        self.connect_signals()

    def setup_layout(self, layout):
        layout.addWidget(self.confusion_matrix_button, 0, 0)
        layout.addWidget(self.metrics_button, 0, 1)

        layout.addWidget(QLabel('Train Path'), 1, 0)
        layout.addWidget(self.train_path_input, 1, 1)
        layout.addWidget(self.train_path_button, 1, 2)
        layout.addWidget(QLabel('Valid Path'), 2, 0)
        layout.addWidget(self.valid_path_input, 2, 1)
        layout.addWidget(self.valid_path_button, 2, 2)
        layout.addWidget(QLabel('Test Path'), 3, 0)
        layout.addWidget(self.test_path_input, 3, 1)
        layout.addWidget(self.test_path_button, 3, 2)

        layout.addWidget(self.view_train_images_button, 4, 0)
        layout.addWidget(self.view_valid_images_button, 4, 1)
        layout.addWidget(self.view_test_images_button, 4, 2)

        list_layout = QHBoxLayout()
        list_layout.addWidget(self.train_list)
        list_layout.addWidget(self.valid_list)
        list_layout.addWidget(self.test_list)
        layout.addLayout(list_layout, 5, 0, 1, 3)  # span 1 row, 3 columns

        layout.addWidget(self.image_label, 6, 0, 1, 3)  # span 1 row, 3 columns

        layout.addWidget(QLabel('Select Model'), 7, 0)
        layout.addWidget(self.model_selector, 7, 1)
        layout.addWidget(QLabel('Epochs'), 8, 0)
        layout.addWidget(self.epoch_input, 8, 1)
        layout.addWidget(QLabel('Batch Size'), 9, 0)
        layout.addWidget(self.batch_input, 9, 1)
        layout.addWidget(QLabel('Patience'), 10, 0)
        layout.addWidget(self.patience_input, 10, 1)
        layout.addWidget(self.checkpoint_radio, 11, 0)
        layout.addWidget(self.train_button, 12, 0, 1, 2)  # span 1 row, 2 columns
        layout.addWidget(self.plot_button, 12, 2)

        self.setLayout(layout)

    def connect_signals(self):
        self.metrics_button.clicked.connect(self.show_metrics)
        self.confusion_matrix_button.clicked.connect(self.display_confusion_matrix)
        self.train_path_button.clicked.connect(lambda: self.get_path(self.train_path_input, self.train_list))
        self.valid_path_button.clicked.connect(lambda: self.get_path(self.valid_path_input, self.valid_list))
        self.test_path_button.clicked.connect(lambda: self.get_path(self.test_path_input, self.test_list))
        self.view_train_images_button.clicked.connect(
            lambda: self.open_folder(self.train_path_input.text(), self.train_list))
        self.view_valid_images_button.clicked.connect(
            lambda: self.open_folder(self.valid_path_input.text(), self.valid_list))
        self.view_test_images_button.clicked.connect(
            lambda: self.open_folder(self.test_path_input.text(), self.test_list))
        self.train_button.clicked.connect(self.on_train_click)
        self.plot_button.clicked.connect(self.plot_history)
        self.train_list.clicked.connect(lambda: self.display_image(self.train_list, self.train_path_input.text()))
        self.valid_list.clicked.connect(lambda: self.display_image(self.valid_list, self.valid_path_input.text()))
        self.test_list.clicked.connect(lambda: self.display_image(self.test_list, self.test_path_input.text()))

    def on_train_click(self):
        selected_model = self.model_selector.currentText()
        epochs = int(self.epoch_input.text())
        batch_size = int(self.batch_input.text())
        patience = int(self.patience_input.text())

        threading.Thread(target=self.train_model, args=(selected_model, epochs, batch_size, patience),
                         daemon=True).start()

    def display_confusion_matrix(self):
        if self.model is not None:
            test_path = self.test_path_input.text() if self.test_path_input else 'test'
            test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
            test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=1,
                                                              shuffle=False)

            y_true = test_generator.classes
            y_pred_probs = self.model.predict(test_generator)
            y_pred = y_pred_probs.argmax(axis=1)

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
            fig, ax = plt.subplots(figsize=(18, 20))
            disp.plot(ax=ax)
            plt.xticks(rotation=45)
            plt.show()
        else:
            QMessageBox.warning(self, "Model Not Trained", "Train the model first to display the confusion matrix.")

    def show_metrics(self):
        try:
          if self.model is not None:
              test_path = self.test_path_input.text() if self.test_path_input.text() else 'test'
              test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
              test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=1, shuffle=False)

              y_true = test_generator.classes
              y_pred_probs = self.model.predict(test_generator)
              y_pred = y_pred_probs.argmax(axis=1)

              report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()), zero_division=1)

              # Correctly parsing the classification report to extract precision, recall, and f1-score
              lines = report.split('\n')
              class_names = []
              precision_values = []
              recall_values = []
              f1_values = []

              for line in lines[2:-3]:  # Skip the header and summary lines
                  parts = line.split()
                  if len(parts) == 5:  # Only valid lines with 5 parts
                      class_names.append(parts[0])
                      precision_values.append(float(parts[1]))
                      recall_values.append(float(parts[2]))
                      f1_values.append(float(parts[3]))

              # Plot the metrics
              fig, ax = plt.subplots(figsize=(10, 6))
              ind = np.arange(len(class_names))
              width = 0.2

              ax.bar(ind - width, precision_values, width, label='Precision')
              ax.bar(ind, recall_values, width, label='Recall')
              ax.bar(ind + width, f1_values, width, label='F1-score')

              ax.set_xticks(ind)
              ax.set_xticklabels(class_names)
              ax.legend()

              plt.xlabel('Classes')
              plt.title('Classification Metrics')
              plt.show()
          else:
              QMessageBox.warning(self, "Model Not Trained", "Train the model first to show metrics.")
        except Exception as e:
          QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")

    def train_model(self, model_name, epochs, batch_size, patience):
        img_size = (224, 224)
        train_path = self.train_path_input.text() if self.train_path_input else 'train'
        valid_path = self.valid_path_input.text() if self.valid_path_input else 'valid'
        test_path = self.test_path_input.text() if self.test_path_input else 'test'

        if model_name == 'Xception':
            self.model = self.create_xception_model(img_size)
            self.preprocess_input = xception_preprocess
        elif model_name == 'EfficientNetB0':
            self.model = self.create_efficientnet_model(img_size)
            self.preprocess_input = efficientnet_preprocess
        elif model_name == 'ResNet50':
            self.model = self.create_resnet_model(img_size)
            self.preprocess_input = resnet_preprocess
        elif model_name == 'CustomCNN':
            self.model = self.create_custom_cnn_model(img_size)
            self.preprocess_input = xception_preprocess

        train_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
        valid_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)

        train_generator = train_datagen.flow_from_directory(train_path, target_size=img_size, batch_size=batch_size,
                                                            class_mode='categorical')
        valid_generator = valid_datagen.flow_from_directory(valid_path, target_size=img_size, batch_size=batch_size,
                                                            class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(test_path, target_size=img_size, batch_size=batch_size,
                                                          class_mode='categorical')

        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        custom_callback = CustomCallback()
        callbacks = [custom_callback, early_stopping]

        if self.checkpoint_radio.isChecked():
            checkpoint_path = "model_checkpoint.h5"
            model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
            callbacks.append(model_checkpoint)

        self.history = self.model.fit(train_generator, validation_data=valid_generator, epochs=epochs,
                                      callbacks=callbacks)

        test_loss, test_accuracy = self.model.evaluate(test_generator)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        model_save_path = f"{model_name}_model.h5"
        self.model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        self.training_completed.emit()

    def get_path(self, path_input, list_widget):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        path_input.setText(folder_path)

    def open_folder(self, path, list_widget):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "The path does not exist.")
            return

        list_widget.clear()
        for entry in os.scandir(path):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                list_widget.addItem(entry.name)

    def display_image(self, list_widget, folder_path):
        selected_item = list_widget.currentItem()
        if selected_item:
            file_path = os.path.join(folder_path, selected_item.text())
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def create_xception_model(self, img_size):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        return Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(10, activation='softmax')
        ])

    def create_efficientnet_model(self, img_size):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        return Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(10, activation='softmax')
        ])

    def create_resnet_model(self, img_size):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        return Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(10, activation='softmax')
        ])

    def create_custom_cnn_model(self, img_size):
        input_img = Input(shape=img_size + (3,))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(10, activation='softmax')(x)
        return Model(inputs=input_img, outputs=output)


    def on_training_completed(self):
        QMessageBox.information(self, "Training Complete", "Training has been completed successfully.")
        self.plot_button.setEnabled(True)

    def plot_history(self):
        if self.history:
            plt.figure()
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Train the model first to plot the history.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MLTrainerGUI()
    ex.show()
    sys.exit(app.exec_())