# -*- coding: utf-8 -*-
import sys
import traceback

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import keras

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QComboBox, \
    QMessageBox, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from keras.src.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.class_labels = ["African Leopard", "Caracal", "Cheetah", "Clouded Leopard", "Jaguar", "Lions", "Ocelot",
                             "Puma", "Snow Leopard", "Tiger"]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Resim Seçici')

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap(''))
        self.test_path_input = QLineEdit(self)  # Add this line for test path input
        self.test_path_input.setPlaceholderText('Enter test path')  # Optional placeholder text
        self.select_button = QPushButton('Resim Seç', self)
        self.select_button.clicked.connect(self.open_image)

        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(["Xception", "EfficientNetB0", "Resnet", "CNN"])

        self.analyze_button = QPushButton('Resmi Analiz Et', self)
        self.analyze_button.clicked.connect(self.analyze_image)

        self.hybrid_button = QPushButton('Hybrid Model Voting', self)
        self.hybrid_button.clicked.connect(self.hybrid_model_vote)
        self.confusion_matrix_button = QPushButton('Confusion Matrix', self)
        self.confusion_matrix_button.clicked.connect(self.show_confusion_matrix)
        self.select_test_path_button = QPushButton('Select Test Path', self)
        self.select_test_path_button.clicked.connect(self.select_test_path)
        self.show_metrics_button = QPushButton('Show Metrics', self)
        self.show_metrics_button.clicked.connect(self.show_metrics)


        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.model_combobox)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.hybrid_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.test_path_input)
        layout.addWidget(self.select_test_path_button)
        layout.addWidget(self.confusion_matrix_button)
        layout.addWidget(self.show_metrics_button)

        self.setLayout(layout)

    def show_metrics(self):
        try:
            selected_model = self.model_combobox.currentText()
            if selected_model == "Xception":
                model_path = "big_cats_model(224x244).h5"
                model = load_model(model_path)
                preprocess_func = keras.applications.xception.preprocess_input
            elif selected_model == "EfficientNetB0":
                model_path = "big_cats_efficientnetB0_model(224x244).h5"
                model = load_model(model_path)
                preprocess_func = keras.applications.efficientnet.preprocess_input
            elif selected_model == "Resnet":
                model_path = "resnet_model.h5"
                model = load_model(model_path)
                preprocess_func = lambda x: x / 255.0
            elif selected_model == "CNN":
                model_path = "final_cnn_model (1).h5"
                model = load_model(model_path)
                preprocess_func = None
            else:
                print("Geçersiz model seçimi.")
                return

            if model is not None:
                test_path = self.test_path_input.text() if self.test_path_input else 'test'
                test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
                test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=1,
                                                                  shuffle=False)

                y_true = test_generator.classes
                y_pred_probs = model.predict(test_generator)
                y_pred = y_pred_probs.argmax(axis=1)

                report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()),
                                              zero_division=1)

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
    def select_test_path(self):
        test_path_dialog = QFileDialog()
        test_path = test_path_dialog.getExistingDirectory(self, 'Select Test Path', '')

        if test_path:
            self.test_path_input.setText(test_path)
    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Resim Seç', '',
                                                   'Image Files (*.png;*.jpg;*.jpeg;*.gif;*.bmp)')

        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaledToWidth(300)
            self.image_label.setPixmap(pixmap)
            self.image_path = file_path

    def show_confusion_matrix(self):
        try:
            selected_model = self.model_combobox.currentText()
            if selected_model == "Xception":
                model_path = "big_cats_model(224x244).h5"
                model = load_model(model_path)
                preprocess_func = keras.applications.xception.preprocess_input
            elif selected_model == "EfficientNetB0":
                model_path = "big_cats_efficientnetB0_model(224x244).h5"
                model = load_model(model_path)
                preprocess_func = keras.applications.efficientnet.preprocess_input
            elif selected_model == "Resnet":
                model_path = "resnet_model.h5"
                model = load_model(model_path)
                preprocess_func = lambda x: x / 255.0
            elif selected_model == "CNN":
                model_path = "final_cnn_model (1).h5"
                model = load_model(model_path)
                preprocess_func = None
            else:
                print("Geçersiz model seçimi.")
                return

            if model is not None:
                test_path = self.test_path_input.text() if self.test_path_input else 'test'
                test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
                test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=1,
                                                                  shuffle=False)

                y_true = test_generator.classes
                y_pred_probs = model.predict(test_generator)
                y_pred = y_pred_probs.argmax(axis=1)

                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
                fig, ax = plt.subplots(figsize=(18, 20))
                disp.plot(ax=ax)
                plt.xticks(rotation=45)
                plt.show()
            else:
                QMessageBox.warning(self, "Model Not Trained", "Train the model first to display the confusion matrix.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")

    def analyze_image(self):
        if hasattr(self, 'image_path'):
            selected_model = self.model_combobox.currentText()

            if selected_model == "Xception":
                from keras.applications.xception import preprocess_input
                model_path = "big_cats_model(224x244).h5"
                model = load_model(model_path)
                self.pred_model(model, preprocess_input)

            elif selected_model == "EfficientNetB0":
                from keras.applications.efficientnet import preprocess_input
                model_path = "big_cats_efficientnetB0_model(224x244).h5"
                model = load_model(model_path)
                self.pred_model(model, preprocess_input)

            elif selected_model == "Resnet":
                from keras.preprocessing import image

                model_path = "resnet_model.h5"  # Update this path to your ResNet model file
                model = load_model(model_path)

                # Load and preprocess the test image
                img = image.load_img(self.image_path,
                                     target_size=(224, 224))  # Adjust target size based on your ResNet model
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Rescale pixel values to [0, 1], if needed

                # Predict the class for the test image
                self.perform_predictions(img_array, model)

            elif selected_model == "CNN":
                model_path = "final_cnn_model (1).h5"
                model = load_model(model_path)
                self.pred_model(model, None)  # Pass None for preprocess_input

            else:
                print("Geçersiz model seçimi.")

    def pred_model(self, model, preprocess_func):
        try:
            # Seçilen resmi yükleme ve ön işleme
            img = image.load_img(self.image_path, target_size=(224, 224))  # Adjust target size accordingly
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            if preprocess_func:
                img_array = preprocess_func(img_array)

            # Modeli kullanarak tahmin yapma
            self.perform_predictions(img_array, model)

        except Exception as e:
            print("An error occurred:", e)

    def perform_predictions(self, img_array, model):
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        class_labels = ["African Leopard", "Caracal", "Cheetah", "Clouded Leopard", "Jaguar", "Lions", "Ocelot",
                        "Puma", "Snow Leopard", "Tiger"]
        predicted_class_label = class_labels[predicted_class_index]
        result_text = f"Predicted Class Label: {predicted_class_label}"
        self.result_label.setText(result_text)

    def hybrid_model_vote(self):
        if hasattr(self, 'image_path'):
            # Load models with their corresponding preprocessing functions
            models_with_preprocessing = [
                ("Xception", load_model("big_cats_model(224x244).h5"), keras.applications.xception.preprocess_input),
                ("EfficientNetB0", load_model("big_cats_efficientnetB0_model(224x244).h5"),
                 keras.applications.efficientnet.preprocess_input),
                ("ResNet", load_model("resnet_model.h5"), lambda x: x / 255.0),
                # Assuming ResNet uses simple scaling
                ("CNN", load_model("final_cnn_model (1).h5"), None)
                # If CNN does not require special preprocessing
            ]

            # Get predictions from each model
        all_predictions = []
        for model_name, model, preprocess_func in models_with_preprocessing:
            predicted_class_label = self.get_model_specific_prediction(model, self.image_path, preprocess_func)
            all_predictions.append(predicted_class_label)

        # Perform voting using the aggregate_predictions method
        voted_result = self.aggregate_predictions(all_predictions)

        # Display individual predictions and voting result
        individual_predictions_text = "\n".join([f"{model_name}: {prediction}" for model_name, prediction in
                                                 zip([model[0] for model in models_with_preprocessing],
                                                     all_predictions)])
        final_result_text = f"Voting Result: {voted_result}"
        self.result_label.setText(individual_predictions_text + "\n" + final_result_text)

    def get_model_specific_prediction(self, model, image_path, preprocess_func):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        if preprocess_func is not None:
            img_array = preprocess_func(img_array)

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        return self.class_labels[predicted_class_index]

    def aggregate_predictions(self, predictions):
        class_labels = ["African Leopard", "Caracal", "Cheetah", "Clouded Leopard", "Jaguar", "Lions", "Ocelot", "Puma",
                        "Snow Leopard", "Tiger"]

        vote_counts = {}
        for prediction in predictions:
            if prediction in vote_counts:
                vote_counts[prediction] += 1
            else:
                vote_counts[prediction] = 1

        # Find the prediction with the maximum votes
        most_common_prediction = max(vote_counts, key=vote_counts.get)
        return most_common_prediction


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSelector()
    window.show()
    sys.exit(app.exec_())