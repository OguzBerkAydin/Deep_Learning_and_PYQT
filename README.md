       	



 


T.C.
KASTAMONU ÜNİVERSİTESİ
MÜHENDİSLİK VE MİMARLIK FAKÜLTESİ
BİLGİSAYAR MÜHENDİSLİĞİ PROGRAMI	





YAPAY ZRKA VE UZMAN SİSTEMLER




DEEP LEARNING İLE YAPILAN MODEL PROJESİ
HAZIRLAYAN
194410021- Arman Soylu
194410009 Oğuz Berk Aydın
DANIŞMAN
Doç. Dr. Kemal AKYOL

OCAK - 2024
KASTAMONU





T.C.
KASTAMONU ÜNİVERSİTESİ
MÜHENDİSLİK VE MİMARLIK FAKÜLTESİ

ÖZET

Proje


DEEP LEARNING ve ARAYÜZ PROJESİ

HAZIRLAYANLAR
Arman Soylu
Oğuz Berk Aydın

Kastamonu Üniversitesi
Mühendislik ve Mimarlık Fakültesi
Bilgisayar Mühendisliği Bölümü

Bitirme Projesi Danışmanı/Danışmanları:
Doç. Dr. Kemal AKYOL


Kaggle’dan seçtiğimiz Vahşi hayattaki 10 kedi veri setimizi Kerastan hazır model olarak CNN’ler yardımı ile Vahşi hayattakı 10 kedi veri setimizin modellerini oluşturduk bunun dışında kendi CNN’imizi yaptık ve bunları bir arayüzde hem eğitim hem de gerçek dünya test aşamasında gerçekleştirmiş olduk.Veri setimizi ilk başta temizlemeden işlem yaptığımız için eğitimde sıkıntılar yaşadık ama bozuk resimleri temizledikten sonra eğitim aşaması çok verimli geçti.



1.0 Giriş	4
1.2 Modeller	4
1.3 EfficientNetB0 Eğitim	4
1.3.1 Kütüphaneler	4
1.3.2 Konfigürasyonları Belirleme	6
1.3.3 Veri Ön İşleme ve Veri Çoklama	6
1.3.4 Modelin Tanımlanması ve Derlenmesi (EfficientNetB0):	7
1.3.5 Eğitim için Geri Çağırımların Tanımlanması	8
1.3.6 Modelin Eğitilmesi	8
1.3.7 Eğitim Geçmişinin Çizdirilmesi	8
1.3.8 Modelin Test Kümesinde Değerlendirilmesi	9
1.3.9 Değerlendirme Metriklerinin Gösterilmesi	9
1.3.9.1 Karışıklık Matrisi ve Sınıflandırma Raporu	9
1.3.9.2 Eğitilmiş Modelin ve Ağırlıkların Kaydedilmesi	9
1.4 Xception ve Resnet eğitimleri	10
1.5 Custom CNN	10
2.0 Uygulama Geliştirme Ortamı	11
2.1 Test	11
2.1.1 Arayüz	12
2.1.2 Resim seçme metodu	12
2.1.3 Test İşleminin gerçekleştiği metotlar	13
2.1.4 Hibrit Test İşlemi	14
2.1.5 Başlatma şablonu	15
2.2 Eğitim	15
2.3 Eğitim Ve Test Birleştirilmesi	16
2.4 Test ve Eğitim Arayüzü	17
3.0 Test Sonuçları	18
3.1 Xception ve Efficient	18
3.2 Resnet	19
3.3 Custom CNN	20
4.0 Veri seti ve modeller	21




1.0 Giriş
Bu projenin amacı, doğada bulunan farklı türdeki 10 tane büyük kediyi doğru bir şekilde sınıflandırmak ve keras application üzerinden alınmış 3 tane önceden eğitilmiş derin öğrenme modeli kullanılarak modellerinin bu görevdeki performanslarını karşılaştırmaktır. Daha sonra custom bir CNN yapısı oluşturularak tekrar bir sınıflandırma işlemi yapılacak olup sonuçları karşılaştırılacaktır. Derin öğrenme, görsel tanıma görevlerinde son yıllarda önemli başarılar elde etmiştir ve bu proje, belirli bir alan - yaban hayatı tanıma - üzerinde bu teknolojinin etkinliğini araştırmayı amaçlamaktadır.

1.2 Modeller
-ResNet: 50 katmanlı bir residual network modelidir.
-EfficientNetB0: Ölçeklenebilirlik ve performans dengesi için tasarlanmıştır.
-Xception: Derinlik öncelikli ayrılabilir evrişimler kullanır.
-Özelleştirilmiş CNN: Birden çok evrişim, havuzlama ve yoğun katmanı ile uygun bir CNN oluşturmak.

1.3 EfficientNetB0 Eğitim
Bu aşamada Keras Application’dan indirilen 3 hazır modeli kullanarak modeller eğitilecektir. Her modelin eğitimi hemen hemen aynı olsa dahi ufak tefek değişiklikler yer almaktadır.
1.3.1 Kütüphaneler
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

1. TensorFlow (tf): Açık kaynaklı bir makine öğrenimi ve derin öğrenme kütüphanesidir. Yüksek performanslı sayısal hesaplamalar için kullanılır ve genellikle derin öğrenme modellerinin oluşturulması, eğitimi ve dağıtımı için tercih edilir.
2. ImageDataGenerator (from keras.preprocessing.image): Görüntü verilerini artırmak için kullanılan bir Keras sınıfıdır. Eğitim sırasında veri çeşitliliğini artırmak amacıyla kullanılır, bu da modelin genelleştirme yeteneğini artırabilir.
3. EfficientNetB0 (from keras.applications.efficientnet): Keras kütüphanesinde bulunan ve EfficienNet mimarisine dayalı önceden eğitilmiş bir derin öğrenme modelidir. Düşük hesaplama maliyetiyle yüksek performans sağlamak amacıyla tasarlanmıştır.

4. Sequential (from keras.models): Keras'ta bulunan model oluşturma API'sinin temel sınıfıdır. Model katmanlarını sıralı bir şekilde eklemek için kullanılır.

5. Dense (from keras.layers): Tam bağlantılı (fully connected) katmanı oluşturmak için kullanılır. Bu katman, girişten çıkışa tüm nöronları içerir.

6. GlobalAveragePooling2D (from keras.layers): 2D veri kümesinde küresel ortalama havuzlama işlemi gerçekleştiren bir katmandır. Genellikle önceki katmanlardan gelen özellik haritalarını düzleştirmek için kullanılır.

7. Adam (from keras.optimizers): Stokastik gradyan iniş optimizasyon algoritmasıdır. Modelin eğitimi sırasında ağırlıkları güncellemek için kullanılır.

8. EarlyStopping (from keras.callbacks): Eğitim sırasında modelin performansını izlemek ve belirli bir kriterin karşılandığında eğitimi otomatik olarak durdurmak için kullanılır.

9. Matplotlib.pyplot (as plt): Görselleştirmeler oluşturmak için kullanılan bir çizim kütüphanesidir. Grafikler, tablolar ve diğer görsel öğeleri oluşturmak ve görüntülemek amacıyla kullanılır.

10. NumPy (as np): Sayısal hesaplamalar için kullanılan bir Python kütüphanesidir. Özellikle çok boyutlu diziler ve matrisler üzerinde etkili operasyonlar sunar.

11. Scikit-learn.metrics (from sklearn.metrics): Scikit-learn kütüphanesinde bulunan metrikler arasında yer alan confusion_matrix ve classification_report, modelin performansını değerlendirmek için kullanılır.

12. Seaborn (as sns): Matplotlib tabanlı bir veri görselleştirme kütüphanesidir. Özellikle istatistiksel grafikler oluşturmak ve veri keşfi için kullanılır.


1.3.2 Konfigürasyonları Belirleme
img_size = (224, 224)
train_path = 'train'
valid_path = 'valid'
test_path = 'test'

Bu satırlarda modelin eğitimi için train, validation ve test veri setlerinin yolları verildi. Image size 224, 244 ayarlandı çünkü kullanılacak model bu boyutlarda eğitim yaptığından sonuçlar daha yüksek olacaktır. Bir başka boyut olan 299,299 da kullanılabilirdi.
1.3.3 Veri Ön İşleme ve Veri Çoklama
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

Bu kısım da veri artırma ve ön işleme işlemleri için kullanılacak ImageDataGenerator nesnelerinin oluşturulmasıdır. Bu nesneler, görüntü verilerini çeşitli yöntemlerle değiştirerek eğitim sırasında modelin daha iyi genelleme yapabilmesine ve aşırı uydurmayı önlemeye yardımcı olur. Ayrıca, preprocess_input fonksiyonu aracılığıyla görüntülerin önceden işlenmesi de bu aşamada gerçekleştirilir.
ImageDataGenerator, Keras kütüphanesinde bulunan bir sınıftır ve görüntü verilerini çeşitli şekillerde değiştirerek modelin daha iyi öğrenmesini sağlar.
preprocessing_function parametresi, görüntü verilerini önceden işlemek için kullanılmasını belirtir. Bu, özellikle önceden eğitilmiş bir model kullanılıyorsa ve giriş görüntülerinin belirli bir şekilde ölçeklenmesi veya normalleştirilmesi gerekiyorsa önemlidir.
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical')

Bu kısım, ImageDataGenerator nesnelerini kullanarak eğitim, doğrulama ve test veri kümeleri için veri akışlarını oluşturur. Bu veri akışları, belirtilen dizinlerdeki görüntü verilerini yükleyerek, ön işleme ve veri artırma işlemlerini uygulayarak ve belirli bir batch boyutu ile verileri modelin eğitimine sağlamak üzere yapılandırılır.

flow_from_directory metodu, belirtilen dizindeki görüntü verilerini okuyarak bir veri akışı oluşturur.
train_path, eğitim veri setinin bulunduğu dizini belirtir.
target_size, hedef görüntü boyutunu belirtir.
batch_size, her eğitim iterasyonunda kullanılacak örnek sayısını belirtir.
class_mode='categorical', çoklu sınıflandırma problemi için kategorik etiketleri kullanacağımızı belirtir.
1.3.4 Modelin Tanımlanması ve Derlenmesi (EfficientNetB0):
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')  # assuming 10 classes for big cats
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.SensitivityAtSpecificity(0.5)])

Bu kısımda, EfficientNetB0 mimarisini kullanarak bir transfer öğrenme modeli oluşturulmaktadır.
EfficientNetB0 Modeli: İlk olarak, önceden eğitilmiş EfficientNetB0 mimarisini kullanarak bir temel model (base_model) oluşturulur. Bu model, ImageNet veri kümesi üzerinde eğitilmiş ağırlıkları içerir ve giriş görüntülerin boyutunu (input_shape) belirleyebilirsiniz.
Sequential Modeli: Ardından, bir Sequential modeli oluşturulur. Bu model, sıralı olarak katmanları eklemenizi sağlar.
base_model: İlk katman olarak önceden oluşturulan EfficientNetB0 modeli eklenir. Bu, transfer öğrenme prensibini kullanarak önceden eğitilmiş ağırlıkları almanızı sağlar.
GlobalAveragePooling2D(): Bu katman, önceki katmandan gelen çıktıları alır ve her özellik haritasının ortalamasını alarak düz bir vektöre dönüştürür. Bu, modelin daha önceki özellikleri öğrenmesine yardımcı olabilir ve genel birleşik özellik temsilini sağlar.
Dense(1024, activation='relu'): 1024 nörona sahip bir gizli yoğun katman eklenir. Bu katman, önceki katmanlardan gelen özellikleri daha karmaşık bir temsilasyona dönüştürmeye çalışır.
Dense(10, activation='softmax'): Çıkış katmanı olarak eklenen bu yoğun katman, 10 sınıflı bir çıkış üretir. Bu örnekte, büyük kediler için 10 sınıflı bir sınıflandırma modeli oluşturulmuş gibi görünüyor. Eğer farklı bir sınıflandırma problemiyle uğraşıyorsanız, çıkış katmanındaki nöron sayısını ve aktivasyon fonksiyonunu ayarlamalısınız.
model.compile(...): Modelin derleme aşamasıdır. Bu aşamada, optimize edici (optimizer), kayıp fonksiyonu (loss function) ve metrikler belirlenir. Bu örnekte, Adam optimizer, cross-entropy kayıp fonksiyonu ve çeşitli metrikler (accuracy, precision, recall, sensitivityAtSpecificity) kullanılmıştır.
1.3.5 Eğitim için Geri Çağırımların Tanımlanması
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Check the value of a custom metric (e.g., accuracy)
        if logs.get('val_accuracy') > 0.989:
            print("\nReached 98% validation accuracy. Stopping training.")
            self.model.stop_training = True

custom_callback = CustomCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

1.3.6 Modelin Eğitilmesi
history = model.fit(train_generator, validation_data=valid_generator, epochs=50, callbacks=[custom_callback, early_stopping])

Bu kısımda, model.fit fonksiyonu ile model eğitilir. Epoch değeri 50 olarak verilmektedir. Model checkpoint ve erken durdurma metotlarıda çağırılmaktadır. 

1.3.7 Eğitim Geçmişinin Çizdirilmesi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.show()

Bu kısımda, her epoch sırasında elde edilen train_accuracy ve validation_accuracy ‘nin görsel olarak bir grafiği oluşturulmaktadır.




1.3.8 Modelin Test Kümesinde Değerlendirilmesi
evaluation = model.evaluate(test_generator)

Bu kısımda model test verileri üzerinden sınamaya tabi tutulmaktadır ve değerlendirme metrikleri değişken içerisinde saklanmaktadır.
1.3.9 Değerlendirme Metriklerinin Gösterilmesi
print("\nTest Set Metrics:")
print(f"Loss: {evaluation[0]}")
print(f"Accuracy: {evaluation[1]}")
print(f"Precision: {evaluation[2]}")
print(f"Recall: {evaluation[3]}")
print(f"Sensitivity at Specificity 0.5: {evaluation[4]}")

model test edildikten sonra test sonuçlarını ekrana yazdırıyoruz.

1.3.9.1 Karışıklık Matrisi ve Sınıflandırma Raporu
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

class_labels = list(test_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

Karışıklık matrisini ve sınıflandırma raporunu yazdırılmaktadır. Karışıklık matrisi görsel bir grafik olarak ekranda gösterilmektedir

1.3.9.2 Eğitilmiş Modelin ve Ağırlıkların Kaydedilmesi
model.save('big_cats_efficientnetB0_model(224x224).h5')
model.save_weights('big_cats_efficientnetB0_model_weights(224x224).h5')

Modeller en iyi ağırlıklarla kaydedilmektedir
1.4 Xception ve Resnet eğitimleri
Bu eğitimlerde diğerinden farksızdır lakin içeri aktardığımız kütüphanelerden Xception ve Resnet için uygun olanları değiştireceğiz
Xception için,
from keras.applications.xception import Xception, preprocess_input

Resnet için, 
from keras.applications.resnet50 import resnet50, preprocess_input

1.5 Custom CNN
input_img = Input(shape=(224, 224, 3))
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

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)  # Assuming 10 classes for big cats

model = Model(inputs=input_img, outputs=output)

# Compile the Model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

Giriş Katmanı:
Input fonksiyonu ile bir giriş katmanı oluşturulur. Giriş boyutu 224x224 piksel ve 3 renk kanalından oluşan bir görüntüdür.

Evrişim Katmanları ve Normalizasyon:
İlk evrişim katmanı, 32 filtre ve 3x3 kernel boyutu ile tanımlanır. ReLU aktivasyon fonksiyonu ve 'same' dolgusu kullanılır.
Batch Normalization, evrişim katmanının çıkışını normalize eder.
MaxPooling2D katmanı, evrişim katmanının çıkışındaki önemli özellikleri vurgulamak ve boyutu küçültmek için kullanılır.
Bu işlemler, 64, 128, ve 256 filtre sayısına sahip üç ayrı evrişim katmanı için tekrarlanır.
Dropout Katmanı:
Bir Dropout katmanı, aşırı uydurmayı önlemek için kullanılır. Bu katman, rastgele olarak belirtilen yüzdeyi evrişim katmanındaki birimlerden atar.
Tam Bağlantılı (Dense) Katmanlar:
512 filtre sayısına sahip bir evrişim katmanı eklenir ve yine Batch Normalization ve MaxPooling2D katmanları uygulanır.
Flatten fonksiyonu, evrişim ve havuzlama katmanlarının çıkışını düzleştirir.
Birinci tam bağlantılı katman 1024 nörona sahiptir ve ReLU aktivasyon fonksiyonunu kullanır.
Bir Dropout katmanı, aşırı uydurmayı önlemek için eklenir.
Çıkış Katmanı:
Bir ikinci tam bağlantılı katman, 10 nörona sahiptir ve softmax aktivasyon fonksiyonunu kullanarak 10 farklı büyük kedi sınıfının olasılıklarını üretir.

      2.0 Uygulama Geliştirme Ortamı
Uygulama geliştirme ortamı olarak PYQT5 kütüphanesinden ve QTdesignerdan yararlanılmaktadır. Eğitim ve Test olarak iki sekmede yapılmaktadır.

2.1 Test
import sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

Bu kısımda PYQT5 ile gerekli kütüphaneleri yüklüyoruz

2.1.1 Arayüz
class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Resim Seçici')

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap(''))

        self.select_button = QPushButton('Resim Seç', self)
        self.select_button.clicked.connect(self.open_image)

        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(["Xception", "EfficientNetB0","Resnet","CNN"])

        self.analyze_button = QPushButton('Resmi Analiz Et', self)
        self.analyze_button.clicked.connect(self.analyze_image)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.model_combobox)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

Bu kısımda, Test işlemi için bilgisayardan resim seçilecek şekilde  gerekli butonlar ve modellerin bulunduğu dropdown componenti ekleniyor
2.1.2 Resim seçme metodu
def open_image(self):
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(self, 'Resim Seç', '', 'Image Files (*.png;*.jpg;*.jpeg;*.gif;*.bmp)')

    if file_path:
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaledToWidth(300)
        self.image_label.setPixmap(pixmap)
        self.image_path = file_path


2.1.3 Test İşleminin gerçekleştiği metotlar
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
            model_path = 'resnet_model.h5'
            model = load_model(model_path)

            img = image.load_img(self.image_path,
                                 target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            self.perform_predictions(img_array, model)

        elif selected_model == "CNN":
            model_path = "final_cnn_model (1).h5"
            model = load_model(model_path)
            self.pred_model(model, None)

        else:
            print("Geçersiz model seçimi.")

def perform_predictions(self, img_array, model):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    class_labels = ["African Leopard", "Caracal", "Cheetah", "Clouded Leopard", "Jaguar", "Lions", "Ocelot",
                    "Puma", "Snow Leopard", "Tiger"]
    predicted_class_label = class_labels[predicted_class_index]
    result_text = f"Predicted Class Label: {predicted_class_label}"
    self.result_label.setText(result_text)

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

analyze_image(self):
Bu metod, nesnenin 'image_path' adlı bir özelliğe sahip olup olmadığını kontrol eder.
Bir combo box'tan şu anda seçilen modeli alır.
Seçilen modele göre belirli bir dosyadan önceden eğitilmiş modeli yükler.
Ardından, yüklenen modeli ve uygun ön işleme işlevini kullanarak pred_model metodunu çağırır.

pred_model(self, model, preprocess_func):
Bu metod, bir model ve ön işleme işlevini girdi olarak alır.
Seçilen resmi yükler ve belirtilen ön işleme işlemlerini gerçekleştirir.
Ardından, perform_predictions metodunu çağırarak tahminleri gerçekleştirir.
Herhangi bir hata durumunda bir hata mesajı yazdırır.

perform_predictions(self, img_array, model):
Bu metod, bir resim dizisi ve önceden eğitilmiş bir modeli girdi olarak alır.
Modeli kullanarak verilen resim için sınıf olasılıklarını tahmin eder.
Tahmin edilen sınıf olasılıklarına dayanarak, tahmin edilen sınıf etiketini belirler.
Sonucu ekrana basar veya başka bir işleme yönlendirir.

2.1.4 Hibrit Test İşlemi
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

her modelin tahmin değerleri alınmaktadır ardından bu tahmin değerlerinin ortalaması alınarak nihai sonuç verilmektedir.


2.1.5 Başlatma şablonu
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSelector()
    window.show()
    sys.exit(app.exec_())

Arayüzü çalıştırmak için başlatma şablonu.

2.2 Eğitim
Bu kısımda, gerekli arayüz ayarlanmaları yapıldıktan sonra ilgili butonlara ilgili fonksiyonlar bağlanmaktadır.

Veri çoklama 
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


Train buttonu
def on_train_click(self):
    selected_model = self.model_selector.currentText()
    epochs = int(self.epoch_input.text())
    batch_size = int(self.batch_input.text())
    patience = int(self.patience_input.text())

    threading.Thread(target=self.train_model, args=(selected_model, epochs, batch_size, patience),
                     daemon=True).start()

2.3 Eğitim Ve Test Birleştirilmesi
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from pyqt import ImageSelector  # Assuming the test UI is in a file named test_ui.py
from train_metrics import MLTrainerGUI  # Assuming the train UI is in a file named train_ui.py
from data_aug import ImageSelector
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
        data_aug = ImageSelector()
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

ilgili kodları referans olarak verildikten sonra gerekli sekmeler ve referanlar verilmektedir. Hem Test hem de Train tek bir arayüz ekranında bulunmaktadır.

2.4 Test ve Eğitim Arayüzü
 
 


  			3.0 Test Sonuçları
3.1 Xception ve Efficient
  





3.2 Resnet
  




3.3 Custom CNN
   




4.0 Veri seti ve modeller
https://drive.google.com/file/d/1eD5I24a2UXXsx6VOum-5N9U3ChBdbdx1/view?usp=drive_link
