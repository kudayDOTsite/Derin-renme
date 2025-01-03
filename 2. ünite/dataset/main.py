import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image, UnidentifiedImageError
import os
import matplotlib.pyplot as plt

# --- Dizinler ---
dataset_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/augmented_train'
test_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/test'

# --- Hatalı Resimleri Temizle ---
def remove_corrupt_images(directory):
    num_removed = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Bozuksa hata fırlatır
            except (IOError, UnidentifiedImageError):
                print(f"Hatalı dosya kaldırıldı: {file_path}")
                os.remove(file_path)
                num_removed += 1
    print(f"Toplam {num_removed} dosya kaldırıldı.")

remove_corrupt_images(dataset_dir)
remove_corrupt_images(test_dir)

batch_size = 16

# --- Veri Setini Yükle ---
train_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(64, 64),
    batch_size=batch_size,
    label_mode='int'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(64, 64),
    batch_size=batch_size,
    label_mode='int'
)

# --- Sınıfları Göster ---
class_names = train_dataset.class_names
print("Sınıflar:", class_names)

# --- Görselleştirme ---
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# --- Veri Artırma Katmanları ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# --- Normalleştirme ---
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# --- Doğrulama ve Eğitim Setini Ayır ---
val_size = int(len(train_dataset) * 0.2)
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# --- Veri Artırma ve Normalizasyon ---
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# --- Batchleme Tekrarı Yok ---
# train_dataset = train_dataset.batch(batch_size)  # BU SATIRI KALDIRIN!

# --- CNN Modeli ---
def create_cnn_model(input_shape=(64, 64, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Model Eğitimi ---
epochs = 15
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# --- Modeli Test Et ---
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Doğruluğu: {test_accuracy:.2f}")

# --- Modeli Kaydet ---
model.save('/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/cnn_model.h5')
print("Model başarıyla kaydedildi!")