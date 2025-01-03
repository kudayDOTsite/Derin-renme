import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Modeli Yükle ---
model_path = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/cnn_model.h5'
model = load_model(model_path)
print("Model başarıyla yüklendi!")

# --- Test Veri Setini Yükle ---
test_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/test'
batch_size = 16

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(64, 64),
    batch_size=batch_size,
    label_mode='int'
)

# --- Veri Setindeki Sınıf İsimlerini Al ---
class_names = test_dataset.class_names
print("Sınıflar:", class_names)

# --- Test Görsellerini Al ---
test_images, test_labels = next(iter(test_dataset))

# --- Rastgele Test Görselleri Seç (Mevcut Görsellerle Sınırlı) ---
num_samples = min(9, len(test_images))  # Test verisi sayısını aşma!
indices = random.sample(range(len(test_images)), num_samples)
selected_images = tf.gather(test_images, indices)
selected_labels = tf.gather(test_labels, indices)

# --- Tahmin Yap ---
predictions = model.predict(selected_images)
predicted_labels = np.argmax(predictions, axis=1)

# --- Sonuçları Görselleştir ---
plt.figure(figsize=(10, 10))
for i in range(num_samples):
    plt.subplot(3, 3, i + 1)
    plt.imshow(selected_images[i].numpy().astype("uint8"))
    plt.title(f"Tahmin: {class_names[predicted_labels[i]]}\nGerçek: {class_names[selected_labels[i]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()