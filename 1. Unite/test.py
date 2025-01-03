import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# 1. Modeli Yükle
model = load_model('mnist_advanced_model.h5')
print("Model başarıyla yüklendi.")

# 2. Test Verisini Yükle
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 3. Veriyi Normalize Et
x_test = x_test / 255.0

# 4. Rastgele 10 Test Görseli Seç
random_indices = np.random.choice(len(x_test), 10)
test_images = x_test[random_indices]
test_labels = y_test[random_indices]

# 5. Modeli Kullanarak Tahmin Yap
predictions = model.predict(test_images)

# 6. Tahminleri Göster
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f'Tahmin: {np.argmax(predictions[i])}\nGerçek: {test_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()