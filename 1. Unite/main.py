import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Veri Setini Yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Görselleştirme
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Etiket: {y_train[0]}')
plt.show()

# 3. Veriyi Normalize Et
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. Etiketleri Kategorik Hale Getir
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 5. Modeli Kur
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 6. Modeli Derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. Modeli Eğit
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 8. Modeli Test Et
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test doğruluğu: {accuracy:.2f}')

# 9. Modeli Kaydet
model.save('mnist_advanced_model.h5')
print(f'Model kaydedildi!')