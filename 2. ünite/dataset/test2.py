import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Modeli Yükle ---
model_path = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/cnn_model.h5'
model = load_model(model_path)
print("Model başarıyla yüklendi!")

# --- Test Edilecek Fotoğrafın Yolu (Manuel Olarak Verin) ---
img_path = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/IMG_1046.jpeg'  # iPhone ile çekilen fotoğraf yolu

# --- Fotoğrafı Yükle ve Yeniden Boyutlandır ---
def preprocess_image(img_path):
    img = Image.open(img_path)
    
    # Fotoğrafı 64x64 boyutuna küçült
    img = img.resize((64, 64))
    
    # Modelin beklediği formata dönüştür (numpy array)
    img_array = image.img_to_array(img)
    
    # Batch formatına dönüştür
    img_array = np.expand_dims(img_array, axis=0)
    
    # 0-255 aralığındaki piksel değerlerini 0-1 arasına getir (normalize et)
    img_array = img_array / 255.0
    
    return img_array

# --- Fotoğrafı Ön İşlemden Geçir ---
input_image = preprocess_image(img_path)

# --- Tahmin Yap ---
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction)

# --- Sınıf İsimlerini Yükle ---
class_names = ['beyblade', 'kapak', 'yoyo']  # Modelin eğitildiği sınıflar
predicted_label = class_names[predicted_class]

# --- Sonucu Göster ---
print(f"Tahmin Edilen Sınıf: {predicted_label}")