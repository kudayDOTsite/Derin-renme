import os
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- Dizinler ---
dataset_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/train'
augmented_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset/augmented_train'

# --- Hedef Boyut ---
target_size = (64, 64)  # Resimler 64x64 boyutuna ayarlanacak

# --- Augmentation Parametreleri ---
datagen = ImageDataGenerator(
    rotation_range=30,  # Döndürme
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    shear_range=0.2,  # Kesme
    zoom_range=0.3,  # Yakınlaştırma/Uzaklaştırma
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest'  # Boş alanları doldurma
)


# --- Resim Boyutlandırma ve Yeniden Adlandırma ---
def resize_and_rename_images(base_dir, target_size):
    for sub_dir in ['train', 'test']:
        sub_dir_path = os.path.join(base_dir, sub_dir)

        for class_name in os.listdir(sub_dir_path):
            class_dir = os.path.join(sub_dir_path, class_name)

            # Klasör varsa işleme başla
            if not os.path.isdir(class_dir):
                continue

            print(f"İşlenen sınıf: {class_name}")
            files = sorted(os.listdir(class_dir))

            for idx, file in enumerate(files):
                old_path = os.path.join(class_dir, file)
                ext = file.split('.')[-1]  # Dosya uzantısını al
                new_name = f"{class_name}_{idx + 1:03d}.{ext}"  # beyblade_001.jpeg
                new_path = os.path.join(class_dir, new_name)

                try:
                    img = Image.open(old_path)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Yüksek kaliteli yeniden boyutlandırma
                    img.save(new_path)
                    
                    # Eski dosyayı kaldır (yeniden adlandırma)
                    if old_path != new_path:
                        os.remove(old_path)

                    print(f"Kaydedildi: {new_name}")

                except (IOError, UnidentifiedImageError) as e:
                    print(f"Bozuk dosya kaldırıldı: {old_path}")
                    os.remove(old_path)


# --- Augmentation (Veri Artırma) ve Kaydetme ---
def augment_images(dataset_dir, augmented_dir):
    # Her sınıf için augmentation yap
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Augmented verileri kaydetmek için dizin oluştur
        augmented_class_dir = os.path.join(augmented_dir, class_name)
        os.makedirs(augmented_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path)  # Görseli yükle
            x = img_to_array(img)  # Tensöre çevir
            x = x.reshape((1,) + x.shape)  # Batch formatına sok

            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=augmented_class_dir,
                save_prefix='aug',
                save_format='jpeg'
            ):
                i += 1
                if i > 40:  # Her resim için 40 varyasyon oluştur
                    break
            print(f"Augmentation tamamlandı: {img_name} için {i} yeni varyasyon")


# --- Bozuk Resimleri Kontrol Et ve Kaldır ---
def check_and_remove_corrupt_images(directory):
    num_removed = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Resmi doğrula
            except (IOError, UnidentifiedImageError) as e:
                print(f"Bozuk dosya kaldırılıyor: {file_path}")
                os.remove(file_path)
                num_removed += 1
    print(f"Toplam {num_removed} bozuk dosya kaldırıldı.")


# --- KODU ÇALIŞTIR ---
if __name__ == "__main__":
    # Bozuk resimleri temizle
    check_and_remove_corrupt_images(dataset_dir)

    # Resimleri boyutlandır ve yeniden adlandır
    resize_and_rename_images('/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset', target_size)

    # Veri artırma yap ve disk üzerinde kaydet
    augment_images(dataset_dir, augmented_dir)