import os
from PIL import Image

# Ana dizin yolu (dataset dizininin bulunduğu yer)
base_dir = '/Users/kuday/Documents/AI/Derin Öğrenme/2. ünite/dataset'  # Kendi dizininizi buraya yazın

# İşlem yapılacak alt klasörler (train ve test)
sub_dirs = ['train', 'test']

# İsimlendirme ve boyutlandırma fonksiyonu
def resize_and_rename_images(base_dir, sub_dirs, target_size=(64, 64)):
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        
        # train/ veya test/ içindeki alt klasörler (beyblade, yoyo, kapak vb.)
        for class_dir in os.listdir(sub_dir_path):
            class_dir_path = os.path.join(sub_dir_path, class_dir)
            
            # Dosyaları sırayla al ve yeniden adlandır
            if os.path.isdir(class_dir_path):
                files = sorted(os.listdir(class_dir_path))  # Dosyaları sıralı al
                for idx, file in enumerate(files):
                    # Dosya uzantısını koruyarak yeni isim oluştur
                    ext = file.split('.')[-1]  # Dosya uzantısını al (.jpeg)
                    new_name = f"{class_dir}_{idx + 1:03d}.{ext}"  # Örn: beyblade_001.jpeg
                    
                    # Eski ve yeni dosya yolunu tanımla
                    old_path = os.path.join(class_dir_path, file)
                    new_path = os.path.join(class_dir_path, new_name)
                    
                    try:
                        # Resmi aç ve yeniden boyutlandır
                        img = Image.open(old_path)
                        img = img.resize(target_size, Image.Resampling.LANCZOS)  # Yüksek kaliteli yeniden boyutlandırma
                        img.save(new_path)  # Yeniden kaydet
                        
                        # Dosyayı yeniden adlandır
                        print(f"İşlendi ve Kaydedildi: {old_path} -> {new_path}")
                        
                    except Exception as e:
                        print(f"Hata oluştu: {old_path} - {str(e)}")

# İsimlendirme ve boyutlandırmayı başlat
resize_and_rename_images(base_dir, sub_dirs)