# 🔢 MNIST Rakam Sınıflandırma (PyTorch ANN)

Bu proje, PyTorch kütüphanesi kullanılarak geliştirilmiş bir Yapay Sinir Ağı (ANN) modelidir. Projenin amacı, el yazısı rakamlardan oluşan ünlü **MNIST** veri setini kullanarak 0 ile 9 arasındaki rakamları yüksek doğrulukla tanımaktır.

## 🚀 Proje Özellikleri
- **Veri Seti:** MNIST (60,000 Eğitim, 10,000 Test örneği).
- **Model Mimarisi:** 3 katmanlı Tam Bağlantılı (Fully Connected) Sinir Ağı.
- **Donanım:** CUDA (GPU) desteği (mevcutsa otomatik kullanılır).
- **Görselleştirme:** Eğitim öncesi veri örnekleri ve eğitim sonrası Loss (kayıp) grafiği.
- **Normalizasyon:** Veriler Tensor formatına çevrilerek mean=0.5, std=0.5 değerleri ile normalize edilmiştir.

## 🛠️ Kurulum ve Çalıştırma

Projenin çalışması için sisteminizde Python yüklü olmalıdır. Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:
###  Bağımlılıkları İndirme
Terminali (veya komut satırını) açın ve projenin bulunduğu klasöre giderek aşağıdaki komutu çalıştırın:

```bash
# Gerekli tüm kütüphaneleri  tek seferde yükler:
pip install -r requirements.txt
````
### Projeyi Çalıştırın:
Terminal üzerinden ana dosyanızı  şu komutla başlatın:

```bash
python main.py
```
