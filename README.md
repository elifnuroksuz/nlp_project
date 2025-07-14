NLP Metin Sınıflandırma Projesi 🎬
Bu proje, IMDB film yorumları üzerinde duygu analizi (sentiment analysis) yaparak yorumları olumlu ve olumsuz olarak sınıflandıran kapsamlı bir doğal dil işleme (NLP) çalışmasıdır.
📋 Proje Özeti
NLTK, spaCy ve Scikit-learn kütüphaneleri kullanılarak:

Metin temizleme ve preprocessing işlemleri
Tokenizasyon ve lemmatizasyon teknikleri
Bag of Words (BoW) ve TF-IDF vektörizasyon yöntemleri
5 farklı makine öğrenmesi algoritması ile model eğitimi
Hiperparametre optimizasyonu ve çapraz doğrulama
Performans karşılaştırması ve model seçimi

🎯 Ana Hedefler

Film yorumlarını otomatik olarak olumlu/olumsuz sınıflandırmak
Farklı vektörizasyon yöntemlerinin performansını karşılaştırmak
En iyi makine öğrenmesi algoritmasını belirlemek
Metin madenciliği sürecinin tüm aşamalarını uygulamak

🏆 Başarılan Sonuçlar
En İyi Model Performansı:

Algoritma: Lojistik Regresyon + TF-IDF
F1-Score: 0.8812 (88.12%)
Accuracy: 89.57%
Veri Seti: 50,000 IMDB film yorumu

Vektörizasyon Karşılaştırması:

TF-IDF: F1-Score 0.8812 ⭐
Bag of Words: F1-Score 0.5100

🔧 Kullanılan Teknolojiler
Kütüphaneler:

NLTK: Tokenizasyon, stemming, stop words
spaCy: Lemmatizasyon, POS tagging
Scikit-learn: Vektörizasyon, ML algoritmaları, metrikler
Pandas: Veri manipülasyonu
NumPy: Sayısal hesaplamalar
Matplotlib & Seaborn: Görselleştirme
BeautifulSoup: HTML temizleme

Makine Öğrenmesi Algoritmaları:

Lojistik Regresyon (En İyi: F1=0.8812)
Random Forest (F1=0.8427)
Naive Bayes (F1=0.8543)
Karar Ağacı (F1=0.7323)
KNN (F1=0.5550)

📁 Proje Yapısı
nlp-metin-siniflandirma/
├── 01_Kurulum_ve_veri_hazırlığı.ipynb
├── 02_metin_temizleme.ipynb
├── 03_tokenization_lemmatization.ipynb
├── 04_BagofWordsveTF-IDFvektörizasyonu.ipynb
├── 05_metin_sınıflandırma_modeli_train_test.ipynb
├── 06_metin_sınıflandırma_modeli_cross_validaton.ipynb
├── README.md
└── models/ (kaydedilen modeller)
🚀 Proje Adımları
ADIM 1: Kurulum ve Veri Hazırlığı

Veri setini anlama ve keşifsel analiz
Sınıf dengesi kontrolü
Metin uzunlukları analizi
Eksik veri tespiti
Görselleştirmeler

ADIM 2: Metin Temizleme

HTML etiketlerini temizleme (BeautifulSoup)
Noktalama işaretlerini kaldırma
Sayıları temizleme
Küçük harfe çevirme
Fazla boşlukları düzenleme

ADIM 3: Tokenizasyon ve Lemmatizasyon

Metin tokenizasyonu (NLTK & spaCy)
Stop words (durdurma kelimeleri) kaldırma
Lemmatizasyon (kök forme indirgeme)
Stemming işlemleri
Sık kullanılan kelimelerin görselleştirilmesi

ADIM 4: Vektörizasyon

Bag of Words (BoW) implementasyonu
TF-IDF vektörizasyonu
Sparse matrix formatında kaydetme
Kelime önem skorlarının analizi

ADIM 5: Model Eğitimi ve Test

Train/test split (%80-%20)
4 farklı model kombinasyonu eğitimi
Performans metrikleri hesaplama
Confusion matrix analizi
En iyi model seçimi

ADIM 6: Çapraz Doğrulama ve Optimizasyon

10-kat çapraz doğrulama
Hiperparametre optimizasyonu
5 algoritma ile kapsamlı karşılaştırma
Performans görselleştirmeleri
Final model kaydetme

📊 Performans Metrikleri
Projede kullanılan değerlendirme metrikleri:

Accuracy: Doğru tahmin oranı
Precision: Pozitif tahminlerin doğruluk oranı
Recall: Gerçek pozitiflerin yakalanma oranı
F1-Score: Precision ve recall'un harmonik ortalaması
MCC: Matthews Correlation Coefficient
Confusion Matrix: Sınıflandırma detay analizi

🎯 Önemli Bulgular

TF-IDF >> Bag of Words: TF-IDF vektörizasyonu, BoW'dan önemli ölçüde daha iyi performans gösterdi
Lojistik Regresyon: En kararlı ve yüksek performanslı algoritma
Metin Temizleme: Preprocessing adımları model performansını önemli ölçüde artırdı
Çapraz Doğrulama: Modelin genelleme yeteneğini doğruladı

🔮 Gelecek Geliştirmeler

Deep Learning modelleri (LSTM, BERT)
Ensemble yöntemleri
Çok sınıflı duygu analizi
Gerçek zamanlı tahmin API'si
Web arayüzü geliştirme

📈 Sonuç
Bu proje, klasik NLP tekniklerinin film yorumu duygu analizi probleminde nasıl etkili bir şekilde kullanılabileceğini göstermiştir. %89.57 doğruluk oranıyla pratik kullanıma hazır bir model geliştirilmiştir.
