# nlp_project
NLTK, spaCy ve Scikit-learn kullanılarak metin temizleme, tokenizasyon, vektörleştirme ve basit metin sınıflandırma modelleri uygulanmıştır.

ADIM 1 -> 01_Kurulum_ve_veri_hazırlığı:
Amaç: Veri setini anlamak, sınıfların dengesini kontrol etmek, metin uzunluklarını analiz etmek ve eksik verileri tespit ederek veri ön işleme için temel bir anlayış oluşturmak. Grafikler, verinin yapısını görselleştirerek daha iyi anlamayı sağladı.

Kullanılan Kütüphanler:
pandas (pd): Veri analizi ve manipülasyonu için; tablo verileriyle çalışır, filtreleme, gruplama yapar.
numpy (np): Sayısal hesaplamalar için; dizi/matris işlemleri ve matematiksel fonksiyonlar sağlar.
matplotlib.pyplot (plt): Veri görselleştirme için; grafikler (çizgi, çubuk, vb.) oluşturur.
seaborn (sns): Gelişmiş veri görselleştirme; daha estetik ve kolay grafikler çizer.
re: Düzenli ifadeler için; metin arama, eşleştirme ve düzenleme yapar.

df.shape: Veri setinin boyutunu (satır ve sütun sayısı) verir.
df.info(): Sütunların veri tipleri, eksik değerler ve bellek kullanımı hakkında özet bilgi sunar.
df.columns.tolist(): Veri setindeki sütun isimlerini liste olarak döndürür.

ADIM 2-> 02_metin_temizleme:
Amaç: Metin verilerini (yorumları) analiz için uygun hale getirmek; HTML etiketleri, noktalama işaretleri, sayılar ve fazla boşlukları temizleyerek metni sadeleştirmek ve standartlaştırmak. Görselleştirmeler, temizlemenin etkisini anlamaya yardımcı oldu. Temizlenmiş veri, sonraki adımlar (ör. makine öğrenmesi) için hazırlandı.

BeautifulSoup: web sitelerinden veri çekmek (web scraping) için kullanılır, HTML ve XML dökümanlarını parçalayıp içindeki verileri kolayca almamızı sağlar.

Metin Temizleme ve Normalizasyon:
HTML Etiketlerini Temizleme: BeautifulSoup ile yorumlardaki HTML etiketleri (<br /> gibi) kaldırıldı, review_no_html sütunu oluşturuldu.
Noktalama İşaretlerini Temizleme: re.sub ile noktalama işaretleri boşlukla değiştirildi, review_no_punct sütunu oluşturuldu.
Sayıları Temizleme: Sayılar kaldırıldı, review_no_numbers sütunu oluşturuldu.
Küçük Harfe Çevirme: Tüm metin küçük harfe çevrildi, review_lower sütunu oluşturuldu.
Fazla Boşlukları Temizleme: Fazla boşluklar kaldırılıp metin düzgünleştirildi, review_clean sütunu oluşturuldu.

ADIM 3-> 03_tokenization_lemmatization
Amaç: Metin verilerini tokenizasyon, durdurma kelimelerini kaldırma, lemmatizasyon ve stemming işlemleriyle işleyerek makine öğrenmesi modelleri için uygun hale getirmek, sık kullanılan kelimeleri görselleştirmek ve işlenmiş veriyi kaydetmek.

Kullanılan Kütüphanler:
pandas, numpy: Veri işleme.
nltk, spacy: Tokenizasyon, durdurma kelimeleri, lemmatizasyon.
sklearn: Vektörizasyon ve sınıflandırma.
matplotlib, seaborn: Görselleştirme.

1. Tokenizasyon (Tokenization): Metni daha küçük parçalara (tokenlara) ayırma işlemidir.
Amaç: Metni analiz edilebilir birimler haline getirmek, böylece kelime veya cümle bazında işleme mümkün olur.
Örnek:
Giriş: "I love watching movies!"
Çıkış (kelime tokenları): ["I", "love", "watching", "movies", "!"]

2. Lemmatizasyon (Lemmatization): Kelimeleri dilbilgisi açısından kök (lemma) formuna indirgeme işlemidir. Kelimenin anlamını ve dilbilgisi bağlamını koruyarak, farklı çekimlerini (ör. çoğul, fiil zamanları) temel haline getirir.
Amaç: Aynı anlama gelen farklı kelime formlarını (ör. "running", "ran" → "run") birleştirerek analizde tutarlılık sağlamak.
Örnek:
Giriş: "running", "ran", "runs"
Çıkış: "run"

3. Stemming: Kelimeleri köklerine indirgeme işlemidir, ancak lemmatizasyondan farklı olarak dilbilgisi bağlamını dikkate almaz; sadece ekleri keser (daha kaba bir yöntem).
Amaç: Kelime varyasyonlarını azaltmak, ancak bazen anlamsız kökler üretir (ör. "better" → "bet").
Örnek:
Giriş: "running", "runner"
Çıkış: "run"

NLTK: Python tabanlı, doğal dil işleme (NLP) için açık kaynaklı bir kütüphanedir.
Esnek, hafif, akademik ve öğrenme amaçlı projeler için ideal. Ancak manuel ayar (ör. POS etiketleme) gerektirebilir ve bazı işlemlerde daha az doğruluk sunar.

spaCy:Modern, endüstri odaklı bir NLP kütüphanesidir.
Hızlı, bağlam duyarlı, büyük dil modelleriyle çalışır, ancak daha fazla bellek kullanır. Stemming yerine lemmatizasyona odaklanır.


ADIM 4 -> 04_BagofWordsveTF-IDFvektörizasyonu
Amaç:Metin verilerini (yorumları) sayısal vektörlere çevirerek makine öğrenmesi modelleri (ör. duygu analizi) için hazır hale getirmek, kelime önemlerini görselleştirmek ve bellek dostu bir formatta (sparse) kaydetmek.

BoW (Bag of Words): Metni, her kelimenin dokümandaki sıklığına (frekansına) dayalı olarak temsil eden basit bir vektörleştirme yöntemidir. Her doküman, kelime dağarcığındaki tüm kelimeler için bir vektörle ifade edilir; kelime varsa sıklığı yazılır, yoksa sıfır konur.
Örnek:
Doküman: "I love movies"
Kelime dağarcığı: [I, love, movies, hate]
BoW vektörü: [1, 1, 1, 0] (her kelimenin sıklığı).
(+) Avantajlar: Basit, hızlı, anlaşılır.
(-) Dezavantajlar: Kelime bağlamını/önemini dikkate almaz, sık kullanılan anlamsız kelimeler (ör. "the") ağırlık kazanabilir, vektörler büyük ve seyrek (sparse) olur.

TF-IDF (Term Frequency-Inverse Document Frequency):Kelimelerin önemini hem dokümandaki sıklığına (TF) hem de tüm dokümanlar arasında ne kadar nadir olduğuna (IDF) göre ölçen bir vektörleştirme yöntemidir. Nadir kelimeler daha yüksek ağırlık alır.
TF (Terim Frekansı): Bir kelimenin bir dokümanda kaç kez geçtiği (sıklık).
IDF (Ters Doküman Frekansı): Kelimenin tüm dokümanlarda ne kadar nadir olduğunu ölçer (nadir kelimeler daha önemlidir).
Örnek:
"movie" sık geçer, düşük IDF (düşük önem).
"masterpiece" az geçer, yüksek IDF (yüksek önem).
Doküman: "This movie is a masterpiece" → "masterpiece" daha yüksek TF-IDF skoru alır.
(+) Avantajlar: Anlamsız sık kelimelerin (ör. "the") etkisini azaltır, önemli kelimeleri öne çıkarır.
(-) Dezavantajlar: Hala kelime sırası/bağlamı dikkate almaz, hesaplama BoW’a göre daha karmaşıktır.

ADIM 5-> 05_metin_sınıflandırma_modeli_train_test
AmaçIMDB veri setindeki film yorumlarını (olumlu/olumsuz) sınıflandırmak için BoW (Bag of Words) ve TF-IDF vektörleriyle iki farklı makine öğrenmesi modeli (Naive Bayes ve Lojistik Regresyon) eğitmek, performanslarını karşılaştırmak, en iyi modeli seçmek ve gelecekteki kullanımlar için kaydetmek. Ayrıca, model performanslarını görselleştirerek analiz etmek ve sonuçları düzenli bir şekilde kaydetmek.

Veriyi Eğitim ve Test Setlerine Ayırma:
İşlem: BoW ve TF-IDF vektörleri, %80 eğitim (40,000 örnek) ve %20 test (10,000 örnek) olacak şekilde ayrıldı (train_test_split, random_state=42, stratify=y ile dengeli bölme).
Detaylar: Sınıf dengesi korundu (eğitim: 20,000 olumlu/olumsuz; test: 5,000 olumlu/olumsuz).

Modelleri Eğitme:
İşlem: Dört model eğitildi:
Naive Bayes (BoW)
Lojistik Regresyon (BoW)
Naive Bayes (TF-IDF)
Lojistik Regresyon (TF-IDF)

Performans Metrikleri:
Doğruluk (Accuracy): Doğru tahminlerin oranı (ör. 0.8957).
Hassasiyet (Precision): Pozitif tahminlerin ne kadarının doğru olduğu.
Geri Çağırma (Recall): Gerçek pozitiflerin ne kadarının doğru tahmin edildiği.
F1 Skoru: Precision ve recall’un harmonik ortalaması; dengeli bir performans ölçüsü.
Karışıklık Matrisi (Confusion Matrix): Gerçek ve tahmin edilen sınıfların karşılaştırması (ör. Negatif/Pozitif).

Modelleri Değerlendirme:
İşlem: Her modelin performansı, doğruluk (accuracy), hassasiyet (precision), geri çağırma (recall) ve F1 skoru ile değerlendirildi (classification_report).
Detaylar:
Naive Bayes (BoW): Accuracy 0.8577, F1 0.8577
Lojistik Regresyon (BoW): Accuracy 0.8851, F1 0.8851
Naive Bayes (TF-IDF): Accuracy 0.8605, F1 0.8605
Lojistik Regresyon (TF-IDF): Accuracy 0.8957, F1 0.8957
TF-IDF modelleri, BoW’a göre daha iyi performans gösterdi; Lojistik Regresyon, Naive Bayes’ten daha başarılı.


ADIM 6-> 06_metin_sınıflandırma_modeli_cross_validaton 
Amaç: IMDB veri seti kullanılarak film yorumları üzerinde duygu analizi (olumlu/olumsuz) yapıldı. Metinler TF-IDF ve Bag of Words (BoW) yöntemleriyle vektörleştirildi, 20.000 örnekten oluşan bir veri seti ile beş makine öğrenmesi modeli (KNN, Naive Bayes, Karar Ağacı, Lojistik Regresyon, Random Forest) eğitildi. Hiperparametre optimizasyonu ve 10-kat çapraz doğrulama ile modellerin performansı değerlendirildi. Lojistik Regresyon, TF-IDF ile en iyi performansı (F1-score: 0.8812) gösterdi. Sonuçlar görselleştirildi, modeller ve vektörler kaydedildi. TF-IDF, BoW'dan (F1: 0.5100) daha iyi sonuç verdi, bu nedenle duygu analizi için önerildi.

Makine Öğrenmesi Algoritmaları
KNN (K-En Yakın Komşu): Yeni bir örneği, en yakın komşularına bakarak sınıflandırır. Ancak metin verilerinde genellikle düşük performans gösterir (F1: 0.5550).
Naive Bayes: Olasılıksal bir modeldir ve metin sınıflandırmada (özellikle BoW ile) etkilidir. TF-IDF ile iyi performans gösteriyor (F1: 0.8543).
Karar Ağacı: Veriyi dallara ayırarak kararlar alır. Orta düzey performans (F1: 0.7323).
Lojistik Regresyon: İkili sınıflandırma için optimize edilmiş, doğrusal bir model. En iyi performansı gösteriyor (F1: 0.8812).
Random Forest: Birden fazla karar ağacının birleşimi. İyi performans gösteriyor (F1: 0.8427) ancak Lojistik Regresyon'un gerisinde.

Hiperparametre Optimizasyonu: Bir makine öğrenmesi modelinin performansını artırmak için en iyi parametre kombinasyonlarını bulma işlemi.

Çapraz Doğrulama (Cross-Validation): Modelin genelleme yeteneğini değerlendirmek için veri setini birden fazla alt kümeye bölerek test etme yöntemi.

Performans Metrikleri
Tanım: Modelin başarısını ölçmek için kullanılan ölçütler.
Koddaki Yeri:
Doğruluk (Accuracy): Doğru tahminlerin oranı.
Kesinlik (Precision): Pozitif tahminlerin ne kadarının doğru olduğu.
Duyarlılık (Recall): Gerçek pozitiflerin ne kadarının doğru tahmin edildiği.
Özgüllük (Specificity): Gerçek negatiflerin ne kadarının doğru tahmin edildiği.
F1-Score: Kesinlik ve duyarlılığın harmonik ortalaması (dengeli bir metrik).
MCC (Matthews Correlation Coefficient): Sınıflandırma performansını dengesiz veri setlerinde bile iyi ölçen bir metrik.

Karışıklık Matrisi (Confusion Matrix):  Modelin tahminlerini gerçek sınıflarla karşılaştıran bir tablo (True Positives, False Positives, vb.).
