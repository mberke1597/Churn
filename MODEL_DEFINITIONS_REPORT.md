# Customer Churn Prediction - Model Definitions Report

## Proje Özeti
Bu rapor, Customer Churn (Müşteri Kaybı) tahmin projesi kapsamında kullanılan tüm makine öğrenimi modellerinin detaylı açıklamalarını içermektedir.

**Veri Seti:** Customer-Churn-Records.csv (10,000 kayıt, 18 özellik)  
**Hedef Değişken:** Exited (Müşterinin bankadan ayrılıp ayrılmadığı)  
**Veri Ön İşleme:** StandardScaler, OneHotEncoder, RandomOverSampler

---

## 1. Naive Bayes (Gaussian)

### Tanım
Naive Bayes, Bayes teoremini temel alan olasılıksal bir sınıflandırma algoritmasıdır. "Naive" (saf) terimi, özelliklerin birbirinden bağımsız olduğu varsayımından gelir.

### Kullanılan Varyant
**Gaussian Naive Bayes:** Sürekli değişkenler için kullanılır ve özelliklerin normal dağılıma sahip olduğunu varsayar.

### Parametreler
```python
GaussianNB()
```
- Varsayılan parametreler kullanılmıştır

### Çalışma Prensibi
1. Her sınıf için özelliklerin olasılık dağılımını hesaplar
2. Yeni bir veri geldiğinde, her sınıf için posterior olasılığı hesaplar
3. En yüksek olasılığa sahip sınıfı tahmin olarak verir

### Avantajları
- Hızlı eğitim ve tahmin
- Az veri ile iyi performans
- Çok sınıflı problemlerde etkili

### Dezavantajları
- Özellikler arası bağımsızlık varsayımı gerçekçi olmayabilir
- Sürekli değişkenlerin normal dağıldığını varsayar

### Proje Sonuçları
- **Accuracy:** %70.46
- Pozitif sınıf (Churn) tahmininde tatmin edici performans
- Negatif sınıf tahmininde orta seviye başarı

---

## 2. Decision Tree (Karar Ağacı)

### Tanım
Decision Tree, verileri ardışık sorular sorarak dallandıran ve her bir dal sonunda karar veren ağaç yapısında bir algoritmadır.

### Kullanılan Parametreler
```python
DecisionTreeClassifier(
    criterion='entropy',      # Bilgi kazancı ölçütü
    min_samples_split=2,      # Bir düğümün bölünmesi için gereken minimum örnek sayısı
    max_depth=11,             # Ağacın maksimum derinliği
    random_state=0
)
```

### Hiperparametre Optimizasyonu
RandomizedSearchCV ile en iyi parametreler bulundu:
- **Denenen max_depth:** [3, 4, 5, 6, 7, 9, 11]
- **Denenen min_samples_split:** [2, 3, 4, 5, 6, 7]
- **Denenen criterion:** ['entropy', 'gini']

### Çalışma Prensibi
1. En iyi özelliği seçer (entropy veya gini kullanarak)
2. Bu özelliğe göre veriyi böler
3. Her alt küme için aynı işlemi tekrar eder
4. Durdurma kriteri (max_depth, min_samples_split) karşılanana kadar devam eder

### Özellik Önemleri
En önemli özellikler hesaplandı ve feature_imp değişkeninde saklandı.

### Avantajları
- Yorumlanabilir ve görselleştirilebilir
- Hem kategorik hem sürekli değişkenlerle çalışır
- Veri ön işleme gerektirmez

### Dezavantajları
- Overfitting'e eğilimli
- Küçük veri değişikliklerine hassas
- Bias'lı olabilir

### Proje Sonuçları
- **Accuracy:** %76.63
- Naive Bayes'e göre %6 iyileşme
- Hem Churn hem Non-churn tahmininde dengeli performans

---

## 3. Random Forest (Rastgele Orman)

### Tanım
Random Forest, birçok karar ağacının birleştirildiği bir ensemble (topluluk) öğrenme yöntemidir. Her ağaç farklı veri alt kümeleri ve özellikler kullanılarak eğitilir.

### Kullanılan Parametreler
```python
RandomForestClassifier(
    n_estimators=100,         # Ağaç sayısı
    min_samples_split=7,      # Bölünme için minimum örnek
    max_depth=11,             # Maksimum derinlik
    criterion='entropy',      # Bölünme kriteri
    random_state=0
)
```

### Hiperparametre Optimizasyonu
RandomizedSearchCV ile 5-fold cross-validation kullanılarak optimize edildi.

### Çalışma Prensibi
1. Bootstrap sampling ile veri alt kümeleri oluşturur
2. Her alt küme için bir karar ağacı eğitir
3. Her ağaçta rastgele özellik seçimi yapar
4. Tahmin sırasında tüm ağaçların oylamasını alır (majority voting)

### Özellik Önemleri
feature_imp_random değişkeninde saklandı. Her özelliğin karar vermede ne kadar kullanıldığını gösterir.

### Avantajları
- Overfitting'e karşı dayanıklı
- Yüksek doğruluk oranı
- Özellik önemi çıkarabilir
- Eksik verilere toleranslı

### Dezavantajları
- Yavaş tahmin (çok sayıda ağaç)
- Bellek kullanımı yüksek
- Yorumlanması zor

### Proje Sonuçları
- **Accuracy:** %83.63
- Projede en yüksek doğruluk oranı
- Hem Churn hem Non-churn sınıflarında mükemmel performans

---

## 4. Extra Trees (Extremely Randomized Trees)

### Tanım
Extra Trees, Random Forest'a benzer bir ensemble yöntemidir ancak daha fazla rastgelelik içerir. Bootstrap yerine tüm veriyi kullanır ve bölünme noktalarını tamamen rastgele seçer.

### Kullanılan Parametreler
```python
ExtraTreesClassifier(
    n_estimators=100,
    min_samples_split=4,
    max_depth=11,
    criterion='entropy',
    random_state=0
)
```

### Random Forest'tan Farkları
1. Bootstrap sampling kullanmaz (tüm veriyi kullanır)
2. Bölünme noktalarını tamamen rastgele seçer (en iyi değil)
3. Genellikle daha hızlı eğitim

### Avantajları
- Random Forest'tan daha hızlı
- Overfitting'e daha dayanıklı
- Variance azaltma

### Dezavantajları
- Bias artabilir
- Bazen Random Forest'tan düşük doğruluk

### Proje Sonuçları
- **Accuracy:** %82.06
- Random Forest'tan biraz düşük ama yine de yüksek performans
- İyi bir alternatif model

---

## 5. K-Means Clustering

### Tanım
K-Means, gözetimsiz bir öğrenme algoritmasıdır. Verileri benzerliklerine göre K adet kümeye ayırır. Bu projede sınıflandırma probleminde kullanılmaya çalışıldı.

### Kullanılan Parametreler
```python
KMeans(
    n_clusters=2,        # Küme sayısı (Churn / Non-churn)
    random_state=0
)
```

### Çalışma Prensibi
1. K adet merkez noktası rastgele seçer
2. Her veri noktasını en yakın merkeze atar
3. Merkezleri yeniden hesaplar (küme ortalaması)
4. Merkezler değişmeyene kadar 2-3 adımlarını tekrarlar

### Neden Başarısız?
- K-Means gözetimsiz bir algoritmadır (etiket kullanmaz)
- Kümelerin sınıflarla eşleşmesi garantili değildir
- Sınıflandırma probleminde uygun değildir

### Proje Sonuçları
- **Accuracy:** %54.5
- En düşük performans
- Churn prediction için uygun değil

---

## 6. K-Nearest Neighbors (KNN)

### Tanım
KNN, yeni bir veriyi en yakın K komşusunun sınıflarına bakarak sınıflandırır. "Lazy learning" (tembel öğrenme) olarak bilinir çünkü eğitim aşaması yoktur.

### Kullanılan Parametreler
```python
KNeighborsClassifier(
    n_neighbors=1,        # Komşu sayısı (GridSearch ile bulundu)
    metric='minkowski',   # Uzaklık metriği
    p=2                   # p=2 Euclidean distance
)
```

### Hiperparametre Optimizasyonu
GridSearchCV ile k=1..9 aralığında denendi, k=1 en iyi sonucu verdi.

### Çalışma Prensibi
1. Yeni veri noktası gelir
2. Tüm eğitim verilerine uzaklık hesaplar
3. En yakın K komşuyu bulur
4. Bu komşuların çoğunluk sınıfını tahmin olarak verir

### Neden k=1 Optimal?
k=1 genellikle overfitting işaretidir, ancak:
- Veri dengeli ve yeterli büyüklükte
- StandardScaler kullanıldı
- Yine de dikkatli olunmalı

### Avantajları
- Basit ve anlaşılır
- Eğitim gerektirmez
- Çok sınıflı problemlerde etkili

### Dezavantajları
- Yavaş tahmin (tüm veriyi tararar)
- Özellik ölçeklendirmeye duyarlı
- Yüksek boyutlu veride kötü (curse of dimensionality)

### Proje Sonuçları
- **Accuracy:** %67.1
- Orta seviye performans
- Bu veri seti için ideal değil

---

## 7. Logistic Regression (Lojistik Regresyon)

### Tanım
Logistic Regression, sınıflandırma problemleri için kullanılan doğrusal bir modeldir. İkili sınıflandırmada olasılık hesaplar.

### Kullanılan Parametreler
```python
LogisticRegression(
    random_state=1,
    max_iter=1000        # Maksimum iterasyon sayısı
)
```

### Çalışma Prensibi
1. Özelliklerin doğrusal kombinasyonunu hesaplar
2. Sigmoid fonksiyonu ile 0-1 arası olasılığa dönüştürür
3. Threshold (genellikle 0.5) kullanarak sınıfı belirler

### Matematiksel Formül
```
P(y=1|x) = 1 / (1 + e^(-z))
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Avantajları
- Hızlı ve verimli
- Olasılık tahminleri verir
- Düzenlilikleştirme (regularization) destekler
- Yorumlanabilir katsayılar

### Dezavantajları
- Sadece doğrusal ilişkileri yakalar
- Karmaşık ilişkilerde zayıf
- Outlier'lara hassas

### Proje Sonuçları
- **Accuracy:** Rapor değişkeninde (logistic_normal)
- Churn sınıfında iyi performans
- Non-churn sınıfında zayıf

---

## 8. AdaBoost (Adaptive Boosting)

### Tanım
AdaBoost, zayıf öğrenicileri (weak learners) sıralı olarak eğitip güçlü bir model oluşturan bir boosting algoritmasıdır. Her iterasyonda yanlış sınıflandırılan örneklere daha fazla ağırlık verir.

### Kullanılan Parametreler
```python
AdaBoostClassifier(
    n_estimators=500,       # Zayıf öğrenici sayısı
    learning_rate=0.5,      # Öğrenme hızı
    random_state=0
)
```

### Hiperparametre Optimizasyonu
RandomizedSearchCV ile optimize edildi:
- **learning_rate:** [0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.005]
- **n_estimators:** [300, 500]

### Çalışma Prensibi
1. İlk modeli tüm veriye eşit ağırlıkla eğit
2. Yanlış tahmin edilen örneklerin ağırlığını artır
3. Yeni model oluştur (ağırlıklı veriye odaklanır)
4. Tüm modellerin ağırlıklı oylamasını al

### Avantajları
- Overfitting'e dirençli
- Özellik seçimi yapabilir
- Zayıf modellerden güçlü model oluşturur

### Dezavantajları
- Noise'a ve outlier'lara hassas
- Yavaş eğitim
- Parametre ayarı kritik

### Proje Sonuçları
- En dengeli model
- Churn tahmininde mükemmel
- Genel accuracy biraz düşük ama güvenilir

---

## 9. Gradient Boosting

### Tanım
Gradient Boosting, her yeni modelin önceki modellerin hatalarını düzeltmeye odaklandığı bir boosting tekniğidir. Gradient descent optimizasyonu kullanır.

### Kullanılan Parametreler
```python
GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.5,      # Yüksek! Dikkat edilmeli
    random_state=0
)
```

### ⚠️ Önemli Not
learning_rate=0.5 oldukça yüksektir. Tipik değerler 0.01-0.1 aralığındadır.

### Çalışma Prensibi
1. Basit bir model ile başla
2. Rezidüelleri (hataları) hesapla
3. Yeni model rezidüelleri tahmin etmeye çalış
4. Modelleri topla (learning_rate ile ağırlıklandırılmış)
5. Durdurma kriteri gelene kadar tekrarla

### AdaBoost'tan Farkları
- AdaBoost: Örnek ağırlıklarını değiştirir
- Gradient Boosting: Rezidüelleri modeller

### Avantajları
- Çok güçlü performans
- Eksik verilere toleranslı
- Farklı kayıp fonksiyonları kullanılabilir

### Dezavantajları
- Overfitting riski
- Yavaş eğitim
- Dikkatli hiperparametre ayarı gerekir

### Proje Sonuçları
- **Accuracy:** %81.33
- İyi performans
- learning_rate düşürülerek iyileştirilebilir

---

## 10. LightGBM (Light Gradient Boosting Machine)

### Tanım
LightGBM, Microsoft tarafından geliştirilen hızlı ve dağıtık gradient boosting framework'üdür. Geleneksel GBDT'den daha hızlı ve hafıza verimlidir.

### Kullanılan Parametreler
```python
LGBMClassifier(
    subsample=0.5,
    reg_lambda=0.3,
    reg_alpha=0.1,
    num_leaves=9,
    n_estimators=500,
    min_child_weight=7,
    min_child_samples=9,
    max_depth=4,
    learning_rate=0.8,        # Çok yüksek!
    colsample_bytree=0.9,
    random_state=0
)
```

### Hiperparametre Optimizasyonu
Kapsamlı RandomizedSearchCV ile 3-fold CV yapıldı.

### Temel Yenilikler
1. **Leaf-wise tree growth:** Level-wise yerine yaprak bazlı büyüme
2. **Histogram-based learning:** Sürekli değerleri histogram kutularına böler
3. **GOSS:** Gradient-based One-Side Sampling
4. **EFB:** Exclusive Feature Bundling

### Avantajları
- Çok hızlı eğitim
- Düşük bellek kullanımı
- Yüksek doğruluk
- Büyük veri setleri için ideal

### Dezavantajları
- Küçük veri setlerinde overfitting
- Parametre sayısı fazla
- learning_rate=0.8 çok yüksek (0.01-0.1 olmalı)

### Proje Sonuçları
- **Accuracy:** %81.66
- Sadece negatif sınıfı öğrendi
- Churn tahmininde başarısız
- learning_rate ve diğer parametreler ayarlanmalı

---

## 11. XGBoost (Extreme Gradient Boosting)

### Tanım
XGBoost, gradient boosting algoritmasının optimize edilmiş ve ölçeklenebilir implementasyonudur. Kaggle yarışmalarında çok başarılı olmuştur.

### Kullanılan Parametreler
```python
XGBClassifier(
    subsample=0.7,
    reg_lambda=0.3,        # L2 regularization
    reg_alpha=0.3,         # L1 regularization
    n_estimators=500,
    min_child_weight=3,
    max_depth=6,
    learning_rate=0.3,     # Yüksek
    gamma=0.9,             # Minimum loss reduction
    colsample_bytree=0.3,
    random_state=0
)
```

### Temel Özellikler
1. **Regularization:** L1 (Lasso) ve L2 (Ridge) destekler
2. **Sparse Aware:** Eksik verileri otomatik işler
3. **Weighted Quantile Sketch:** Ağırlıklı veri için etkili
4. **Cross-validation:** Built-in CV desteği
5. **Tree Pruning:** Max-depth'ten sonra budama yapar
6. **Parallel Processing:** CPU çekirdeklerini kullanır

### LightGBM vs XGBoost
| Özellik | XGBoost | LightGBM |
|---------|---------|----------|
| Ağaç Büyümesi | Level-wise | Leaf-wise |
| Hız | Orta | Çok Hızlı |
| Bellek | Orta | Düşük |
| Doğruluk | Yüksek | Yüksek |
| Küçük Veri | İyi | Overfit riski |

### Avantajları
- Çok güçlü performans
- Regularization ile overfitting kontrolü
- Eksik veri desteği
- Feature importance çıkarır

### Dezavantajları
- Parametre sayısı çok fazla
- Eğitim yavaş olabilir
- Interpretability düşük

### Proje Sonuçları
- **Accuracy:** %82.53
- İyi genel performans
- Negatif sınıfı daha iyi öğrendi
- Churn tahmininde orta seviye

---

## Model Karşılaştırması

### Accuracy Sıralaması
1. **Random Forest:** 83.63%
2. **XGBoost:** 82.53%
3. **Extra Trees:** 82.06%
4. **LightGBM:** 81.66%
5. **Gradient Boosting:** 81.33%
6. **Decision Tree:** 76.63%
7. **Naive Bayes:** 70.46%
8. **K-Nearest Neighbors:** 67.1%
9. **K-Means:** 54.5%

### Churn Tahmini (Asıl Hedef) Açısından
**En İyi:** AdaBoost - Churn sınıfında en dengeli performans  
**İyi:** Random Forest, Decision Tree  
**Orta:** XGBoost, Gradient Boosting, Extra Trees  
**Zayıf:** LightGBM (sadece Non-churn öğrendi)

---

## Veri Ön İşleme

### 1. Feature Engineering
- **Complain** sütunu çıkarıldı (Exited ile %100 korelasyon)
- **RowNumber, CustomerId, Surname** çıkarıldı (model için gereksiz)

### 2. Encoding
```python
# OneHotEncoder kullanıldı
pd.get_dummies(df[['Geography', 'Gender', 'Card Type']])
```

### 3. Ölçeklendirme
```python
StandardScaler()
# Her özelliği ortalama=0, std=1 olacak şekilde ölçeklendirir
```

### 4. Sınıf Dengeleme
```python
RandomOverSampler()
# Azınlık sınıfını (Churn) örnekleyerek dengeler
```

### 5. Train-Test Split
```python
train_test_split(test_size=0.3, random_state=0)
# %70 eğitim, %30 test
```

---

## Özellik Önemleri

### Decision Tree'ye Göre
En önemli özellikler `feature_imp` değişkeninde

### Random Forest'a Göre
En önemli özellikler `feature_imp_random` değişkeninde

### Chi-Squared Test
`featureScores` değişkeninde tüm özelliklerin Chi-squared skorları

**Genel Bulgu:** Age, NumOfProducts ve Balance en önemli özellikler

---

## Outlier Analizi

### Metod: IQR (Interquartile Range)
```python
Q1 = 25. percentile
Q3 = 75. percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

### Kontrol Edilen Değişkenler
- CreditScore
- Age
- Balance
- EstimatedSalary
- Point Earned

### Sonuç
Outlier analizi hücresinin çıktısında detaylı rapor bulunmaktadır.

---

## Öneriler ve Sonuçlar

### En İyi Model Seçimi

**Accuracy için:** Random Forest (%83.63)  
**Churn Detection için:** AdaBoost (dengeli performans)  
**Hız için:** Naive Bayes veya Logistic Regression  
**Büyük veri için:** LightGBM (parametreler ayarlanmalı)

### İyileştirme Önerileri

1. **Learning Rate Ayarı**
   - LightGBM: 0.8 → 0.01-0.1
   - Gradient Boosting: 0.5 → 0.01-0.1
   - AdaBoost: 0.5 → 0.1-0.3

2. **LightGBM için**
   - class_weight='balanced' ekle
   - scale_pos_weight ayarla
   - min_child_samples azalt

3. **Feature Engineering**
   - Age grupları oluştur (genç, orta yaş, yaşlı)
   - Balance/EstimatedSalary oranı
   - Tenure/Age oranı
   - Polynomial features

4. **Ensemble Yöntemleri**
   - Stacking: Random Forest + AdaBoost + XGBoost
   - Voting Classifier: Soft voting
   - Blending

5. **Cross-Validation**
   - Stratified K-Fold (K=5 veya 10)
   - Time-series split (eğer zaman verisi varsa)

### Üretim (Production) İçin

**Önerilen Model:** Random Forest
- Yüksek accuracy
- Dengeli performans
- Robust (aykırı değerlere toleranslı)
- Özellik önemleri çıkarabilir

**Alternatif:** AdaBoost
- Churn detection'da daha başarılı
- Daha az overfitting
- Yeni verilere daha genellenebilir

---

## Teknik Detaylar

### Kullanılan Kütüphaneler
```python
pandas, numpy              # Veri işleme
matplotlib, seaborn        # Görselleştirme
sklearn                    # ML modelleri ve metrikler
imblearn                   # Sınıf dengeleme
lightgbm                   # LightGBM
xgboost                    # XGBoost
yellowbrick                # Confusion Matrix görselleştirme
```

### Değerlendirme Metrikleri
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Cross-Validation Stratejileri
- RandomizedSearchCV (2-5 fold)
- GridSearchCV (2 fold)

---

## Sonuç

Bu proje, 11 farklı makine öğrenimi modeli ile customer churn prediction yapılmıştır. Random Forest en yüksek accuracy'yi verirken, AdaBoost churn detection konusunda en dengeli sonuçları üretmiştir. 

Boosting algoritmaları (AdaBoost, XGBoost, LightGBM, Gradient Boosting) genel olarak iyi performans göstermiş, ancak learning_rate parametreleri optimize edilmelidir.

Projenin başarısı için veri ön işleme (StandardScaler, OneHotEncoder, RandomOverSampler) kritik rol oynamıştır.

---

**Rapor Tarihi:** 26 Aralık 2025  
**Dosya:** customer-churn.ipynb  
**Veri Seti:** Customer-Churn-Records.csv
