# Film Öneri Sistemi

Bu proje, film veri seti kullanarak kullanıcılara iki farklı algoritma ile film önerileri sunan bir sistemdir.

## Proje Hakkında

Bu projede, iki farklı öneri algoritması kullanılmıştır:

1. **ALS (Alternating Least Squares)**: İşbirlikçi filtreleme tabanlı bir öneri algoritmasıdır. Kullanıcıların film puanlamalarındaki benzerliklerden yararlanarak öneriler oluşturur.

2. **İçerik Tabanlı Filtreleme (Content-Based Filtering)**: Filmlerin içerik özelliklerine (tür, etiketler, vb.) dayalı bir öneri algoritmasıdır. Kullanıcının beğendiği filmlere benzer içerikteki filmleri önerir.

## Proje Yapısı

- `src/data_loader.py`: Spark ile veri yükleme ve temel ön işleme işlemlerini yapan sınıf.
- `src/model_trainer.py`: ALS algoritmasını eğiten ve değerlendiren sınıf.
- `src/content_based_recommender.py`: İçerik tabanlı filtreleme algoritmasını uygulayan sınıf.
- `src/simple_app.py`: Ana Flask web uygulaması ve API endpointleri. (Kullanıcı arayüzü ve öneri servisleri burada.)
- `src/templates/index.html`: Modern ve kullanıcı dostu web arayüzü şablonu.

> Not: Tüm öneriler ve metrikler Flask arayüzünde görselleştirilir. Öneri kartlarında hem similarity (benzerlik) hem de varsa kullanıcının gerçek rating'i gösterilir.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyaç vardır:

```
pyspark==3.1.2
flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0
matplotlib==3.4.3
seaborn==0.11.2
networkx==2.6.3
findspark==1.4.2
```

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

2. Veri setini indirin ve `archive` klasörüne (proje ana dizininde olacak şekilde) çıkarın. Örnek yol: `BIL401Proje/archive/`

## Çalıştırma

1. Proje dizininde aşağıdaki komutu çalıştırın:
   ```
   python src/simple_app.py
   ```

2. Web tarayıcınızda `http://localhost:5000` adresine gidin.

## Özellikler

- ALS ve İçerik Tabanlı algoritmalar ile öneri
- Kullanıcı bazlı film önerileri
- Her öneri için similarity (benzerlik) ve varsa rating (kullanıcının puanı) gösterimi
- Algoritma performans metriklerinin (precision, recall, f1, coverage, diversity, novelty, RMSE, MAE) tablo halinde karşılaştırılması
- Kullanıcı puanlamalarının ve öneri geçmişinin görüntülenmesi
- Modern ve kullanıcı dostu Flask arayüzü
- Spark tabanlı büyük veri desteği

## Algoritma Karşılaştırması ve Metrikler

Proje arayüzünde, iki algoritmanın tüm önemli metrikleri tablo halinde karşılaştırmalı gösterilir:

- **ALS**: Precision, Recall, F1 Score, Coverage, Diversity, Novelty, RMSE, MAE
- **İçerik Tabanlı**: Precision, Recall, F1 Score, Coverage, Diversity, Novelty, RMSE, MAE
- **Jaccard Benzerliği**: ALS ve içerik tabanlı önerilerin ortak film oranı

Her öneri kartında ayrıca:
- Benzerlik oranı (similarity)
- Kullanıcı daha önce puanladıysa gerçek rating
ayrı ayrı gösterilir.

### Notlar
- Spark kurulumu ve Java gereksinimi için sisteminizde Java ve Spark yüklü olmalıdır.
- Uygulama, büyük veri setleriyle çalışmaya uygundur. Küçük örnek veriyle de test edebilirsiniz.
- Tüm ayarlar ve veri yolu, `src/simple_app.py` ve `src/data_loader.py` dosyalarında düzenlenebilir.