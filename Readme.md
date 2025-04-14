# Film Öneri Sistemi

Bu proje, film veri seti kullanarak kullanıcılara iki farklı algoritma ile film önerileri sunan bir sistemdir.

## Proje Hakkında

Bu projede, iki farklı öneri algoritması kullanılmıştır:

1. **ALS (Alternating Least Squares)**: İşbirlikçi filtreleme tabanlı bir öneri algoritmasıdır. Kullanıcıların film puanlamalarındaki benzerliklerden yararlanarak öneriler oluşturur.

2. **İçerik Tabanlı Filtreleme (Content-Based Filtering)**: Filmlerin içerik özelliklerine (tür, etiketler, vb.) dayalı bir öneri algoritmasıdır. Kullanıcının beğendiği filmlere benzer içerikteki filmleri önerir.

## Proje Yapısı

- `src/data_loader.py`: Veri yükleme ve ön işleme işlemlerini gerçekleştiren sınıf
- `src/model_trainer.py`: ALS modelini eğiten ve değerlendiren sınıf
- `src/content_based_recommender.py`: İçerik tabanlı filtreleme algoritmasını uygulayan sınıf
- `src/graph_visualizer.py`: Graf görselleştirme işlemlerini gerçekleştiren sınıf
- `src/app.py`: Flask web uygulaması
- `src/templates/index.html`: Web arayüzü şablonu

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

2. Veri setini indirin ve `c:/Users/efeba/Desktop/archive` dizinine çıkarın.

## Çalıştırma

1. Proje dizininde aşağıdaki komutu çalıştırın:
   ```
   python src/app.py
   ```

2. Web tarayıcınızda `http://localhost:5000` adresine gidin.

## Özellikler

- İki farklı öneri algoritmasının karşılaştırılması
- Kullanıcı bazlı film önerileri
- Algoritma performans metrikleri
- Kullanıcı puanlamalarının görüntülenmesi
- Modern ve kullanıcı dostu arayüz

## Algoritma Karşılaştırması

Bu projede, ALS ve İçerik Tabanlı Filtreleme algoritmalarının performansları karşılaştırılmıştır:

- **ALS**: RMSE ve MAE metrikleri ile değerlendirilmiştir.
- **İçerik Tabanlı Filtreleme**: Precision, Recall ve F1 Score metrikleri ile değerlendirilmiştir.
- **Jaccard Benzerliği**: İki algoritmanın önerilerinin benzerliği ölçülmüştür.