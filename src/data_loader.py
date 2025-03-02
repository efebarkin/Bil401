import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, count, avg, stddev, isnan, when, isnull, regexp_extract
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler

class DataLoader:
    def __init__(self, spark):
        self.spark = spark
        self.data_path = "c:/Users/efeba/Desktop/archive"
        self.setup_logging()
        
    def setup_logging(self):
        """Loglama sistemini ayarlar"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('movie_recommender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_dataframe(self, df, name):
        """DataFrame'i doğrular ve temel istatistikleri loglar"""
        self.logger.info(f"\nValidating {name} DataFrame:")
        
        # Satır ve sütun sayılarını logla
        row_count = df.count()
        col_count = len(df.columns)
        self.logger.info(f"Shape: {row_count} rows, {col_count} columns")
        
        # Eksik değerleri kontrol et
        for column in df.columns:
            # Kolonun veri tipini al
            col_type = df.schema[column].dataType.typeName()
            
            # Veri tipine göre null kontrolü yap
            if col_type in ['double', 'float']:
                null_count = df.filter(col(column).isNull() | isnan(column)).count()
            else:
                null_count = df.filter(col(column).isNull()).count()
                
            if null_count > 0:
                self.logger.warning(f"Column {column} has {null_count} null values")
                
        # Duplicate kontrol
        duplicate_count = df.count() - df.dropDuplicates().count()
        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate rows")
            
        return df
    
    def handle_missing_values(self, df):
        """Eksik değerleri işler"""
        # Sayısal kolonlar için ortalama ile doldur
        numeric_columns = [item[0] for item in df.dtypes if item[1] in ['double', 'int']]
        for column in numeric_columns:
            df = df.withColumn(column,
                when(col(column).isNull() | isnan(column),
                    df.select(avg(column)).first()[0]
                ).otherwise(col(column)))
        
        return df
    
    def load_ratings(self):
        """
        Rating dosyasını yükler ve ön işleme yapar
        Columns: userId, movieId, rating, timestamp
        """
        self.logger.info("Loading ratings data...")
        
        ratings_df = self.spark.read.csv(
            os.path.join(self.data_path, "rating.csv"),
            header=True,
            inferSchema=True
        )
        
        # Veri doğrulama
        ratings_df = self.validate_dataframe(ratings_df, "ratings")
        
        # Eksik değerleri işle
        ratings_df = self.handle_missing_values(ratings_df)
        
        # Rating kolonunu double'a çevir
        ratings_df = ratings_df.withColumn("rating", col("rating").cast(DoubleType()))
        
        # Timestamp'i işle
        ratings_df = ratings_df.withColumn("timestamp", col("timestamp").cast("timestamp"))
            
        return ratings_df

    def load_movies(self):
        """
        Film dosyasını yükler ve ön işleme yapar
        Columns: movieId, title, genres
        """
        self.logger.info("Loading movies data...")
        
        movies_df = self.spark.read.csv(
            os.path.join(self.data_path, "movie.csv"),
            header=True,
            inferSchema=True
        )
        
        # Veri doğrulama
        movies_df = self.validate_dataframe(movies_df, "movies")
        
        # Eksik değerleri işle
        movies_df = self.handle_missing_values(movies_df)
        
        # Genres'i array'e çevir ve one-hot encoding uygula
        movies_df = movies_df.withColumn("genres", split(col("genres"), "\\|"))
        
        # Yıl bilgisini title'dan çıkar
        movies_df = movies_df.withColumn(
            "year",
            when(
                col("title").rlike(".*\\((\\d{4})\\)$"),
                regexp_extract(col("title"), ".*\\((\\d{4})\\)$", 1).cast("integer")
            ).otherwise(None)
        )
            
        return movies_df

    def load_tags(self):
        """
        Etiket dosyasını yükler ve ön işleme yapar
        Columns: userId, movieId, tag, timestamp
        """
        self.logger.info("Loading tags data...")
        
        tags_df = self.spark.read.csv(
            os.path.join(self.data_path, "tag.csv"),
            header=True,
            inferSchema=True
        )
        
        # Veri doğrulama
        tags_df = self.validate_dataframe(tags_df, "tags")
        
        # Eksik değerleri işle
        tags_df = self.handle_missing_values(tags_df)
        
        return tags_df

    def load_genome_scores(self):
        """
        Genome scores dosyasını yükler
        Columns: movieId, tagId, relevance
        """
        self.logger.info("Loading genome scores data...")
        
        genome_scores_df = self.spark.read.csv(
            os.path.join(self.data_path, "genome_scores.csv"),
            header=True,
            inferSchema=True
        )
        
        # Veri doğrulama
        genome_scores_df = self.validate_dataframe(genome_scores_df, "genome scores")
        
        # Eksik değerleri işle
        genome_scores_df = self.handle_missing_values(genome_scores_df)
        
        return genome_scores_df

    def load_genome_tags(self):
        """
        Genome tags dosyasını yükler
        Columns: tagId, tag
        """
        self.logger.info("Loading genome tags data...")
        
        genome_tags_df = self.spark.read.csv(
            os.path.join(self.data_path, "genome_tags.csv"),
            header=True,
            inferSchema=True
        )
        
        # Veri doğrulama
        genome_tags_df = self.validate_dataframe(genome_tags_df, "genome tags")
        
        # Eksik değerleri işle
        genome_tags_df = self.handle_missing_values(genome_tags_df)
        
        return genome_tags_df

    def get_movie_statistics(self, ratings_df):
        """
        Filmler için temel istatistikleri hesaplar
        """
        return ratings_df.groupBy("movieId").agg(
            count("rating").alias("rating_count"),
            avg("rating").alias("rating_mean"),
            stddev("rating").alias("rating_stddev")
        )

    def get_user_statistics(self, ratings_df):
        """
        Kullanıcılar için temel istatistikleri hesaplar
        """
        return ratings_df.groupBy("userId").agg(
            count("rating").alias("rating_count"),
            avg("rating").alias("rating_mean"),
            stddev("rating").alias("rating_stddev")
        )

    def get_popular_genres(self, movies_df, ratings_df, n=10):
        """
        En popüler türleri bulur
        """
        # Film türlerini patlat
        genres_df = movies_df.select(
            col("movieId"),
            explode(col("genres")).alias("genre")
        )
        
        # Türlere göre ortalama rating ve film sayısını hesapla
        return genres_df.join(ratings_df, "movieId") \
            .groupBy("genre") \
            .agg(
                count("movieId").alias("movie_count"),
                avg("rating").alias("avg_rating"),
                count("rating").alias("rating_count")
            ) \
            .orderBy(col("rating_count").desc()) \
            .limit(n)

    def get_genre_statistics(self, ratings_df, movies_df):
        """Film türlerine göre istatistikleri hesaplar"""
        # Önce film-tür eşleştirmesini yap
        movies_exploded = movies_df.select("movieId", explode("genres").alias("genre"))
        
        # Türlere göre değerlendirme sayısı ve ortalama puanı hesapla
        genre_stats = movies_exploded.join(ratings_df, "movieId") \
            .groupBy("genre") \
            .agg(
                count("rating").alias("rating_count"),
                avg("rating").alias("avg_rating"),
                count("movieId").alias("movie_count")
            ) \
            .orderBy("rating_count", ascending=False)
            
        return genre_stats

    def create_feature_vector(self, df):
        """Feature vektörü oluşturur"""
        # Sayısal kolonları seç
        numeric_cols = [item[0] for item in df.dtypes if item[1] in ['double', 'int']]
        
        # VectorAssembler ile feature vektörü oluştur
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
        df = assembler.transform(df)
        
        return df

    def get_top_rated_movies(self, ratings_df, movies_df, n=10, min_ratings=100):
        """En çok değerlendirilen ve en yüksek puan alan filmleri getirir"""
        # Film başına değerlendirme istatistiklerini hesapla
        movie_stats = ratings_df.groupBy("movieId") \
            .agg(
                count("rating").alias("rating_count"),
                avg("rating").alias("avg_rating"),
                stddev("rating").alias("rating_stddev")
            ) \
            .filter(col("rating_count") >= min_ratings)  # Minimum değerlendirme sayısı filtresi
        
        # Film bilgileriyle birleştir
        top_movies = movie_stats.join(movies_df, "movieId") \
            .select(
                "movieId",
                "title",
                "genres",
                "rating_count",
                "avg_rating",
                "rating_stddev"
            ) \
            .orderBy(["rating_count", "avg_rating"], ascending=[False, False]) \
            .limit(n)
        
        return top_movies