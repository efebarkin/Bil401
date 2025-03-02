import os
import logging
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc
from data_loader import DataLoader
from model_trainer import ModelTrainer

def setup_logging():
    """Loglama sistemini ayarlar"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('movie_recommender.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_spark_session():
    """Spark oturumu oluşturur"""
    # Geçici dizin yolu oluştur
    spark_temp_dir = os.path.join(os.getcwd(), "spark_temp")
    os.makedirs(spark_temp_dir, exist_ok=True)
    
    return SparkSession.builder \
        .appName("MovieRecommender") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.local.dir", spark_temp_dir) \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.default.parallelism", "4") \
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC") \
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC") \
        .master("local[*]") \
        .getOrCreate()

def load_and_preprocess_data(data_loader):
    """Veriyi yükler ve ön işler"""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing data...")
    
    # Verileri yükle
    ratings_df = data_loader.load_ratings()
    movies_df = data_loader.load_movies()
    tags_df = data_loader.load_tags()
    genome_scores_df = data_loader.load_genome_scores()
    genome_tags_df = data_loader.load_genome_tags()
    
    # Veri setleri hakkında temel bilgileri göster
    logger.info("\nDataset sizes:")
    logger.info(f"Ratings: {ratings_df.count():,} rows")
    logger.info(f"Movies: {movies_df.count():,} rows")
    logger.info(f"Tags: {tags_df.count():,} rows")
    logger.info(f"Genome Scores: {genome_scores_df.count():,} rows")
    logger.info(f"Genome Tags: {genome_tags_df.count():,} rows")
    
    return ratings_df, movies_df, tags_df, genome_scores_df, genome_tags_df

def analyze_data(data_loader, ratings_df, movies_df):
    """Veri analizi yapar"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing data...")
    
    # Film istatistiklerini hesapla
    movie_stats = data_loader.get_movie_statistics(ratings_df)
    print("\n=== Film Değerlendirme İstatistikleri ===")
    logger.info("\nMovie rating statistics:")
    movie_stats.describe().show()
    
    # En çok değerlendirilen filmleri göster
    print("\n=== En Çok Değerlendirilen Filmler ===")
    logger.info("\nTop rated movies:")
    top_movies = data_loader.get_top_rated_movies(ratings_df, movies_df)
    top_movies.show(10, truncate=False)
    
    # Film türü istatistiklerini göster
    print("\n=== Film Türü İstatistikleri ===")
    logger.info("\nGenre statistics:")
    genre_stats = data_loader.get_genre_statistics(ratings_df, movies_df)
    genre_stats.show(10)
    
    # Kullanıcı istatistiklerini göster
    print("\n=== Kullanıcı Değerlendirme İstatistikleri ===")
    logger.info("\nUser rating statistics:")
    user_stats = data_loader.get_user_statistics(ratings_df)
    user_stats.describe().show()

def train_and_evaluate_model(model_trainer, ratings_df):
    """Modeli eğitir ve değerlendirir"""
    logger = logging.getLogger(__name__)
    logger.info("Training and evaluating model...")
    
    # Cross validation ile modeli eğit
    best_model, predictions, rmse, mae = model_trainer.train_with_cross_validation(ratings_df)
    
    # Tahminleri görselleştir
    model_trainer.plot_predictions(predictions)
    
    # Örnek öneriler
    sample_user_id = ratings_df.select("userId").distinct().limit(1).collect()[0][0]
    recommendations = model_trainer.get_top_recommendations(best_model, sample_user_id)
    
    logger.info(f"\nSample recommendations for user {sample_user_id}:")
    recommendations.show(10, False)
    
    return best_model

def visualize_predictions(best_model, ratings_df):
    # Tahminleri görselleştir
    pass

def main():
    # Loglama sistemini ayarla
    logger = setup_logging()
    
    # Spark log seviyesini ayarla
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.ERROR)
    
    spark = None
    try:
        # Spark oturumu başlat
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
        
        # DataLoader ve ModelTrainer nesnelerini oluştur
        data_loader = DataLoader(spark)
        model_trainer = ModelTrainer()
        
        # Veriyi yükle ve ön işle
        logger.info("Loading and preprocessing data...")
        ratings_df, movies_df, tags_df, genome_scores_df, genome_tags_df = load_and_preprocess_data(data_loader)
        
        # Veri analizi yap
        analyze_data(data_loader, ratings_df, movies_df)
        
        # Modeli eğit ve değerlendir
        best_model = train_and_evaluate_model(model_trainer, ratings_df)
        
        # Tahminleri görselleştir
        visualize_predictions(best_model, ratings_df)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if spark:
            # Spark oturumunu düzgün şekilde kapat
            spark.stop()
            logger.info("Spark session stopped successfully")

if __name__ == "__main__":
    main()