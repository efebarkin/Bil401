import os
import logging
import findspark
findspark.init()
from pyspark.sql import SparkSession
import tempfile
import shutil
import sys

from data_loader import DataLoader
from model_trainer import ModelTrainer
from content_based_recommender import ContentBasedRecommender

# Loglama ayarları
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Model dosya yolları
ALS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/als_model")
CONTENT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/content_model")

# Spark oturumu oluştur
def create_spark_session():
    """Çok basit bir Spark oturumu oluşturur"""
    # Önceki Spark oturumlarının geçici dosyalarını temizle
    try:
        import subprocess
        if sys.platform == 'win32':
            subprocess.call(['taskkill', '/F', '/IM', 'java.exe'], 
                           stderr=subprocess.DEVNULL, 
                           stdout=subprocess.DEVNULL)
    except:
        pass
    
    # Geçici dizin oluştur
    temp_dir = os.path.join(tempfile.gettempdir(), "spark_temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    os.makedirs(temp_dir, exist_ok=True)
    
    # Çok basit Spark yapılandırması
    return SparkSession.builder \
        .appName("MovieRecommendationModelTraining") \
        .config("spark.driver.memory", "1g") \
        .config("spark.local.dir", temp_dir) \
        .master("local[1]") \
        .getOrCreate()

def train_models():
    """Modelleri eğit ve kaydet"""
    logger.info("Starting model training process...")
    
    try:
        # Java süreçlerini temizle (Windows'ta)
        if sys.platform == 'win32':
            try:
                import subprocess
                subprocess.call(['taskkill', '/F', '/IM', 'java.exe'], 
                               stderr=subprocess.DEVNULL, 
                               stdout=subprocess.DEVNULL)
            except:
                pass
        
        # Spark oturumu başlat
        spark = create_spark_session()
        
        # Veri yükleyici ve model eğiticileri oluştur
        data_loader = DataLoader(spark)
        model_trainer = ModelTrainer()
        content_recommender = ContentBasedRecommender(spark)
        
        # Veriyi yükle - eğitim için daha büyük örneklem
        logger.info("Loading data...")
        ratings_df = data_loader.load_ratings()
        ratings_df = ratings_df.sample(fraction=0.2, seed=42).cache()  # %20 örneklem
        
        movies_df = data_loader.load_movies().cache()
        
        tags_df = data_loader.load_tags()
        tags_df = tags_df.sample(fraction=0.2, seed=42).cache()
        
        try:
            genome_scores_df = data_loader.load_genome_scores()
            genome_scores_df = genome_scores_df.sample(fraction=0.1, seed=42).cache()
            
            genome_tags_df = data_loader.load_genome_tags().cache()
        except Exception as e:
            logger.warning(f"Error loading genome data: {str(e)}")
            genome_scores_df = None
            genome_tags_df = None
        
        # ALS modelini eğit ve kaydet
        logger.info("Training ALS model...")
        try:
            # Klasörü oluştur
            os.makedirs(os.path.dirname(ALS_MODEL_PATH), exist_ok=True)
            
            # Modeli eğit
            als_model, rmse, mae, r2 = model_trainer.train_with_cross_validation(ratings_df, num_folds=3)
            
            # Modeli kaydet
            model_trainer.save_model(als_model, ALS_MODEL_PATH)
            
            logger.info(f"ALS model trained and saved successfully. RMSE: {rmse}, MAE: {mae}, R²: {r2}")
        except Exception as e:
            logger.error(f"Error training ALS model: {str(e)}")
        
        # İçerik tabanlı modeli hazırla ve kaydet
        logger.info("Preparing content-based model...")
        try:
            # Klasörü oluştur
            os.makedirs(CONTENT_MODEL_PATH, exist_ok=True)
            
            # Modeli hazırla
            content_recommender.prepare_data(movies_df, tags_df, genome_scores_df, genome_tags_df)
            
            # Modeli kaydet
            content_recommender.save_model(CONTENT_MODEL_PATH)
            
            logger.info("Content-based model prepared and saved successfully")
        except Exception as e:
            logger.error(f"Error preparing content-based model: {str(e)}")
            try:
                # Basit modeli dene
                content_recommender.simple_prepare_data(movies_df)
                content_recommender.save_model(CONTENT_MODEL_PATH)
                logger.info("Simple content-based model prepared and saved successfully")
            except Exception as e2:
                logger.error(f"Error preparing simple content-based model: {str(e2)}")
        
        # Spark oturumunu kapat
        spark.stop()
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_models()
