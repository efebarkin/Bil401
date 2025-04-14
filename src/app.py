import os
import logging
import findspark
findspark.init()

from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import json
import tempfile

from data_loader import DataLoader
from model_trainer import ModelTrainer
from content_based_recommender import ContentBasedRecommender

# Loglama sistemini ayarla
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('movie_recommender.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Spark oturumu oluştur
def create_spark_session():
    """Create and configure a Spark session."""
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    import tempfile
    import os
    import shutil
    import subprocess
    import sys
    
    # Java süreçlerini temizle
    try:
        if sys.platform == 'win32':
            subprocess.call(['taskkill', '/F', '/IM', 'java.exe'], 
                           stderr=subprocess.DEVNULL, 
                           stdout=subprocess.DEVNULL)
    except:
        pass
    
    # Geçici dizin oluştur ve temizle
    temp_dir = os.path.join(tempfile.gettempdir(), "spark_temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    os.makedirs(temp_dir, exist_ok=True)
    
    # Çok basit Spark yapılandırması
    return SparkSession.builder \
        .appName("MovieRecommendationSystem") \
        .config("spark.driver.memory", "1g") \
        .config("spark.local.dir", temp_dir) \
        .master("local[1]") \
        .getOrCreate()

# Flask uygulamasını oluştur
app = Flask(__name__)
logger = setup_logging()

# Global değişkenler
spark = None
data_loader = None
model_trainer = None
content_recommender = None
als_model = None
ratings_df = None
movies_df = None
tags_df = None
genome_scores_df = None
genome_tags_df = None
movie_id_to_title = {}
user_ids = []

# Model dosya yolları
ALS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/als_model")
CONTENT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/content_model")

def initialize():
    global spark, data_loader, model_trainer, content_recommender
    global ratings_df, movies_df, tags_df, genome_scores_df, genome_tags_df
    global als_model, movie_id_to_title, user_ids
    
    try:
        # Spark oturumu başlat
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
        
        # DataLoader, ModelTrainer ve ContentBasedRecommender nesnelerini oluştur
        data_loader = DataLoader(spark)
        model_trainer = ModelTrainer()
        content_recommender = ContentBasedRecommender(spark)
        
        # Veriyi yükle - daha küçük veri seti kullan
        logger.info("Loading data...")
        ratings_df = data_loader.load_ratings()
        # Veri setini küçült (bellek sorunlarını önlemek için)
        ratings_df = ratings_df.sample(fraction=0.1, seed=42)
        ratings_df.cache()
        
        movies_df = data_loader.load_movies()
        movies_df.cache()
        
        # Sadece gerekli verileri yükle
        tags_df = data_loader.load_tags()
        tags_df = tags_df.sample(fraction=0.1, seed=42)
        tags_df.cache()
        
        # Genom verilerini örnekle
        try:
            genome_scores_df = data_loader.load_genome_scores()
            genome_scores_df = genome_scores_df.sample(fraction=0.05, seed=42)
            genome_scores_df.cache()
            
            genome_tags_df = data_loader.load_genome_tags()
            genome_tags_df.cache()
        except Exception as e:
            logger.warning(f"Error loading genome data: {str(e)}")
            genome_scores_df = None
            genome_tags_df = None
        
        # Film ID'lerini ve başlıklarını eşleştir - önbellek kullanarak
        movie_titles_df = movies_df.select("movieId", "title")
        movie_titles_df.cache()
        movie_id_to_title = {row['movieId']: row['title'] for row in movie_titles_df.collect()}
        movie_titles_df.unpersist()
        
        # Kullanıcı ID'lerini al (ilk 20 kullanıcı)
        user_ids_df = ratings_df.select("userId").distinct().limit(20)
        user_ids_df.cache()
        user_ids = [row['userId'] for row in user_ids_df.collect()]
        user_ids_df.unpersist()
        
        # Sadece önceden eğitilmiş ALS modelini yükle
        logger.info("Checking for existing ALS model...")
        
        try:
            if os.path.exists(ALS_MODEL_PATH):
                logger.info(f"Loading existing ALS model from {ALS_MODEL_PATH}...")
                als_model = model_trainer.load_model(ALS_MODEL_PATH)
                if als_model is None:
                    logger.warning("Failed to load ALS model. Please run train_models.py first.")
            else:
                logger.warning(f"ALS model not found at {ALS_MODEL_PATH}. Please run train_models.py first.")
                als_model = None
        except Exception as e:
            logger.error(f"Error loading ALS model: {str(e)}")
            als_model = None
        
        # Sadece önceden hazırlanmış içerik tabanlı modeli yükle
        logger.info("Checking for existing content-based model...")
        
        try:
            if os.path.exists(os.path.join(CONTENT_MODEL_PATH, "vectorizer.pkl")):
                logger.info(f"Loading existing content-based model from {CONTENT_MODEL_PATH}...")
                if content_recommender.load_model(CONTENT_MODEL_PATH):
                    logger.info("Content-based model loaded successfully")
                else:
                    logger.warning("Failed to load content-based model. Please run train_models.py first.")
            else:
                logger.warning(f"Content-based model not found at {CONTENT_MODEL_PATH}. Please run train_models.py first.")
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
        
        logger.info("Initialization complete")
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        # Kritik hata - basit bir model ile devam et
        try:
            if spark is None:
                spark = create_spark_session()
            if data_loader is None:
                data_loader = DataLoader(spark)
            if model_trainer is None:
                model_trainer = ModelTrainer()
            if content_recommender is None:
                content_recommender = ContentBasedRecommender(spark)
                
            # Minimal veri yükle
            ratings_df = data_loader.load_ratings().sample(fraction=0.01, seed=42).cache()
            movies_df = data_loader.load_movies().cache()
            tags_df = None
            genome_scores_df = None
            genome_tags_df = None
            
            # Basit film bilgileri
            movie_id_to_title = {row['movieId']: row['title'] for row in movies_df.select("movieId", "title").limit(100).collect()}
            user_ids = [row['userId'] for row in ratings_df.select("userId").distinct().limit(10).collect()]
            
            # ALS modeli yok
            als_model = None
            
            # Basit içerik tabanlı model
            content_recommender.simple_prepare_data(movies_df)
            
            logger.info("Fallback initialization complete")
        except Exception as e2:
            logger.error(f"Fatal error in fallback initialization: {str(e2)}")
            raise e

# Uygulama başlatıldığında initialize fonksiyonunu çağır
with app.app_context():
    initialize()

@app.route('/')
def index():
    global user_ids
    return render_template('index.html', user_ids=user_ids)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        user_id = int(data.get('userId'))
        algorithm = data.get('algorithm', 'als')  # 'als' veya 'content_based'
        n = int(data.get('count', 10))
        
        recommendations = []
        
        if algorithm == 'als':
            # ALS önerileri - model olmadığında
            if als_model is None:
                return jsonify({
                    'userId': user_id,
                    'algorithm': algorithm,
                    'recommendations': [],
                    'error': 'ALS model is not available due to training errors'
                })
            
            # ALS önerileri - model varsa
            als_recs = model_trainer.get_top_recommendations(als_model, user_id, n)
            als_recs_list = []
            
            for row in als_recs.collect():
                movie_id = row['movieId']
                title = movie_id_to_title.get(movie_id, f"Film {movie_id}")
                als_recs_list.append({
                    'movieId': int(movie_id),
                    'title': title,
                    'rating': float(row['rating'])
                })
            
            recommendations = als_recs_list
            
        elif algorithm == 'content_based':
            # Content-based öneriler
            content_recs = content_recommender.get_recommendations_for_user(user_id, ratings_df, n)
            
            if not content_recs.empty:
                recommendations = content_recs.to_dict('records')
                # Benzerlik skorunu rating olarak göster
                for rec in recommendations:
                    rec['rating'] = float(rec['similarity'])
                    del rec['similarity']
        
        return jsonify({
            'userId': user_id,
            'algorithm': algorithm,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/compare_algorithms', methods=['POST'])
def compare_algorithms():
    try:
        data = request.json
        user_id = int(data.get('userId'))
        n = int(data.get('count', 10))
        
        # ALS önerileri
        als_recs = model_trainer.get_top_recommendations(als_model, user_id, n)
        als_recs_list = []
        
        for row in als_recs.collect():
            movie_id = row['movieId']
            title = movie_id_to_title.get(movie_id, f"Film {movie_id}")
            als_recs_list.append({
                'movieId': int(movie_id),
                'title': title,
                'rating': float(row['rating'])
            })
        
        # Content-based öneriler
        content_recs = content_recommender.get_recommendations_for_user(user_id, ratings_df, n)
        content_recs_list = []
        
        if not content_recs.empty:
            content_recs_list = content_recs.to_dict('records')
            # Benzerlik skorunu rating olarak göster
            for rec in content_recs_list:
                rec['rating'] = float(rec['similarity'])
                del rec['similarity']
        
        # Önerilerin kesişimini bul
        als_movie_ids = set(rec['movieId'] for rec in als_recs_list)
        content_movie_ids = set(rec['movieId'] for rec in content_recs_list)
        common_movie_ids = als_movie_ids.intersection(content_movie_ids)
        
        # Jaccard benzerliği hesapla
        jaccard_similarity = len(common_movie_ids) / len(als_movie_ids.union(content_movie_ids)) if als_movie_ids or content_movie_ids else 0
        
        return jsonify({
            'userId': user_id,
            'alsRecommendations': als_recs_list,
            'contentBasedRecommendations': content_recs_list,
            'commonMovies': list(common_movie_ids),
            'jaccardSimilarity': jaccard_similarity
        })
        
    except Exception as e:
        logger.error(f"Error comparing algorithms: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_user_ratings', methods=['POST'])
def get_user_ratings():
    try:
        data = request.json
        user_id = int(data.get('userId'))
        
        # Kullanıcının izlediği filmleri al
        user_ratings = ratings_df.filter(col("userId") == user_id).select("movieId", "rating")
        
        # Film başlıklarını ekle
        user_ratings_list = []
        
        for row in user_ratings.collect():
            movie_id = row['movieId']
            title = movie_id_to_title.get(movie_id, f"Film {movie_id}")
            user_ratings_list.append({
                'movieId': int(movie_id),
                'title': title,
                'rating': float(row['rating'])
            })
        
        # Puanlamaya göre sırala
        user_ratings_list.sort(key=lambda x: x['rating'], reverse=True)
        
        return jsonify({
            'userId': user_id,
            'ratings': user_ratings_list
        })
        
    except Exception as e:
        logger.error(f"Error getting user ratings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search_movies', methods=['POST'])
def search_movies():
    try:
        data = request.json
        query = data.get('query', '').lower()
        
        if not query or len(query) < 2:
            return jsonify({'movies': []})
        
        # Film başlıklarında arama yap
        matching_movies = []
        
        for movie_id, title in movie_id_to_title.items():
            if query in title.lower():
                matching_movies.append({
                    'movieId': int(movie_id),
                    'title': title
                })
                
                # En fazla 20 sonuç göster
                if len(matching_movies) >= 20:
                    break
        
        return jsonify({'movies': matching_movies})
        
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_algorithms')
def evaluate_algorithms():
    try:
        # ALS modelini değerlendir
        _, test_predictions, als_rmse, als_mae = model_trainer.train_with_cross_validation(ratings_df)
        
        # Content-based modeli değerlendir
        content_metrics = content_recommender.evaluate(ratings_df, test_users=50)
        
        return jsonify({
            'als': {
                'rmse': float(als_rmse),
                'mae': float(als_mae)
            },
            'contentBased': {
                'precision': float(content_metrics['precision']),
                'recall': float(content_metrics['recall']),
                'f1Score': float(content_metrics['f1_score'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error evaluating algorithms: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def shutdown_spark(exception=None):
    global spark
    if spark:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == '__main__':
    try:
        # Java süreçlerini temizle
        import subprocess
        import sys
        if sys.platform == 'win32':
            try:
                subprocess.call(['taskkill', '/F', '/IM', 'java.exe'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except:
                pass
        
        # Uygulama başlat
        initialize()
        # Debug modunu kapat
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
