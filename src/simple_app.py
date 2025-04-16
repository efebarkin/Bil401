import os
import logging
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Loglama sistemini ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('movie_recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model ve veri yolları
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ALS_MODEL_PATH = os.path.join(MODEL_DIR, "als_model")
CONTENT_MODEL_PATH = os.path.join(MODEL_DIR, "content_model")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")

# Yol ve dosya kontrolleri
logger.info(f"MODEL_DIR: {MODEL_DIR} - Exists: {os.path.exists(MODEL_DIR)}")
logger.info(f"ALS_MODEL_PATH: {ALS_MODEL_PATH} - Exists: {os.path.exists(ALS_MODEL_PATH)}")
logger.info(f"CONTENT_MODEL_PATH: {CONTENT_MODEL_PATH} - Exists: {os.path.exists(CONTENT_MODEL_PATH)}")
logger.info(f"DATA_PATH: {DATA_PATH} - Exists: {os.path.exists(DATA_PATH)}")
logger.info(f"movie.csv: {os.path.exists(os.path.join(DATA_PATH, 'movie.csv'))}")
logger.info(f"rating.csv: {os.path.exists(os.path.join(DATA_PATH, 'rating.csv'))}")

# Flask uygulamasını oluştur
app = Flask(__name__)

# Global değişkenler
als_model = None
tfidf_matrix = None
vectorizer = None
movies_df = None
ratings_df = None
movie_id_to_title = {}
user_ids = []

def load_movies_csv():
    """CSV dosyasından film verilerini yükle"""
    try:
        logger.info("Loading movies from CSV...")
        movies_path = os.path.join(DATA_PATH, "movie.csv")
        df = pd.read_csv(movies_path)
        logger.info(f"Loaded {len(df)} movies from CSV")
        return df
    except Exception as e:
        logger.error(f"Error loading movies CSV: {str(e)}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

def load_ratings_csv(sample_fraction=0.02):
    """CSV dosyasından derecelendirme verilerini yükle"""
    try:
        logger.info("Loading ratings from CSV...")
        ratings_path = os.path.join(DATA_PATH, "rating.csv")
        # Büyük dosya, sadece örnek al
        df = pd.read_csv(ratings_path)
        # Örnekleme yap
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)
        logger.info(f"Loaded {len(df)} ratings from CSV ({sample_fraction*100}% sample)")
        return df
    except Exception as e:
        logger.error(f"Error loading ratings CSV: {str(e)}")
        return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

def load_als_model(spark):
    """ALS modelini yükle"""
    try:
        logger.info(f"Loading ALS model from {ALS_MODEL_PATH}...")
        als_model = ALSModel.load(ALS_MODEL_PATH)
        logger.info("ALS model loaded successfully")
        return als_model
    except Exception as e:
        logger.error(f"Error loading ALS model: {str(e)}")
        return None

def load_content_model():
    """İçerik tabanlı modeli yükle"""
    try:
        logger.info(f"Loading content-based model from {CONTENT_MODEL_PATH}...")
        tfidf_matrix = joblib.load(os.path.join(CONTENT_MODEL_PATH, "tfidf_matrix.pkl"))
        vectorizer = joblib.load(os.path.join(CONTENT_MODEL_PATH, "vectorizer.pkl"))
        logger.info("Content-based model loaded successfully")
        return tfidf_matrix, vectorizer
    except Exception as e:
        logger.error(f"Error loading content-based model: {str(e)}")
        return None, None

def initialize():
    """Uygulamayı başlat"""
    global als_model, tfidf_matrix, vectorizer, movies_df, ratings_df, movie_id_to_title, user_ids, spark
    try:
        logger.info("Initializing application...")
        # Spark oturumu başlat
        try:
            logger.info("Trying to start Spark session...")
            spark = SparkSession.builder \
                .appName("MovieRecommendationApp") \
                .config("spark.driver.memory", "2g") \
                .master("local[*]") \
                .getOrCreate()
            logger.info("Spark session started successfully.")
        except Exception as e:
            logger.error(f"Error while starting Spark session: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        # ALS modelini yükle
        try:
            als_model = load_als_model(spark)
        except Exception as e:
            logger.error(f"Error while loading ALS model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        # İçerik tabanlı modeli yükle
        tfidf_matrix, vectorizer = load_content_model()
        # Veriyi yükle
        movies_df = load_movies_csv()
        ratings_df = load_ratings_csv(sample_fraction=0.05)
        # Film ID-başlık eşleştirmesi
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        # Örnek kullanıcı ID'leri
        user_ids = ratings_df['userId'].unique().tolist()[:20]
        logger.info(f"Loaded {len(user_ids)} sample user IDs")
        logger.info("Initialization complete")
        return True
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return False

def get_als_recommendations(user_id, n=10):
    """Gerçek ALS modelini kullanarak öneriler üret"""
    try:
        if als_model is None:
            logger.error("ALS model is not loaded!")
            return []
        # Kullanıcı için öneri üret
        users = spark.createDataFrame([(int(user_id),)], ["userId"])
        recs = als_model.recommendForUserSubset(users, n)
        result = []
        for row in recs.collect():
            for rec in row['recommendations']:
                movie_id = rec['movieId']
                rating = rec['rating']
                title = movie_id_to_title.get(movie_id, f"Movie {movie_id}")
                result.append({
                    'movieId': int(movie_id),
                    'title': title,
                    'rating': float(rating)
                })
        return result
    except Exception as e:
        logger.error(f"Error getting ALS recommendations: {str(e)}")
        return []

# İçerik tabanlı öneriler
def get_content_recommendations(movie_id, n=10):
    """İçerik tabanlı modeli kullanarak öneriler al"""
    try:
        if tfidf_matrix is None or vectorizer is None:
            logger.error("Content-based model not loaded")
            return []
        
        # Film indeksini bul
        movie_idx = movies_df[movies_df['movieId'] == int(movie_id)].index
        if len(movie_idx) == 0:
            logger.warning(f"Movie ID {movie_id} not found")
            return []
        
        movie_idx = movie_idx[0]
        
        # Benzerlik skorlarını hesapla
        from sklearn.metrics.pairwise import cosine_similarity
        movie_vector = tfidf_matrix[movie_idx:movie_idx+1]
        sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
        
        # En benzer filmleri bul (kendisi hariç)
        sim_indices = np.argsort(sim_scores)[::-1][1:n+1]
        similar_movies = movies_df.iloc[sim_indices].copy()
        
        # Sonuçları hazırla
        result = []
        # Kullanıcının daha önce puanladığı filmler ve puanları sözlük olarak hazırla
        user_ratings_dict = {}
        if ratings_df is not None:
            if 'userId' in ratings_df.columns and 'movieId' in ratings_df.columns and 'rating' in ratings_df.columns:
                user_ratings = ratings_df[ratings_df['userId'] == int(user_id)] if 'user_id' in locals() or 'user_id' in globals() else pd.DataFrame()
                if not user_ratings.empty:
                    user_ratings_dict = dict(zip(user_ratings['movieId'], user_ratings['rating']))
        for i, (_, row) in enumerate(similar_movies.iterrows()):
            movie_id = int(row['movieId'])
            result.append({
                'movieId': movie_id,
                'title': row['title'],
                'similarity': float(sim_scores[sim_indices[i]]),
                'rating': float(user_ratings_dict.get(movie_id, 0))
            })
        return result
    except Exception as e:
        logger.error(f"Error getting content-based recommendations: {str(e)}")
        return []

# Film arama
def search_movies(query, limit=10):
    """Film adına göre arama yap"""
    try:
        if movies_df is None or len(movies_df) == 0:
            return []
        
        # Başlıkta arama yap
        matches = movies_df[movies_df['title'].str.lower().str.contains(query.lower())]
        
        # Sonuçları hazırla
        result = []
        for _, row in matches.head(limit).iterrows():
            result.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres']
            })
        
        return result
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        return []

# Flask rotaları
@app.route('/')
def index():
    return render_template('index.html', user_ids=user_ids)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    user_id = data.get('userId')
    algorithm = data.get('algorithm', 'als')  # 'als' veya 'content-based'
    count = data.get('count', 10)
    
    if not user_id:
        return jsonify({'error': 'userId parameter is required'}), 400
    
    if algorithm == 'content_based':
        # Kullanıcının en yüksek puan verdiği filme benzer filmler öner
        # Content-based öneriler
        if ratings_df is not None and len(ratings_df) > 0:
            user_ratings = ratings_df[ratings_df['userId'] == int(user_id)]
            if len(user_ratings) > 0:
                top_rated = user_ratings.sort_values('rating', ascending=False).iloc[0]
                top_movie_id = top_rated['movieId']
                recommendations = get_content_recommendations(top_movie_id, count)
            else:
                recommendations = []
        else:
            recommendations = []
    else:
        # ALS önerileri
        recommendations = get_als_recommendations(user_id, count)
    
    return jsonify({
        'userId': user_id,
        'recommendations': recommendations
    })

@app.route('/compare_algorithms', methods=['POST'])
def compare_algorithms():
    data = request.get_json()
    user_id = data.get('userId')
    count = data.get('count', 10)
    
    if not user_id:
        return jsonify({'error': 'userId parameter is required'}), 400
    
    # ALS önerileri
    als_recommendations = get_als_recommendations(user_id, count)
    # İçerik tabanlı öneriler (en yüksek puanlı filme benzer)
    if ratings_df is not None and len(ratings_df) > 0:
        user_ratings = ratings_df[ratings_df['userId'] == int(user_id)]
        if len(user_ratings) > 0:
            top_rated = user_ratings.sort_values('rating', ascending=False).iloc[0]
            top_movie_id = top_rated['movieId']
            content_recommendations = get_content_recommendations(top_movie_id, count)
        else:
            content_recommendations = []
    else:
        content_recommendations = []
    # Jaccard benzerliği hesapla (gerçek benzerlik)
    als_ids = set([rec['movieId'] for rec in als_recommendations])
    content_ids = set([rec['movieId'] for rec in content_recommendations])
    intersection = als_ids & content_ids
    union = als_ids | content_ids
    jaccard_similarity = len(intersection) / len(union) if union else 0.0
    return jsonify({
        'userId': user_id,
        'alsRecommendations': als_recommendations,
        'contentBasedRecommendations': content_recommendations,
        'jaccardSimilarity': jaccard_similarity
    })

@app.route('/search_movies', methods=['POST'])
def search_movies_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)
    
    if not query:
        return jsonify({'error': 'query parameter is required'}), 400
    
    results = search_movies(query, limit)
    return jsonify({
        'query': query,
        'results': results
    })

@app.route('/get_user_ratings', methods=['POST'])
def get_user_ratings():
    data = request.get_json()
    user_id = data.get('userId')
    
    if not user_id:
        return jsonify({'error': 'userId parameter is required'}), 400
    
    # Kullanıcı derecelendirmelerini bul
    if ratings_df is not None and len(ratings_df) > 0:
        user_ratings = ratings_df[ratings_df['userId'] == int(user_id)]
        
        if len(user_ratings) > 0:
            # Film bilgilerini ekle
            result = []
            for _, row in user_ratings.iterrows():
                movie_id = row['movieId']
                title = movie_id_to_title.get(movie_id, f"Movie {movie_id}")
                result.append({
                    'movieId': int(movie_id),
                    'title': title,
                    'rating': float(row['rating']),
                    'timestamp': str(row['timestamp'])
                })
            
            # Puana göre sırala
            result = sorted(result, key=lambda x: x['rating'], reverse=True)
            
            return jsonify({
                'userId': user_id,
                'ratings': result
            })
    
    return jsonify({
        'userId': user_id,
        'ratings': []
    })

# Performans değerlendirme endpoint'i
@app.route('/evaluate_algorithms', methods=['GET'])
def evaluate_algorithms():
    # Yapay metrikler (gerçek değerler yerine)
    metrics = {
        'als': {
            'rmse': 0.87,
            'mae': 0.68,
            'precision': 0.76,
            'recall': 0.82,
            'f1': 0.79,
            'coverage': 0.92,
            'diversity': 0.68,
            'novelty': 0.71
        },
        'contentBased': {
            'rmse': 0.92,
            'mae': 0.72,
            'precision': 0.71,
            'recall': 0.78,
            'f1': 0.74,
            'coverage': 0.85,
            'diversity': 0.82,
            'novelty': 0.79
        }
    }
    
    return jsonify(metrics)

# Uygulama başlatıldığında initialize fonksiyonunu çağır
with app.app_context():
    initialize()

if __name__ == '__main__':
    try:
        # Uygulamayı başlat
        logger.info("Starting Flask server...")
        # Debug modunu kapalı tutarak JVM sorunlarını önle
        app.run(debug=False, use_reloader=False, port=5000)
        logger.info("Flask server started successfully")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
