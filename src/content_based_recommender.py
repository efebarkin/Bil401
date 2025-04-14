import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql.functions import col, collect_list, udf
from pyspark.sql.types import StringType, ArrayType, DoubleType

class ContentBasedRecommender:
    def __init__(self, spark):
        """Content-based recommender sınıfını başlatır"""
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.movie_indices = None
        self.movie_titles = None
        self.movie_ids = None
        
    def simple_prepare_data(self, movies_df):
        """
        Daha basit bir içerik tabanlı filtreleme için film içeriklerini hazırlar
        Sadece film başlıkları ve türleri kullanılır
        """
        self.logger.info("Preparing simple content-based filtering data...")
        
        # Pandas DataFrame'e dönüştür - sadece gerekli sütunları al
        movies_pd = movies_df.select("movieId", "title", "genres").limit(5000).toPandas()
        
        # Tür bilgilerini string'e dönüştür
        movies_pd['genres_str'] = movies_pd['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        
        # İçerik özelliklerini birleştir - sadece başlık ve tür
        movies_pd['content'] = movies_pd['title'] + ' ' + movies_pd['genres_str']
        
        # TF-IDF vektörleştirici oluştur - daha az özellik
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # TF-IDF matrisini hesapla
        self.tfidf_matrix = self.vectorizer.fit_transform(movies_pd['content'])
        self.logger.info(f"Simple TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Film indekslerini, başlıklarını ve ID'lerini sakla
        self.movie_indices = pd.Series(movies_pd.index, index=movies_pd['movieId'])
        self.movie_titles = pd.Series(movies_pd['title'].values, index=movies_pd['movieId'])
        self.movie_ids = movies_pd['movieId'].values
        
        return self.tfidf_matrix
        
    def prepare_data(self, movies_df, tags_df, genome_scores_df, genome_tags_df):
        """
        Film içeriklerini hazırlar
        """
        self.logger.info("Preparing data for content-based filtering...")
        
        # Pandas DataFrame'e dönüştür
        movies_pd = movies_df.toPandas()
        
        # Tag bilgilerini birleştir
        if tags_df is not None:
            tags_pd = tags_df.toPandas()
            # Film başına tüm tag'leri birleştir
            tags_by_movie = tags_pd.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
            # Film dataframe'i ile birleştir
            movies_pd = pd.merge(movies_pd, tags_by_movie, on='movieId', how='left')
            movies_pd['tag'] = movies_pd['tag'].fillna('')
        else:
            movies_pd['tag'] = ''
            
        # Genome tag bilgilerini ekle
        if genome_scores_df is not None and genome_tags_df is not None:
            # Spark ile işlem yap
            # Tag ID'lerini tag isimlerine dönüştür
            genome_tags_map = {row['tagId']: row['tag'] for row in genome_tags_df.collect()}
            
            # UDF ile tag ID'lerini tag isimlerine dönüştür
            @udf(StringType())
            def tag_id_to_name(tag_id):
                return genome_tags_map.get(tag_id, "")
            
            # Relevance değeri yüksek olan tag'leri seç (örn: 0.5'ten büyük)
            relevant_tags = genome_scores_df.filter(col("relevance") > 0.5)
            relevant_tags = relevant_tags.withColumn("tag", tag_id_to_name(col("tagId")))
            
            # Film başına tag'leri birleştir
            film_tags = relevant_tags.groupBy("movieId").agg(
                collect_list("tag").alias("genome_tags")
            )
            
            # UDF ile tag listesini string'e dönüştür
            @udf(StringType())
            def tags_to_string(tags):
                return " ".join(tags) if tags else ""
            
            film_tags = film_tags.withColumn("genome_tags_str", tags_to_string(col("genome_tags")))
            
            # Pandas'a dönüştür ve birleştir
            film_tags_pd = film_tags.select("movieId", "genome_tags_str").toPandas()
            movies_pd = pd.merge(movies_pd, film_tags_pd, on='movieId', how='left')
            movies_pd['genome_tags_str'] = movies_pd['genome_tags_str'].fillna('')
        else:
            movies_pd['genome_tags_str'] = ''
        
        # Tür bilgilerini string'e dönüştür
        movies_pd['genres_str'] = movies_pd['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        
        # İçerik özelliklerini birleştir
        movies_pd['content'] = movies_pd['title'] + ' ' + movies_pd['genres_str'] + ' ' + movies_pd['tag'] + ' ' + movies_pd['genome_tags_str']
        
        # TF-IDF vektörleştirici oluştur
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # TF-IDF matrisini hesapla
        self.tfidf_matrix = self.vectorizer.fit_transform(movies_pd['content'])
        self.logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Film indekslerini, başlıklarını ve ID'lerini sakla
        self.movie_indices = pd.Series(movies_pd.index, index=movies_pd['movieId'])
        self.movie_titles = pd.Series(movies_pd['title'].values, index=movies_pd['movieId'])
        self.movie_ids = movies_pd['movieId'].values
        
        return self.tfidf_matrix
    
    def save_model(self, path):
        """
        İçerik tabanlı modeli belirtilen yola kaydeder
        """
        import joblib
        import os
        
        try:
            self.logger.info(f"Saving content-based model to {path}...")
            # Klasörü oluştur (yoksa)
            os.makedirs(path, exist_ok=True)
            
            # Model bileşenlerini kaydet
            joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
            joblib.dump(self.tfidf_matrix, os.path.join(path, "tfidf_matrix.pkl"))
            joblib.dump(self.movie_indices, os.path.join(path, "movie_indices.pkl"))
            joblib.dump(self.movie_titles, os.path.join(path, "movie_titles.pkl"))
            joblib.dump(self.movie_ids, os.path.join(path, "movie_ids.pkl"))
            
            self.logger.info(f"Content-based model successfully saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving content-based model: {str(e)}")
            return False
    
    def load_model(self, path):
        """
        Belirtilen yoldan içerik tabanlı modeli yükler
        """
        import joblib
        import os
        
        try:
            self.logger.info(f"Loading content-based model from {path}...")
            
            # Model bileşenlerini yükle
            self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
            self.tfidf_matrix = joblib.load(os.path.join(path, "tfidf_matrix.pkl"))
            self.movie_indices = joblib.load(os.path.join(path, "movie_indices.pkl"))
            self.movie_titles = joblib.load(os.path.join(path, "movie_titles.pkl"))
            self.movie_ids = joblib.load(os.path.join(path, "movie_ids.pkl"))
            
            self.logger.info(f"Content-based model successfully loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading content-based model: {str(e)}")
            return False
            
    def get_similar_movies(self, movie_id, n=10):
        """
        Benzer filmleri döndürür
        """
        # Film ID'sinin indeksini bul
        if movie_id not in self.movie_indices:
            self.logger.warning(f"Movie ID {movie_id} not found in the dataset")
            return []
            
        idx = self.movie_indices[movie_id]
        
        # Tüm filmlerle kosinüs benzerliğini hesapla
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Kendisi hariç en benzer n filmi al
        sim_scores_with_indices = list(enumerate(sim_scores))
        sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
        sim_scores_with_indices = sim_scores_with_indices[1:n+1]  # İlk film kendisi olduğu için 1'den başla
        
        # Film indekslerini ve benzerlik skorlarını al
        movie_indices = [i[0] for i in sim_scores_with_indices]
        similarity_scores = [i[1] for i in sim_scores_with_indices]
        
        # Sonuçları döndür
        result = pd.DataFrame({
            'movieId': [self.movie_ids[i] for i in movie_indices],
            'title': [self.movie_titles[self.movie_ids[i]] for i in movie_indices],
            'similarity': similarity_scores
        })
        
        return result
    
    def get_recommendations_for_user(self, user_id, ratings_df, n=10):
        """
        Kullanıcının beğendiği filmlere benzer filmleri önerir
        """
        # Kullanıcının yüksek puanladığı filmleri bul (örn: 4 ve üzeri)
        user_ratings = ratings_df.filter((col("userId") == user_id) & (col("rating") >= 4.0))
        
        if user_ratings.count() == 0:
            self.logger.warning(f"User {user_id} has no high ratings")
            return pd.DataFrame()
        
        # Kullanıcının beğendiği filmlerin ID'lerini al
        liked_movie_ids = [row['movieId'] for row in user_ratings.collect()]
        
        # Her beğenilen film için benzer filmleri bul ve birleştir
        all_similar_movies = pd.DataFrame()
        
        for movie_id in liked_movie_ids:
            similar_movies = self.get_similar_movies(movie_id, n=n)
            if not similar_movies.empty:
                all_similar_movies = pd.concat([all_similar_movies, similar_movies])
        
        # Tekrarlanan filmleri çıkar ve benzerlik skorlarına göre sırala
        if not all_similar_movies.empty:
            all_similar_movies = all_similar_movies.drop_duplicates(subset=['movieId'])
            all_similar_movies = all_similar_movies.sort_values('similarity', ascending=False)
            
            # Kullanıcının zaten izlediği filmleri çıkar
            watched_movie_ids = [row['movieId'] for row in ratings_df.filter(col("userId") == user_id).select("movieId").collect()]
            all_similar_movies = all_similar_movies[~all_similar_movies['movieId'].isin(watched_movie_ids)]
            
            # En iyi n öneriyi döndür
            return all_similar_movies.head(n)
        
        return pd.DataFrame()
    
    def evaluate(self, ratings_df, test_users=100, n_recommendations=10):
        """
        Content-based filtreleme modelini değerlendirir
        """
        self.logger.info("Evaluating content-based filtering model...")
        
        # Test için rastgele kullanıcılar seç
        test_user_ids = [row['userId'] for row in ratings_df.select("userId").distinct().limit(test_users).collect()]
        
        # Her kullanıcı için öneriler oluştur ve değerlendir
        precision_sum = 0
        recall_sum = 0
        count = 0
        
        for user_id in test_user_ids:
            # Kullanıcının verilerini eğitim ve test setlerine ayır
            user_ratings = ratings_df.filter(col("userId") == user_id)
            
            if user_ratings.count() < 5:  # En az 5 derecelendirmesi olan kullanıcıları değerlendir
                continue
                
            # Kullanıcının verilerini %80 eğitim, %20 test olarak ayır
            train_ratings, test_ratings = user_ratings.randomSplit([0.8, 0.2], seed=42)
            
            if train_ratings.count() == 0 or test_ratings.count() == 0:
                continue
            
            # Eğitim verilerine göre öneriler oluştur
            recommendations = self.get_recommendations_for_user(user_id, train_ratings, n=n_recommendations)
            
            if recommendations.empty:
                continue
            
            # Test setindeki beğenilen filmler (4.0 ve üzeri)
            liked_test_movies = [row['movieId'] for row in test_ratings.filter(col("rating") >= 4.0).select("movieId").collect()]
            
            if not liked_test_movies:
                continue
            
            # Önerilen filmler
            recommended_movies = recommendations['movieId'].tolist()
            
            # Precision ve recall hesapla
            relevant_recommendations = set(recommended_movies) & set(liked_test_movies)
            precision = len(relevant_recommendations) / len(recommended_movies) if recommended_movies else 0
            recall = len(relevant_recommendations) / len(liked_test_movies) if liked_test_movies else 0
            
            precision_sum += precision
            recall_sum += recall
            count += 1
        
        # Ortalama precision ve recall
        avg_precision = precision_sum / count if count > 0 else 0
        avg_recall = recall_sum / count if count > 0 else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        self.logger.info(f"Content-Based Filtering Evaluation:")
        self.logger.info(f"Average Precision: {avg_precision:.4f}")
        self.logger.info(f"Average Recall: {avg_recall:.4f}")
        self.logger.info(f"F1 Score: {f1_score:.4f}")
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": f1_score
        }
