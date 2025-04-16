import logging
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, spark=None):
        """Model eğitici sınıfını başlatır"""
        self.logger = logging.getLogger(__name__)
        self.spark = spark
        
        # ALS modelini yapılandır
        self.als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        
        # Değerlendirici oluştur
        self.evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
    
    def train_with_cross_validation(self, ratings_df, num_folds=3):
        """Cross validation ile en iyi modeli eğitir"""
        import time
        total_start_time = time.time()
        
        try:
            self.logger.info("=== ALS Model Training Process Started ===")
            self.logger.info(f"Input data: {ratings_df.count()} ratings from {ratings_df.select('userId').distinct().count()} users on {ratings_df.select('movieId').distinct().count()} movies")
            
            # Hiperparametre grid'ini oluştur
            self.logger.info("Creating hyperparameter grid...")
            param_grid = ParamGridBuilder() \
                .addGrid(self.als.rank, [10, 50, 100]) \
                .addGrid(self.als.maxIter, [5, 10]) \
                .addGrid(self.als.regParam, [0.01, 0.1]) \
                .build()
            
            total_combinations = len(param_grid)
            self.logger.info(f"Created {total_combinations} hyperparameter combinations for grid search")
            self.logger.info(f"Parameter grid: rank=[10, 50, 100], maxIter=[5, 10], regParam=[0.01, 0.1]")
            
            # Cross validator oluştur
            self.logger.info(f"Setting up {num_folds}-fold cross validation...")
            cv = CrossValidator(
                estimator=self.als,
                estimatorParamMaps=param_grid,
                evaluator=self.evaluator,
                numFolds=num_folds,
                seed=42
            )
            
            # Veriyi eğitim ve test setlerine ayır
            self.logger.info("Splitting data into training (80%) and test (20%) sets...")
            split_start_time = time.time()
            training, test = ratings_df.randomSplit([0.8, 0.2], seed=42)
            training.cache()  # Eğitim verisini önbelleğe al
            test.cache()      # Test verisini önbelleğe al
            
            train_count = training.count()
            test_count = test.count()
            split_time = time.time() - split_start_time
            
            self.logger.info(f"Data split complete in {split_time:.2f} seconds")
            self.logger.info(f"Training set: {train_count} ratings ({train_count/ratings_df.count()*100:.1f}%)")
            self.logger.info(f"Test set: {test_count} ratings ({test_count/ratings_df.count()*100:.1f}%)")
            
            # Modeli eğit
            self.logger.info("Starting model training with cross validation...\n"
                          f"This may take a while as we're training {total_combinations} models with {num_folds} folds each...")
            train_start_time = time.time()
            cv_model = cv.fit(training)
            train_time = time.time() - train_start_time
            
            self.logger.info(f"Model training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
            
            # En iyi modelin parametrelerini göster
            best_model = cv_model.bestModel
            best_rank = best_model.rank
            best_max_iter = best_model._java_obj.parent().getMaxIter()
            best_reg_param = best_model._java_obj.parent().getRegParam()
            
            self.logger.info(f"Best model parameters: rank={best_rank}, maxIter={best_max_iter}, regParam={best_reg_param}")
            
            # Test seti üzerinde tahmin yap
            self.logger.info("Generating predictions on test set...")
            predict_start_time = time.time()
            predictions = cv_model.transform(test)
            predict_time = time.time() - predict_start_time
            
            self.logger.info(f"Predictions generated in {predict_time:.2f} seconds")
            
            # Metrikleri hesapla
            self.logger.info("Calculating evaluation metrics...")
            self.evaluator.setMetricName("rmse")
            rmse = self.evaluator.evaluate(predictions)
            self.evaluator.setMetricName("mae")
            mae = self.evaluator.evaluate(predictions)
            
            # R-squared hesapla
            self.evaluator.setMetricName("r2")
            r2 = self.evaluator.evaluate(predictions)
            
            self.logger.info(f"Evaluation metrics:\n"
                          f"  RMSE = {rmse:.4f} (lower is better)\n"
                          f"  MAE = {mae:.4f} (lower is better)\n"
                          f"  R² = {r2:.4f} (higher is better)")
            
            # Toplam süreyi hesapla
            total_time = time.time() - total_start_time
            self.logger.info(f"Total training process completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            self.logger.info("=== ALS Model Training Process Completed ===")
            
            return best_model, rmse, mae, r2
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def plot_predictions(self, predictions):
        """Tahminleri görselleştirir"""
        # Pandas DataFrame'e çevir
        pred_df = predictions.select("rating", "prediction").toPandas()
        
        # Scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pred_df, x="rating", y="prediction", alpha=0.5)
        plt.plot([0, 5], [0, 5], 'r--')  # Diagonal line
        plt.title("Actual vs Predicted Ratings")
        plt.xlabel("Actual Rating")
        plt.ylabel("Predicted Rating")
        plt.savefig("prediction_scatter.png")
        plt.close()
        
        # Residual plot
        plt.figure(figsize=(10, 6))
        pred_df["residual"] = pred_df["rating"] - pred_df["prediction"]
        sns.histplot(data=pred_df, x="residual", bins=50)
        plt.title("Residual Distribution")
        plt.xlabel("Residual (Actual - Predicted)")
        plt.ylabel("Count")
        plt.savefig("residual_distribution.png")
        plt.close()
        
    def save_model(self, model, path):
        """
        ALS modelini belirtilen yola kaydeder
        """
        try:
            self.logger.info(f"Saving ALS model to {path}...")
            model.save(path)
            self.logger.info(f"Model successfully saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path):
        """
        Belirtilen yoldan ALS modelini yükler
        """
        from pyspark.ml.recommendation import ALSModel
        
        try:
            self.logger.info(f"Loading ALS model from {path}...")
            model = ALSModel.load(path)
            self.logger.info(f"Model successfully loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_top_recommendations(self, model, user_id, n=10):
        """Belirli bir kullanıcı için en iyi n film önerisini döndürür"""
        user_recs = model.recommendForUser(user_id, n)
        return user_recs