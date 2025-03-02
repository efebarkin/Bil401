import logging
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        """Model eğitici sınıfını başlatır"""
        self.logger = logging.getLogger(__name__)
        
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
        try:
            # Hiperparametre grid'ini oluştur
            param_grid = ParamGridBuilder() \
                .addGrid(self.als.rank, [10, 50, 100]) \
                .addGrid(self.als.maxIter, [5, 10]) \
                .addGrid(self.als.regParam, [0.01, 0.1]) \
                .build()
            
            # Cross validator oluştur
            cv = CrossValidator(
                estimator=self.als,
                estimatorParamMaps=param_grid,
                evaluator=self.evaluator,
                numFolds=num_folds
            )
            
            # Veriyi eğitim ve test setlerine ayır
            training, test = ratings_df.randomSplit([0.8, 0.2], seed=42)
            
            # Modeli eğit
            self.logger.info("Training model with cross validation...")
            cv_model = cv.fit(training)
            
            # Test seti üzerinde tahmin yap
            predictions = cv_model.transform(test)
            
            # Metrikleri hesapla
            rmse = self.evaluator.evaluate(predictions)
            self.evaluator.setMetricName("mae")
            mae = self.evaluator.evaluate(predictions)
            
            self.logger.info(f"Best model RMSE = {rmse}")
            self.logger.info(f"Best model MAE = {mae}")
            
            return cv_model.bestModel, predictions, rmse, mae
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
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
        
    def get_top_recommendations(self, model, user_id, n=10):
        """Belirli bir kullanıcı için en iyi n film önerisini döndürür"""
        user_recs = model.recommendForUser(user_id, n)
        return user_recs