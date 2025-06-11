from pyspark.ml.evaluation import RegressionEvaluator


def evaluate_model(als_model, ratings_spark_df):
    # Split data into training and testing sets
    ratings_train_df, ratings_test_df = ratings_spark_df.randomSplit([0.8, 0.2], seed=42)

    # Make predictions on the test set
    predictions = als_model.transform(ratings_test_df)

    # Evaluate the model using RMSE (Root Mean Squared Error)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")
