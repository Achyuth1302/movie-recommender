from pyspark.ml.recommendation import ALS


def train_als_model(spark, ratings_spark_df):
    # Prepare the data for ALS model
    ratings_train_df, ratings_test_df = ratings_spark_df.randomSplit([0.8, 0.2], seed=42)

    # Train the ALS model
    als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="movie_id", ratingCol="rating",
              coldStartStrategy="drop")
    als_model = als.fit(ratings_train_df)

    return als_model


def generate_recommendations(als_model, user_id, num_recommendations=5):
    # Generate top movie recommendations for a specific user
    user_recommendations = als_model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["user_id"]),
                                                            num_recommendations)

    # Extract movie recommendations
    recommended_movies = user_recommendations.collect()[0].recommendations
    movie_ids = [row['movie_id'] for row in recommended_movies]

    return movie_ids
