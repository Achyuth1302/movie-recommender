import pandas as pd
from pyspark.sql import SparkSession


import pandas as pd

def load_data():
    # Load ratings and movies data using Pandas with the correct encoding
    ratings_df = pd.read_csv('data/movielens/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='ISO-8859-1')
    movies_df = pd.read_csv('data/movielens/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='ISO-8859-1')

    return ratings_df, movies_df



def load_spark_data():
    # Initialize a Spark session
    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    # Load the ratings data into Spark DataFrame
    ratings_spark_df = spark.read.csv('data/movielens/ml-1m/ratings.dat', sep='::', inferSchema=True,
                                      header=False).toDF('user_id', 'movie_id', 'rating', 'timestamp')

    return spark, ratings_spark_df
