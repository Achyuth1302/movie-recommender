from src.data_loader import load_data, load_spark_data
from src.eda import perform_eda
from src.recommendation_system import train_als_model, generate_recommendations
from src.evaluation import evaluate_model

def main():
    # Load data using Pandas
    ratings_df, movies_df = load_data()

    # Perform EDA (Exploratory Data Analysis)
    perform_eda(ratings_df, movies_df)

    # Initialize Spark session and load data
    spark, ratings_spark_df = load_spark_data()

    # Train ALS (Matrix Factorization) model
    als_model = train_als_model(spark, ratings_spark_df)

    # Generate movie recommendations for a specific user
    recommendations = generate_recommendations(als_model, user_id=1)
    print("Recommendations for user 1:", recommendations)

    # Evaluate the model's performance
    evaluate_model(als_model, ratings_spark_df)

if __name__ == '__main__':
    main()
