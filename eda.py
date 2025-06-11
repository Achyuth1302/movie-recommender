import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(ratings_df, movies_df):
    # EDA on ratings data
    print("Ratings Data Information:")
    print(ratings_df.info())

    # Number of ratings per movie
    movie_ratings_count = ratings_df.groupby('movie_id').size().reset_index(name='ratings_count')
    movie_ratings_count = movie_ratings_count.sort_values(by='ratings_count', ascending=False)

    # Plot top 10 movies by number of ratings
    top_movies = movie_ratings_count.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ratings_count', y='movie_id', data=top_movies, orient='h')
    plt.title('Top 10 Movies by Number of Ratings')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie ID')
    plt.show()

    # EDA on movie genres
    genre_counts = movies_df['genres'].str.split('|', expand=True).stack().value_counts()
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.show()
