from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, avg
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt

spark = (
    SparkSession.builder.master("local[4]")
    .appName("COM6012 Spark Assignment Q4")
    .config("spark.local.dir", "/mnt/parscratch/users/acr23nm")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("WARN") 

# Load the ratings data
ratings = spark.read.csv("./Data/ml-20m/ratings.csv", header=True, inferSchema=True)

# Sort the data by timestamp
ratings_sorted = ratings.orderBy("timestamp")

# Count total number of ratings
total_count = ratings_sorted.count()

# Calculate split indices
train_40_count = int(total_count * 0.4)
train_60_count = int(total_count * 0.6)
train_80_count = int(total_count * 0.8)

# Create training and test datasets for three different splits
train_40 = ratings_sorted.limit(train_40_count)
test_40 = ratings_sorted.subtract(train_40)

train_60 = ratings_sorted.limit(train_60_count)
test_60 = ratings_sorted.subtract(train_60)

train_80 = ratings_sorted.limit(train_80_count)
test_80 = ratings_sorted.subtract(train_80)


# Function to compute RMSE, MSE, and MAE
def compute_metrics(predictions):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
    mse = evaluator.evaluate(predictions)
    evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    mae = evaluator.evaluate(predictions)
    return rmse, mse, mae

# Set the seed
seed = 230123803
################ ALS SETTING 1 ##############

als_1 = ALS( userCol="userId", itemCol="movieId", seed=seed, coldStartStrategy="drop")
model_40_1 = als_1.fit(train_40)
model_60_1 = als_1.fit(train_60)
model_80_1 = als_1.fit(train_80)

# Predictions
predictions_40_1 = model_40_1.transform(test_40)
predictions_60_1 = model_60_1.transform(test_60)
predictions_80_1 = model_80_1.transform(test_80)

# Compute metrics
print("Setting 1")
rmse_40_1, mse_40_1, mae_40_1 = compute_metrics(predictions_40_1)
print(f"RMSE for 40 split: {rmse_40_1} | MSE for 40 split: {mse_40_1} | MAE for 40 split: {mae_40_1}")
rmse_60_1, mse_60_1, mae_60_1 = compute_metrics(predictions_60_1)
print(f"RMSE for 60 split: {rmse_60_1} | MSE for 60 split: {mse_60_1} | MAE for 60 split: {mae_60_1}")
rmse_80_1, mse_80_1, mae_80_1 = compute_metrics(predictions_80_1)
print(f"RMSE for 80 split: {rmse_80_1} | MSE for 80 split: {mse_80_1} | MAE for 80 split: {mae_80_1}")

#################### ALS SETTING 2 ####################
als_2 = ALS(regParam=0.03, userCol="userId", itemCol="movieId", coldStartStrategy="drop", seed=seed, rank = 50)
model_40_2 = als_2.fit(train_40)
model_60_2 = als_2.fit(train_60)
model_80_2 = als_2.fit(train_80)

# Predictions
predictions_40_2 = model_40_2.transform(test_40)
predictions_60_2 = model_60_2.transform(test_60)
predictions_80_2 = model_80_2.transform(test_80)

# Compute metrics
print("Setting 2")
rmse_40_2, mse_40_2, mae_40_2 = compute_metrics(predictions_40_2)
print(f"RMSE for 40 split: {rmse_40_2} | MSE for 40 split: {mse_40_2} | MAE for 40 split: {mae_40_2}")
rmse_60_2, mse_60_2, mae_60_2 = compute_metrics(predictions_60_2)
print(f"RMSE for 60 split: {rmse_60_2} | MSE for 60 split: {mse_60_2} | MAE for 60 split: {mae_60_2}")
rmse_80_2, mse_80_2, mae_80_2 = compute_metrics(predictions_80_2)
print(f"RMSE for 80 split: {rmse_80_2} | MSE for 80 split: {mse_80_2} | MAE for 80 split: {mae_80_2}")


# Create a DataFrame to store the results
results = [
    ("40", "Setting 1", rmse_40_1, mse_40_1, mae_40_1),
    ("40", "Setting 2", rmse_40_2, mse_40_2, mae_40_2),
    ("60", "Setting 1", rmse_60_1, mse_60_1, mae_60_1),
    ("60", "Setting 2", rmse_60_2, mse_60_2, mae_60_2),
    ("80", "Setting 1", rmse_80_1, mse_80_1, mae_80_1),
    ("80", "Setting 2", rmse_80_2, mse_80_2, mae_80_2),
]

results_df = pd.DataFrame(results, columns=["Split", "Setting", "RMSE", "MSE", "MAE"])
print("\nResults for ALS\n")
# Print the DataFrame to see the table
print(results_df)

# Plot the results
# Pivot the DataFrame for easier plotting
rmse_pivot = results_df.pivot(index="Split", columns="Setting", values="RMSE")
mse_pivot = results_df.pivot(index="Split", columns="Setting", values="MSE")
mae_pivot = results_df.pivot(index="Split", columns="Setting", values="MAE")


fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# RMSE Plot
rmse_pivot.plot(kind='bar', ax=ax[0])
ax[0].set_title("RMSE Comparison")
ax[0].set_ylabel("RMSE")
ax[0].set_xlabel("Split")

# MSE Plot
mse_pivot.plot(kind='bar', ax=ax[1])
ax[1].set_title("MSE Comparison")
ax[1].set_ylabel("MSE")
ax[1].set_xlabel("Split")

# MAE Plot
mae_pivot.plot(kind='bar', ax=ax[2])
ax[2].set_title("MAE Comparison")
ax[2].set_ylabel("MAE")
ax[2].set_xlabel("Split")

plt.tight_layout()
plt.savefig("./Output/Q4_figA.jpg")


##################### Question 4B-1 #####################
######################
# Extract user factors for each model
user_factors_40_2 = model_40_2.userFactors
user_factors_60_2 = model_60_2.userFactors
user_factors_80_2 = model_80_2.userFactors

# Cluster users using k-means
def cluster_users(user_factors, k=25, seed=seed):
    kmeans = KMeans(k=k, seed=seed, featuresCol="features")
    return kmeans.fit(user_factors).transform(user_factors)

clusters_40_2 = cluster_users(user_factors_40_2)
clusters_60_2 = cluster_users(user_factors_60_2)
clusters_80_2 = cluster_users(user_factors_80_2)

# Identify top five clusters
def top_five_clusters(clustered_data):
    return clustered_data.groupBy("prediction").count().orderBy("count", ascending=False).head(5)

top_clusters_40_2 = top_five_clusters(clusters_40_2)
top_clusters_60_2 = top_five_clusters(clusters_60_2)
top_clusters_80_2 = top_five_clusters(clusters_80_2)


# Preparing data
cluster_sizes = {
    "40%": [row['count'] for row in top_clusters_40_2],
    "60%": [row['count'] for row in top_clusters_60_2],
    "80%": [row['count'] for row in top_clusters_80_2]
}
cluster_index = [f"Cluster {i+1}" for i in range(5)]
cluster_df = pd.DataFrame(cluster_sizes, index=cluster_index)

print("\nTop Five Largest User Clusters per Time Split\n")
print(cluster_df)

# Visualize the cluster sizes

fig, ax = plt.subplots(figsize=(10, 8))
cluster_df.plot(kind='bar', ax=ax)
ax.set_title('Top Five Largest User Clusters per Time Split')
ax.set_ylabel('Number of Users')
ax.set_xlabel('Clusters')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("./Output/Q4_figB1.jpg")

############################ Question 4B-2 ############################
# Identify the largest cluster
def get_largest_cluster(clustered_data):
    # Finding the cluster with the maximum count
    return clustered_data.groupBy("prediction").count().orderBy("count", ascending=False).first()["prediction"]

largest_cluster_40 = get_largest_cluster(clusters_40_2)
largest_cluster_60 = get_largest_cluster(clusters_60_2)
largest_cluster_80 = get_largest_cluster(clusters_80_2)

def get_top_movies(ratings, user_factors, largest_cluster):
    # Filter for users in the largest cluster
    cluster_users = user_factors.filter(col("prediction") == largest_cluster).select(col("id").alias("userId"))
    
    # Join with ratings to get movies rated by these users
    cluster_ratings = ratings.join(cluster_users, "userId")

    # Group by movieId, calculate average rating and filter for high-rated movies
    movies_largest_cluster = cluster_ratings.groupBy("movieId").agg(avg("rating").alias("avg_rating"))

    # Filter for movies with an average rating >= 4
    top_movies = movies_largest_cluster.filter(col("avg_rating") >= 4)
    return top_movies

top_movies_40 = get_top_movies(ratings, clusters_40_2, largest_cluster_40)
top_movies_60 = get_top_movies(ratings, clusters_60_2, largest_cluster_60)
top_movies_80 = get_top_movies(ratings, clusters_80_2, largest_cluster_80)

# Load movies.csv
movies = spark.read.csv("./Data/ml-20m/movies.csv", header=True, inferSchema=True)

def get_popular_genres(top_movies, movies):
    # Join with movies DataFrame to get genre information
    movie_genres = top_movies.join(movies, top_movies.movieId == movies.movieId).select("genres")
    
    # Explode genres into separate rows
    movie_genres = movie_genres.withColumn("genre", explode(split(col("genres"), "[|]")))
    
    # Count occurrences of each genre and get top 10
    popular_genres = movie_genres.groupBy("genre").count().orderBy("count", ascending=False).head(10)
    return [genre['genre'] for genre in popular_genres]

popular_genres_40 = get_popular_genres(top_movies_40, movies)
popular_genres_60 = get_popular_genres(top_movies_60, movies)
popular_genres_80 = get_popular_genres(top_movies_80, movies)

results = {
    "40%": popular_genres_40,
    "60%": popular_genres_60,
    "80%": popular_genres_80
}
results_df = pd.DataFrame(results)

print("\nTop Ten Popular Genres per Time Split\n")
# Print the DataFrame to see the table
print(results_df)

# Visualize the genre popularity
fig, ax = plt.subplots(figsize=(10, 8))
results_df.plot(kind='bar', ax=ax)
ax.set_title('Top Ten Popular Genres per Time Split')
ax.set_ylabel('Genres')
ax.set_xlabel('Top Genres')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./Output/Q4_figB2.jpg")


sc.stop()