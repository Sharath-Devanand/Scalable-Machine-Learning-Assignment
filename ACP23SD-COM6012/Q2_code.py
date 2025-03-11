from pyspark.sql import SparkSession
from sklearn.datasets import fetch_openml
from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .getOrCreate()

sc = spark.sparkContext

spark.sparkContext.setLogLevel("WARN")

df_freq_sklearn = fetch_openml(data_id=41214, as_frame=True)
df_freq = df_freq_sklearn.data

df_spark = spark.createDataFrame(df_freq)

# Add the hasClaim column
df_spark = df_spark.withColumn("hasClaim", when(df_spark["ClaimNb"] > 0, 1).otherwise(0).cast(IntegerType()))


# Define the seed for stratified split
seed = 23788

# Split the dataset into training and test sets
train_df, test_df = df_spark.randomSplit([0.7, 0.3], seed=seed)

# Stratified split on hasClaim column
train_has_claim = train_df.filter(train_df["hasClaim"] == 1)
train_no_claim = train_df.filter(train_df["hasClaim"] == 0)

test_has_claim = test_df.filter(test_df["hasClaim"] == 1)
test_no_claim = test_df.filter(test_df["hasClaim"] == 0)

fractions = {
    1: train_has_claim.count() / train_df.count(),
    0: train_no_claim.count() / train_df.count()
}

# Perform stratified sampling
train_has_claim_strat = train_has_claim.sampleBy("hasClaim", fractions, seed)
train_no_claim_strat = train_no_claim.sampleBy("hasClaim", fractions, seed)

# Combine the samples
train_stratified = train_has_claim_strat.union(train_no_claim_strat)

# Repeat the same process for the test set
test_fractions = {
    1: test_has_claim.count() / test_df.count(),
    0: test_no_claim.count() / test_df.count()
}

test_has_claim_strat = test_has_claim.sampleBy("hasClaim", test_fractions, seed)
test_no_claim_strat = test_no_claim.sampleBy("hasClaim", test_fractions, seed)

test_stratified = test_has_claim_strat.union(test_no_claim_strat)

# Define numeric and categorical features
numeric_features = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
categorical_features = ['VehBrand', 'VehGas', 'Region','Area']

# Define stages for the pipeline
stages = []

# Assemble numeric features into a single vector
assembler_num = VectorAssembler(inputCols=numeric_features, outputCol="num_features")
stages += [assembler_num]

# Standardize numeric features
scaler = StandardScaler(inputCol="num_features", outputCol="scaled_features", withMean=True, withStd=True)
stages += [scaler]

# Index string categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_features]
stages += indexers

# One-hot encode indexed categorical features
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_onehot") for col in categorical_features]
stages += encoders


# Assemble all features into a single vector
input_cols = ["scaled_features"] + [categorical_col + "_onehot" for categorical_col in categorical_features]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
stages += [assembler]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Sample a small subset from the training set
sampled_train = train_stratified.sample(False, 0.1, seed=seed)

sampled_train.printSchema()

# Fit the pipeline to the sampled training data
pipeline_model = pipeline.fit(sampled_train)

# Transform the sampled training data using the fitted pipeline
sampled_train_transformed = pipeline_model.transform(sampled_train)

# Define the Poisson Regression model
poisson_model = GeneralizedLinearRegression(family="poisson", labelCol="ClaimNb", featuresCol="features")

# Define the Logistic Regression model
logistic_model = LogisticRegression(labelCol="hasClaim", featuresCol="features")

# Define the evaluator for Poisson Regression
evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")

# Define the evaluator for Logistic Regression
evaluator_logistic = BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Define the grid of parameters to search for Poisson Regression
paramGrid_poisson = ParamGridBuilder() \
    .addGrid(poisson_model.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

# Define the grid of parameters to search for Logistic Regression
paramGrid_logistic = ParamGridBuilder() \
    .addGrid(logistic_model.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

# Create the cross-validator for Poisson Regression
cv_poisson = CrossValidator(estimator=poisson_model,
                            estimatorParamMaps=paramGrid_poisson,
                            evaluator=evaluator_poisson,
                            numFolds=5,
                            seed=seed)

# Create the cross-validator for Logistic Regression
cv_logistic = CrossValidator(estimator=logistic_model,
                             estimatorParamMaps=paramGrid_logistic,
                             evaluator=evaluator_logistic,
                             numFolds=5,
                             seed=seed)

# Fit the cross-validators to the transformed training data
cv_model_poisson = cv_poisson.fit(sampled_train_transformed)
cv_model_logistic = cv_logistic.fit(sampled_train_transformed)

# Define the optimal hyperparameters obtained from cross-validation
optimal_regParam_poisson = cv_model_poisson.bestModel._java_obj.getRegParam()
optimal_regParam_logistic = cv_model_logistic.bestModel._java_obj.getRegParam()

pipeline = Pipeline(stages=stages)

# Fit the pipeline to the full dataset
pipeline_model = pipeline.fit(train_stratified)

# Transform the full dataset using the fitted pipeline
df_transformed = pipeline_model.transform(train_stratified)

# Transform the training and test data using the fitted pipeline
train_transformed = pipeline_model.transform(train_stratified)
test_transformed = pipeline_model.transform(test_df)


# Train Poisson Regression model on full dataset
poisson_model = GeneralizedLinearRegression(family="poisson", labelCol="ClaimNb", featuresCol="features", regParam=optimal_regParam_poisson)
poisson_model_trained = poisson_model.fit(df_transformed)

# Train Logistic Regression model with L1 regularization on full dataset
logistic_model_l1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", regParam=optimal_regParam_logistic, elasticNetParam=1.0)
logistic_model_l1_trained = logistic_model_l1.fit(df_transformed)

# Train Logistic Regression model with L2 regularization on full dataset
logistic_model_l2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", regParam=optimal_regParam_logistic, elasticNetParam=0.0)
logistic_model_l2_trained = logistic_model_l2.fit(df_transformed)

# Make predictions on test set
poisson_predictions_test = poisson_model_trained.transform(test_transformed)
logistic_predictions_l1_test = logistic_model_l1_trained.transform(test_transformed)
logistic_predictions_l2_test = logistic_model_l2_trained.transform(test_transformed)

# Evaluate models on the test set and report metrics
evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
rmse_poisson = evaluator_poisson.evaluate(poisson_predictions_test)

#evaluator_logistic = BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_l1 = evaluator_logistic.evaluate(logistic_predictions_l1_test)
auc_l2 = evaluator_logistic.evaluate(logistic_predictions_l2_test)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="hasClaim", predictionCol="prediction", metricName="accuracy")

# Calculate accuracy for Logistic Regression (L1 Regularization)
accuracy_l1 = evaluator_accuracy.evaluate(logistic_predictions_l1_test)

# Calculate accuracy for Logistic Regression (L2 Regularization)
accuracy_l2 = evaluator_accuracy.evaluate(logistic_predictions_l2_test)


# Print model coefficients
print("Poisson Regression Model Coefficients:")
print(poisson_model_trained.coefficients)

print("Logistic Regression Model Coefficients (L1 Regularization):")
print(logistic_model_l1_trained.coefficients)

print("Logistic Regression Model Coefficients (L2 Regularization):")
print(logistic_model_l2_trained.coefficients)

# Print RMSE and AUC
print("RMSE for Poisson Regression:", rmse_poisson)
print("AUC for Logistic Regression (L1 Regularization):", auc_l1)
print("AUC for Logistic Regression (L2 Regularization):", auc_l2)

# Print accuracy for Logistic Regression (L1 Regularization)
print("Accuracy for Logistic Regression (L1 Regularization):", accuracy_l1)


# Print accuracy for Logistic Regression (L2 Regularization)
print("Accuracy for Logistic Regression (L2 Regularization):", accuracy_l2)


# Stop SparkSession
spark.stop()