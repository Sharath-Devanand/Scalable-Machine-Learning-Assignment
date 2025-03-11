import sys
outputFilePath = "Output/Q2_output_trial.txt"
sys.stdout = open(outputFilePath, "w")

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from sklearn.datasets import fetch_openml
from pyspark.sql.functions import when
from pyspark.sql.functions import rand

# Create Spark session
spark = SparkSession.builder \
    .appName("LiabilityClaimPrediction") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

# Set log level
spark.sparkContext.setLogLevel("WARN")

# Load data directly into Spark DataFrame
data_path = fetch_openml(data_id=41214, as_frame=True).data
df = spark.createDataFrame(data_path)

# Check the schema of the DataFrame
df.printSchema()

# Display the first few rows of the DataFrame
df.show()

# Convert ClaimNb to integer type and create hasClaim column
df = df.withColumn("ClaimNb", df["ClaimNb"].cast("int"))
df = df.withColumn("hasClaim", (df["ClaimNb"] > 0).cast("int"))

# Split dataset into training (70%) and test (30%) sets
train, test = df.randomSplit([0.7, 0.3], seed=12345)

# Sample a small subset from the training set for cross-validation
train_subset = train.sample(False, 0.1, seed=12345)

# Define feature columns
feature_columns = ["Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]

# Define stages for preprocessing pipeline
stages = []

# Standardize numeric features
numeric_columns = ["Exposure", "VehAge", "DrivAge", "BonusMalus", "Density"]
for col in numeric_columns:
    assembler = VectorAssembler(inputCols=[col], outputCol=col + "_vec")
    scaler = StandardScaler(inputCol=col + "_vec", outputCol=col + "_scaled")
    stages += [assembler, scaler]

# One-hot encode categorical features
categorical_columns = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]
for col in categorical_columns:
    indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
    encoder = OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=col + "_encoded")
    stages += [indexer, encoder]

# Assemble feature columns
assembler = VectorAssembler(inputCols=[col + "_scaled" for col in numeric_columns] + [col + "_encoded" for col in categorical_columns], outputCol="features")
stages += [assembler]

# Define Poisson regression model
poisson_reg = GeneralizedLinearRegression(labelCol="ClaimNb", featuresCol="features", family="poisson")

# Define Logistic regression models with L1 and L2 regularization
logistic_l1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", elasticNetParam=1)
logistic_l2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", elasticNetParam=0)

# Define pipelines for models
pipeline_poisson = Pipeline(stages=stages + [poisson_reg])
pipeline_logistic_l1 = Pipeline(stages=stages + [logistic_l1])
pipeline_logistic_l2 = Pipeline(stages=stages + [logistic_l2])

# Set up the parameter grid
paramGrid_poisson = ParamGridBuilder() \
    .addGrid(poisson_reg.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

paramGrid_logistic = ParamGridBuilder() \
    .addGrid(logistic_l1.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .addGrid(logistic_l2.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

# Set up cross-validation
crossval_poisson = CrossValidator(estimator=pipeline_poisson,
                                   estimatorParamMaps=paramGrid_poisson,
                                   evaluator=RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse"),
                                   numFolds=3)

crossval_logistic = CrossValidator(estimator=pipeline_logistic_l1,
                                   estimatorParamMaps=paramGrid_logistic,
                                   evaluator=BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="rawPrediction"),
                                   numFolds=3)

# Run cross-validation, and choose the best set of parameters
cvModel_poisson = crossval_poisson.fit(train_subset)
cvModel_logistic = crossval_logistic.fit(train_subset)

# Retrieve the best models
best_model_poisson = cvModel_poisson.bestModel
best_model_logistic = cvModel_logistic.bestModel

# Print the best values of regParam for Poisson regression
print("Best regParam for Poisson regression:", best_model_poisson.stages[-1].getRegParam())

# Print the best values of regParam for Logistic regression with L1 and L2 regularization
print("Best regParam for Logistic regression with L1 regularization:", best_model_logistic.stages[-1].getRegParam())
print("Best regParam for Logistic regression with L2 regularization:", best_model_logistic.stages[-2].getRegParam())

# Train the best models on the full dataset
model_poisson = best_model_poisson.fit(df)
model_logistic = best_model_logistic.fit(df)

# Evaluate Poisson regression model on the test set
predictions_poisson = model_poisson.transform(test)
evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
rmse_poisson = evaluator_poisson.evaluate(predictions_poisson)
print("RMSE for Poisson regression on the test set:", rmse_poisson)

# Evaluate Logistic regression model on the test set
predictions_logistic = model_logistic.transform(test)
evaluator_logistic = BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="rawPrediction", metricName="accuracy")
accuracy_logistic = evaluator_logistic.evaluate(predictions_logistic)
print("Accuracy for Logistic regression on the test set:", accuracy_logistic)

# Stop Spark session
spark.stop()

# Close the output file
sys.stdout.close()