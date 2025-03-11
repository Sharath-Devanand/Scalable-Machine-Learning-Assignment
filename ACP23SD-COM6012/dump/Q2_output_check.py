from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.datasets import fetch_openml
from pyspark.sql.functions import when
import sys

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

outputFilePath = "Output/Q2_output.txt"
sys.stdout = open(outputFilePath, "w")

# Check the schema of the DataFrame
df.printSchema()

# Display the first few rows of the DataFrame
df.show()

# Convert ClaimNb to integer type and create hasClaim column
df = df.withColumn("ClaimNb", df["ClaimNb"].cast("int"))
df = df.withColumn("hasClaim", (df["ClaimNb"] > 0).cast("int"))

# Split dataset into training (70%) and test (30%) sets
train, test = df.randomSplit([0.7, 0.3], seed=12345)

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

# Define logistic regression models with L1 and L2 regularisation and class balancing
logistic_l1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", regParam=0.1, elasticNetParam=1, family="binomial", weightCol="classWeights")
logistic_l2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", regParam=0.1, elasticNetParam=0, family="binomial", weightCol="classWeights")

# Define pipelines for models
pipeline_logistic_l1 = Pipeline(stages=stages + [logistic_l1])
pipeline_logistic_l2 = Pipeline(stages=stages + [logistic_l2])

# Compute class weights for balancing
# Compute class weights for balancing
class_counts = train.groupBy("hasClaim").count().collect()
class_0_count = class_counts[0]["count"]
class_1_count = class_counts[1]["count"]

if class_1_count == 0:
    balance_ratio = 0  # Avoid division by zero
else:
    balance_ratio = class_0_count / class_1_count

if balance_ratio < 0:
    balance_ratio = abs(balance_ratio)  # Ensure positive balance ratio
# Assign higher weight to minority class (hasClaim == 1)
train_balanced = train.withColumn("classWeights", when(train["hasClaim"] == 1, balance_ratio).otherwise(1 - balance_ratio))

# Train logistic regression models with class balancing
model_logistic_l1 = pipeline_logistic_l1.fit(train_balanced)
model_logistic_l2 = pipeline_logistic_l2.fit(train_balanced)

# Evaluate logistic regression models
evaluator = BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="rawPrediction")
auc_l1 = evaluator.evaluate(model_logistic_l1.transform(test))
auc_l2 = evaluator.evaluate(model_logistic_l2.transform(test))
print("AUC for Logistic regression with L1 regularisation:", auc_l1)
print("AUC for Logistic regression with L2 regularisation:", auc_l2)

# Stop Spark session
spark.stop()

# Close the output file
sys.stdout.close()
