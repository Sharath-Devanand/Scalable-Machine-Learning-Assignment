import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from sklearn.datasets import fetch_openml
from pyspark.ml.feature import StringIndexer
import numpy as np
import json


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q2 Assignment") \
        .config("spark.local.dir","/mnt/parscratch/users/acp23ty") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
spark.sparkContext.setLogLevel("ERROR")
seed=18693

df_freq = fetch_openml(data_id=41214, as_frame=True).data

# Convert the Pandas DataFrame to a PySpark DataFrame
df_freq_spark = spark.createDataFrame(df_freq)
df_freq_spark.show(10)

# Create a new column 'hasClaim' to indicate presence or absence of a claim
df_freq_spark = df_freq_spark.withColumn("hasClaim", when(col("ClaimNb") > 0, 1).otherwise(0))
df_freq_spark.groupBy("hasClaim").count().show()


total_count = df_freq_spark.count()

# Calculate the percentages of class 0 and class 1
class_0_percentage = df_freq_spark.filter(col("hasClaim") == 0).count() / total_count
print(class_0_percentage)
class_1_percentage = df_freq_spark.filter(col("hasClaim") == 1).count() / total_count
print(class_1_percentage)

# Define the split ratio for the training set
train_ratio = 0.7

# Calculate the split ratios for each class
split_ratio_0_train = train_ratio * class_0_percentage
print("split ration 0", split_ratio_0_train)
split_ratio_1_train = train_ratio * class_1_percentage
print("split ration 1", split_ratio_1_train)

# Sample the data for the training set using stratified sampling
train_data_0 = df_freq_spark.filter(col("hasClaim") == 0).sample(False, split_ratio_0_train, seed=seed)
train_data_1 = df_freq_spark.filter(col("hasClaim") == 1).sample(False, split_ratio_1_train, seed=seed)

# Concatenate the training sets for both classes
train_df = train_data_0.union(train_data_1)
train_df.groupBy("hasClaim").count().show()

# Calculate the split ratios for each class in the test set
split_ratio_0_test = (1 - train_ratio) * class_0_percentage
split_ratio_1_test = (1 - train_ratio) * class_1_percentage

# Sample the data for the test set using stratified sampling
test_data_0 = df_freq_spark.filter(col("hasClaim") == 0).sample(False, split_ratio_0_test, seed=seed)
test_data_1 = df_freq_spark.filter(col("hasClaim") == 1).sample(False, split_ratio_1_test, seed=seed)

# Concatenate the test sets for both classes
test_df = test_data_0.union(test_data_1)
test_df.groupBy("hasClaim").count().show()

# Verify the class distribution in the training set
#print("Training Data non-zero claims percentage: %g" % (trainingData.filter(trainingData.hasClaim == 1).count() / trainingData.count()))
#print("Training Data zero claims percentage: %g" % (trainingData.filter(trainingData.hasClaim == 0).count() / trainingData.count()))
##print("Training Data count for class 0 (hasClaim = 0): %d" % trainingData.filter(trainingData.hasClaim == 0).count())


# Verify the class distribution in the test set
#print("\nTest Data non-zero claims percentage: %g" % (testData.filter(testData.hasClaim == 1).count() / testData.count()))
#print("Test Data zero claims percentage: %g" % (testData.filter(testData.hasClaim == 0).count() / testData.count()))
#print("Test Data count for class 1 (hasClaim = 1): %d" % testData.filter(testData.hasClaim == 1).count())
#print("Test Data count for class 0 (hasClaim = 0): %d" % testData.filter(testData.hasClaim == 0).count()

#df_freq_spark.printSchema()
# Define numeric features
numeric_features = ["VehPower","Exposure",  "VehAge", "DrivAge", "BonusMalus", "Density"]
categorical_features = ["Area", "VehBrand", "VehGas", "Region"]




# Define stages for the pipeline
stages = []

# Assemble numeric features into a single vector
assembler_num = VectorAssembler(inputCols=numeric_features, outputCol="num_features")
stages += [assembler_num]

# Standardize numeric features
scaler = StandardScaler(inputCol="num_features", outputCol="scaled_features", withMean=True, withStd=True)
stages += [scaler]

# One-hot encoding

# Index string categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col+"_encoded", handleInvalid="keep") for col in categorical_features]
stages += indexers

# Assemble all features into a single vector
input_cols = ["scaled_features"] + [categorical_col + "_encoded" for categorical_col in categorical_features]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
stages += [assembler]

# Create the pipeline
pipeline = Pipeline(stages=stages)

sampled_train_df = train_df.sample(False, 0.1, seed=seed)
sampled_train_df.printSchema()
sampled_train_df.groupBy("hasClaim").count().show()

#print("train df schema")
#train_df.printSchema()
# Apply transformations
pipelineModel = pipeline.fit(sampled_train_df)
sample_train_transformed = pipelineModel.transform(sampled_train_df)
sample_train_transformed.printSchema()


#Sample a small subset from the training se
#sampled_train_df = train_df_transformed.sample(withReplacement=False, fraction=0.1, seed= 18683)
#print("sample train df schema")
#sampled_train_df.printSchema()



# Define Poisson regression model
poisson = GeneralizedLinearRegression(family="poisson", labelCol="ClaimNb", featuresCol="features")
paramGrid_poisson = ParamGridBuilder().addGrid(poisson.regParam, [0.001, 0.01, 0.1, 1,10]).build()

# Perform cross-validation
cv_poisson = CrossValidator(estimator= poisson, estimatorParamMaps= paramGrid_poisson, evaluator=RegressionEvaluator(labelCol='ClaimNb', predictionCol='prediction', metricName='rmse'), numFolds=5)




# Define Logistic regression models with L1 and L2 regularisation
logistic_l1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", maxIter=10, regParam=0.01, elasticNetParam=1)
paramGrid_logistic_l1 = ParamGridBuilder().addGrid(logistic_l1.regParam, [0.001, 0.01, 0.1, 1, 10]).build()
evaluator_logistic=BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="prediction", metricName="areaUnderROC")
cv_logistic_l1 = CrossValidator(estimator= logistic_l1, estimatorParamMaps= paramGrid_logistic_l1, evaluator=evaluator_logistic, numFolds=5) 



logistic_l2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", maxIter=10, regParam=0.01, elasticNetParam=0)
paramGrid_logistic_l2 = ParamGridBuilder().addGrid(logistic_l2.regParam, [0.001, 0.01, 0.1, 1,10]).build()
cv_logistic_l2 = CrossValidator(estimator= logistic_l2, estimatorParamMaps= paramGrid_logistic_l2, evaluator=evaluator_logistic, numFolds=5) 


# Fit cross-validators to the sampled subset
model_poisson = cv_poisson.fit(sample_train_transformed)
model_logistic_l1 = cv_logistic_l1.fit(sample_train_transformed)
model_logistic_l2 = cv_logistic_l2.fit(sample_train_transformed)


# Best Poisson regParam
best_poisson_regParam = model_poisson.bestModel._java_obj.getRegParam()
#best_glr_regParam = cvModel_glr.bestModel._java_obj.getRegParam()

# Best Logistic (L1) regParam
best_logistic_l1_regParam = model_logistic_l1.bestModel.getRegParam()

# Best Logistic (L2) regParam
best_logistic_l2_regParam = model_logistic_l2.bestModel.getRegParam()

# Print best regParams
print("Best Poisson regParam:", best_poisson_regParam)
print("Best Logistic (L1) regParam:", best_logistic_l1_regParam)
print("Best Logistic (L2) regParam:", best_logistic_l2_regParam)
#===============================================================

# Re-define models with optimal hyperparameters
poisson = GeneralizedLinearRegression(family="poisson", labelCol="ClaimNb", featuresCol="features", regParam=best_poisson_regParam)
logistic_l1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", maxIter=10, regParam=best_logistic_l1_regParam, elasticNetParam=1)
logistic_l2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", maxIter=10, regParam=best_logistic_l2_regParam, elasticNetParam=0)

# Train models on the full training data
pipelineModel_poisson = pipeline.fit(train_df)
train_df_transformed_poisson = pipelineModel_poisson.transform(train_df)
model_poisson = poisson.fit(train_df_transformed_poisson)

pipelineModel_logistic = pipeline.fit(train_df)
train_df_transformed_l1 = pipelineModel_logistic.transform(train_df)
model_logistic_l1 = logistic_l1.fit(train_df_transformed_l1)

#pipelineModel_logistic_l2 = pipeline.fit(train_df)
train_df_transformed_l2 = pipelineModel_logistic.transform(train_df)
model_logistic_l2 = logistic_l2.fit(train_df_transformed_l2)



test_df_transformed = pipelineModel_poisson.transform(test_df)
test_df_transformed_l1 = pipelineModel_logistic.transform(test_df)

predictions_pois = model_poisson.transform(test_df_transformed)
predictions_l1 = model_logistic_l1.transform(test_df_transformed_l1)
predictions_l2 = model_logistic_l1.transform(test_df_transformed_l1)


# Evaluate models on the test set
#evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")

rmse_poisson = RegressionEvaluator(labelCol="ClaimNb").evaluate(predictions_pois,{RegressionEvaluator.metricName: "rmse"})

accuracy_logistic_l1 = MulticlassClassificationEvaluator(labelCol="hasClaim", metricName="accuracy").evaluate(predictions_l1)

#evaluator_logistic = BinaryClassificationEvaluator(labelCol="hasClaim", rawPredictionCol="prediction", metricName="areaUnderROC")
#evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="hasClaim", predictionCol="prediction", metricName="accuracy")

accuracy_logistic_l2 = MulticlassClassificationEvaluator(labelCol="hasClaim", metricName="accuracy").evaluate(predictions_l2)


#rmse_poisson = evaluator_poisson.evaluate(model_poisson.transform(test_df_transformed))
#auc_logistic_l1 = evaluator_logistic.evaluate(model_logistic_l1.transform(test_df_transformed_l1))
#accuracy_logistic_l1 = evaluator_accuracy.evaluate(model_logistic_l1.transform(test_df_transformed_l1))

#auc_logistic_l2 = evaluator_logistic.evaluate(model_logistic_l2.transform(test_df_transformed_l1))
#accuracy_logistic_l2 = evaluator_accuracy.evaluate(model_logistic_l2.transform(test_df_transformed_l1))

# Print evaluation metrics
print("Poisson Regression (RMSE):", rmse_poisson)
#print("Logistic Regression (L1) - AUC:", auc_logistic_l1)
#print("Logistic Regression (L2) - AUC:", auc_logistic_l2)
print()

# Get accuracy from the evaluator for L1 and L2

print("Logistic Regression (L1) - Accuracy:", accuracy_logistic_l1)
print("Logistic Regression (L2) - Accuracy:", accuracy_logistic_l2)


print()
# Print model coefficients (if applicable)
print("===============Poisson Regression Coefficients:")
print()
print(model_poisson.coefficients)  # Might require additional configuration depending on Spark version

print("===============Logistic Regression (L1) Coefficients:")
print()
print(model_logistic_l1.coefficients)  # Might require additional configuration depending on Spark version

print("===============Logistic Regression (L2) Coefficients:")
print()
print(model_logistic_l2.coefficients)  # Might require additional configuration depending on Spark version



# Stop the SparkSession
spark.stop()
