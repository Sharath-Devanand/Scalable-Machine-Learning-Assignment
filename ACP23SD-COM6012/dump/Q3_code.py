from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#from pyspark.sql.functions import rand

import sys
outputFilePath = "Output/Q2_output_trial.txt"
sys.stdout = open(outputFilePath, "w")

# Initialize SparkSession
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/acp22abj") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load your dataset

header_names = ["label"] + ["feature{}".format(i) for i in range(1, 29)]

df = spark.read.csv("/users/acp22abj/com6012/acp22abj-COM6012/Data/HIGGS.csv", header=False, inferSchema=True).toDF(*header_names)

seed = 23788

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

# Sample 1% of the data with class balancing from the training set
train_df_sampled = train_df.sampleBy("label", fractions={0: 0.01, 1: 0.01}, seed=seed)

# Select features and target column for the sampled training data
feature_cols = train_df_sampled.columns[1:]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df_assembled = assembler.transform(train_df_sampled)

# Split the data into features and labels for training and testing sets
train_df_assembled = train_df_assembled.select("features", "label")


# Define models
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
gbt = GBTClassifier(labelCol="label", featuresCol="features")
mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[len(feature_cols), 10, 2])

#########################################################################################################################

"""
# Define parameter grids for each model
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 3, 5]) \
    .build()

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [50, 100, 150]) \
    .addGrid(gbt.maxDepth, [3, 5, 7]) \
    .addGrid(gbt.stepSize, [0.1, 0.2, 0.3]) \
    .build()

mlp_param_grid = ParamGridBuilder() \
    .addGrid(mlp.layers, [[len(feature_cols), 10, 2], [len(feature_cols), 20, 2], [len(feature_cols), 30, 2]]) \
    .addGrid(mlp.blockSize, [128, 256, 512]) \
    .addGrid(mlp.stepSize, [0.03, 0.1, 0.3]) \
    .build()
"""

######################################################################################################################

# Define parameter grids for each model
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50]) \
    .addGrid(rf.maxDepth, [5]) \
    .addGrid(rf.minInstancesPerNode, [1]) \
    .build()

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [50]) \
    .addGrid(gbt.maxDepth, [3]) \
    .addGrid(gbt.stepSize, [0.1]) \
    .build()

mlp_param_grid = ParamGridBuilder() \
    .addGrid(mlp.layers, [[len(feature_cols), 10, 2]]) \
    .addGrid(mlp.blockSize, [128]) \
    .addGrid(mlp.stepSize, [0.03]) \
    .build()

######################################################################################################################

# Define evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")

# Perform cross-validation to find the best configuration of parameters for each model
cv_rf = CrossValidator(estimator=rf,
                       estimatorParamMaps=rf_param_grid,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=seed)

cv_gbt = CrossValidator(estimator=gbt,
                        estimatorParamMaps=gbt_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

cv_mlp = CrossValidator(estimator=mlp,
                        estimatorParamMaps=mlp_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

# Fit models
cv_model_rf = cv_rf.fit(train_df_assembled)
cv_model_gbt = cv_gbt.fit(train_df_assembled)
cv_model_mlp = cv_mlp.fit(train_df_assembled)


# Retrieve best parameters for each model
best_params_rf = cv_model_rf.bestModel.extractParamMap()
best_params_gbt = cv_model_gbt.bestModel.extractParamMap()
best_params_mlp = cv_model_mlp.bestModel.extractParamMap()

# Define models with the best parameters found for each model
best_params_rf_str = {str(k): v for k, v in best_params_rf.items() if k.name != "RandomForestClassifier_5a2fc2e65a0d__bootstrap"}
best_params_gbt_str = {str(k): v for k, v in best_params_gbt.items()}
best_params_mlp_str = {str(k): v for k, v in best_params_mlp.items()}

best_params_rf_str = {k.name: v for k, v in best_params_rf.items() if k.name in ['numTrees', 'maxDepth', 'minInstancesPerNode']}
best_params_gbt_str = {k.name: v for k, v in best_params_gbt.items() if k.name in ['maxIter', 'maxDepth', 'stepSize']}
best_params_mlp_str = {k.name: v for k, v in best_params_mlp.items() if k.name in ['layers', 'blockSize', 'stepSize']}

rf_best = RandomForestClassifier(labelCol="label", featuresCol="features").setParams(**best_params_rf_str)
gbt_best = GBTClassifier(labelCol="label", featuresCol="features").setParams(**best_params_gbt_str)
mlp_best = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features").setParams(**best_params_mlp_str)

# Fit models on the training set

rf_model_full = rf_best.fit(train_df_assembled)
gbt_model_full = gbt_best.fit(train_df_assembled)
mlp_model_full = mlp_best.fit(train_df_assembled)

# Evaluate the models on the test set
test_df_assembled = assembler.transform(test_df)
test_df_assembled = test_df_assembled.select("features", "label")

predictions_rf_full = rf_model_full.transform(test_df_assembled)
predictions_gbt_full = gbt_model_full.transform(test_df_assembled)
predictions_mlp_full = mlp_model_full.transform(test_df_assembled)

auc_rf_full = evaluator.evaluate(predictions_rf_full)
auc_gbt_full = evaluator.evaluate(predictions_gbt_full)
auc_mlp_full = evaluator.evaluate(predictions_mlp_full)

print("Random Forest - AUC on full dataset:", auc_rf_full)
print("Gradient Boosting - AUC on full dataset:", auc_gbt_full)
print("Multilayer Perceptron - AUC on full dataset:", auc_mlp_full)

# Stop SparkSession
spark.stop()

# Close the output file
sys.stdout.close()