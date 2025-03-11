from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import os

# Initialize SparkSession
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load your dataset

header_names = ["label"] + ["feature{}".format(i) for i in range(1, 29)]

df = spark.read.csv("Data/HIGGS.csv", header=False, inferSchema=True).toDF(*header_names)

seed = 23788

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

# Sample 1% of the data with class balancing from the training set
train_df_sampled = train_df.sampleBy("label", fractions={0: 0.01, 1: 0.01}, seed=seed)

train_df_sampled = train_df_sampled.select("label", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16", "feature17", "feature18", "feature19", "feature20", "feature21", "feature22", "feature23", "feature24", "feature25", "feature26", "feature27", "feature28")  


# Select features and target column for the sampled training data
feature_cols = train_df_sampled.columns[1:]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df).select("label", "features")
test_df = assembler.transform(test_df).select("label", "features")

# Define evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")

####################################### MLP #######################################

mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[len(feature_cols), 10, 2])

mlp_pipeline = Pipeline(stages=[assembler, mlp])

mlp_param_grid = ParamGridBuilder() \
    .addGrid(mlp.layers, [[len(feature_cols), 10, 2],[len(feature_cols), 5, 2],[len(feature_cols), 15, 2]]) \
    .addGrid(mlp.stepSize, [0.1,0.05,0.2]) \
    .addGrid(mlp.maxIter, [10,20,30]) \
    .build()

cv_mlp = CrossValidator(estimator=mlp_pipeline,
                        estimatorParamMaps=mlp_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

cv_model_mlp = cv_mlp.fit(train_df_sampled)

def java_array_to_python_list(java_array):
    return list(java_array)


print("\nMultilayer Perceptron - Best Model:")
best_params_mlp = {
    'layers': java_array_to_python_list(cv_model_mlp.bestModel.stages[-1]._java_obj.getLayers()),
    'stepSize': cv_model_mlp.bestModel.stages[-1]._java_obj.getStepSize(),
    'maxIter': cv_model_mlp.bestModel.stages[-1]._java_obj.getMaxIter()
}

print(best_params_mlp)



####################################### GBT #######################################

gbt = GBTClassifier(labelCol="label", featuresCol="features")

gbt_pipeline = Pipeline(stages=[assembler, gbt])

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3,5,7]) \
    .addGrid(gbt.stepSize, [0.1,0.2,0.3]) \
    .addGrid(gbt.maxIter, [10,20,30]) \
    .build()

cv_gbt = CrossValidator(estimator=gbt_pipeline,
                        estimatorParamMaps=gbt_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

cv_model_gbt = cv_gbt.fit(train_df_sampled)

print("\nGradient Boosting - Best Model:")
best_params_gbt = {
    'stepSize': cv_model_gbt.bestModel.stages[-1]._java_obj.getStepSize(),
    'maxDepth': cv_model_gbt.bestModel.stages[-1]._java_obj.getMaxDepth(),
    'maxIter': cv_model_gbt.bestModel.stages[-1]._java_obj.getMaxIter()
}

print(best_params_gbt)



######################### RANDOM FOREST #########################

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

rf_pipeline = Pipeline(stages=[assembler, rf])

rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [25,50,75]) \
    .addGrid(rf.maxDepth, [3,5,7]) \
    .addGrid(rf.maxBins, [32,64,16]) \
    .build()

cv_rf = CrossValidator(estimator=rf_pipeline,
                       estimatorParamMaps=rf_param_grid,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=seed)

cv_model_rf = cv_rf.fit(train_df_sampled)

print("Random Forest - Best Model:")
best_params_rf = {
    'numTrees': cv_model_rf.bestModel.stages[-1]._java_obj.getNumTrees(),
    'maxDepth': cv_model_rf.bestModel.stages[-1]._java_obj.getMaxDepth(),
    'maxBins': cv_model_rf.bestModel.stages[-1]._java_obj.getMaxBins()
}

print(best_params_rf)

#################################################################

# Create the best models with the best hyperparameters

rf_best = RandomForestClassifier(labelCol="label", featuresCol="features").setParams(**best_params_rf)
gbt_best = GBTClassifier(labelCol="label", featuresCol="features").setParams(**best_params_gbt)
mlp_best = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features").setParams(**best_params_mlp)

# Fit models on the training set
rf_model_full = rf_best.fit(train_df)
gbt_model_full = gbt_best.fit(train_df)
mlp_model_full = mlp_best.fit(train_df)

# Evaluate the models on the test set
predictions_rf_full = rf_model_full.transform(test_df)
predictions_gbt_full = gbt_model_full.transform(test_df)
predictions_mlp_full = mlp_model_full.transform(test_df)

auc_rf_full = evaluator.evaluate(predictions_rf_full)
auc_gbt_full = evaluator.evaluate(predictions_gbt_full)
auc_mlp_full = evaluator.evaluate(predictions_mlp_full)

print("Random Forest - AUC on full dataset:", auc_rf_full)
print("Gradient Boosting - AUC on full dataset:", auc_gbt_full)
print("Multilayer Perceptron - AUC on full dataset:", auc_mlp_full)


# Stop SparkSession
spark.stop()