import cdsw
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler



conf = SparkConf().setAppName("wine-quality-build-model")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# # Get params
param_numTrees=10
param_maxDepth=10
param_impurity="gini"

cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)


train = sqlContext.read.csv("/user/admin/titanic/train.csv",
                                    header="true",
                                    sep=",",
                                    inferSchema="true")

train = train.dropna()


featureIndexer = VectorAssembler(
    inputCols = ['Age', 'Fare'],
    outputCol = 'features')

labelIndexer = StringIndexer(inputCol = 'Survived', outputCol = 'label')

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier


(trainingData, testData) = train.randomSplit([0.7, 0.3])
 
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', 
                                    numTrees = param_numTrees, 
                                    maxDepth = param_maxDepth,  
                                    impurity = param_impurity)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, classifier])
model = pipeline.fit(trainingData)

predictions = model.transform(testData)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)


cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)

model.write().overwrite().save("models/spark")

#!rm -r -f models/spark
#!rm -r -f models/spark_rf.tar
!mkdir models
!hdfs dfs -get ./models/spark models/
!tar -cvf models/spark_rf.tar models/spark

cdsw.track_file("models/spark_rf.tar")

sc.stop()


