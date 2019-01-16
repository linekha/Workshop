from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("Titanic-api") \
      .master("local[*]") \
      .getOrCreate()
     
model = PipelineModel.load("/user/admin/models/spark")

features = ['Age', 'Fare']


def predict(args):
  wine=args["feature"].split(";")
  feature = spark.createDataFrame([map(float,wine[:11])], features)
  result=model.transform(feature).collect()[0].prediction
  if result == 1.0:
    return {"result": "N'a pas survecu"}
  else:
    return {"result" : "A survecu"}
  
# pre-heat the model
predict({"feature": "70;12"})