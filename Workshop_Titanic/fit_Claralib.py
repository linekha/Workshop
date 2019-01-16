import cdsw
import claraml
from claraml import FP_forests
import pyspark.sql.functions as F
from pyspark import SparkContext,SparkConf,SQLContext

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

train = sqlContext.read.csv("/user/admin/titanic/train.csv",
                                    header="true",
                                    sep=",",
                                    inferSchema="true")

train = train.dropna()

fp=FP_forests(train)

# Remove unvalid data
train=train.fillna("Q",subset="Embarked")


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


indexed_data.select('Survived','Sex','Embarked','Cabin','SibSp','Survived_Index','Sex_Index','Sex_Vec','Embarked_Index','Embarked_Vec','features').show(1,truncate=False)

indexed_data=fp.indexers(train, "Survived", ['Sex', 'Embarked', 'Cabin', 'SibSp'], ["Age", "Fare"])
fp.create_paramgrid([10, 20], [3, 2])
fp.create_evaluator("accuracy", "prediction")


fp.get_model([RandomForestClassifier])
fp.get_predictions([MulticlassClassificationEvaluator])
fp.choose_best_model()


best_model= fp.choose_best_model()
get_df=best_model[0]
get_df.select('*').show(2,truncate=False)

