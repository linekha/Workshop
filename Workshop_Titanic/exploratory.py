
import claraprep
from claraprep import dataprep 
import pyspark.sql.functions as F
from pyspark import SparkContext,SparkConf,SQLContext

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

!hdfs dfs -ls 
!hdfs dfs -mkdir -p /user/admin/titanic
!hdfs dfs -put /home/cdsw/train.csv /user/admin/titanic

data = sqlContext.read.csv("/user/admin/titanic/train.csv",
                           header="true",
                           sep=",",
                           inferSchema="true")

data.select('Embarked').distinct().show()

data = data.dropna()
data=data.fillna("Q",subset="Embarked")

#data temptable for sql
data.createOrReplaceTempView('data_sql')
sqlContext.sql('select * from data_sql')

#Ou encore en api python-sql
data.groupBy('Embarked').agg(F.count('Embarked')).show()



#ClaraLib functions

dp=dataprep.dataprep(data)

data_cleaned = dp.cleanup(data,('double','integer'))

data_pd = data.toPandas()
dp.vars_density(data_pd,['Age','Fare','SibSp','Parch'],'green')
dp.data_repartition(data_pd)
dp.data_crossed_repartition(data_pd,['Sex','Embarked','Cabin','SibSp','Parch','Pclass'],'Survived')
dp.Pearson_heatmap(data_pd,1,True)




