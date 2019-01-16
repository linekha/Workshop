#Initialisation d'une session Spark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F 

spark= SparkSession.builder.enableHiveSupport().getOrCreate()

#Lecture d'une table Hive depuis Spark
df = spark.table("default.checkouts20")
#Lecture d'un fichier sur hdfs depuis Spark
df_hdfs= spark.read.csv("/user/admin/CldAnalytics/checkouts.csv")
#Lecture d'un fichier sur adls depuis Spark
df_adls=spark.read.csv("")


#Visualisation schema table 
df.printSchema()

#Browse hdfs depuis cdsw
!hdfs dfs -ls /user/admin/written
  
#Browse adls depuis cdsw
!hdfs dfs -ls adl://adlfsalm.azuredatalakestore.net/
  
#Conversion en view 
df.createOrReplaceTempView("df")


#Exemple d'operations
df.count()
spark.sql("select coun(*) from df")


#Conversion en pandas et manip pandas
df.toPandas()
df.shape()
df.iloc[1:]



#Ecriture d'un fichier parquet sur hdfs
spark.write.parquet("/user/admin/written/")