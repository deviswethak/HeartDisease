import numpy as np # linear algebra 
import pandas as pd # data processing 
import os 
print(os.listdir("../input")) 
#creating a spark session 
from pyspark.sql import SparkSession 
import pyspark.sql as sparksql 
spark = SparkSession.builder.appName('stroke').getOrCreate() 
train = spark.read.csv('../input/train_2v.csv', inferSchema=True,header=True) 
test = spark.read.csv('../input/test_2v.csv', inferSchema=True,header=True) 
#Exploring the training data 
train.printSchema() 
train.dtypes 
train.head(5) 
train.toPandas().head(5) 
#Lets also look at test data 
test.describe().show() 
#Lets look the the target distribution 
train.groupBy('stroke').count().show() 
#Training feature analysis 
train.createOrReplaceTempView('table') 
# sql query to find the number of people in specific work_type who have had stroke and not 
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE 
stroke == 1 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show() 
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE 
stroke == 0 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show() 
#Is it related to gender !!!
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT 
COUNT(gender) FROM table WHERE gender == 'Male') as percentage FROM table WHERE 
stroke== 1 AND gender = 'Male' GROUP BY gender").show() 
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT 
COUNT(gender) FROM table WHERE gender == 'Female') as percentage FROM table WHERE 
stroke== 1 AND gender = 'Female' GROUP BY gender").show() 
spark.sql("SELECT COUNT(age)*100/(SELECT COUNT(age) FROM table WHERE stroke 
==1) as percentage FROM table WHERE stroke == 1 AND age>=50").show() 
#Cleaning up training data 
train.describe().show() 
# fill in missing values for smoking status 
# As this is categorical data, we will add one data type "No Info" for the missing one 
train_f = train.na.fill('No Info', subset=['smoking_status']) 
test_f = test.na.fill('No Info', subset=['smoking_status']) 
# fill in miss values for bmi 
# as this is numecial data , we will simple fill the missing values with mean 
from pyspark.sql.functions import mean 
mean = train_f.select(mean(train_f['bmi'])).collect() 
mean_bmi = mean[0][0] 
train_f = train_f.na.fill(mean_bmi,['bmi']) 
test_f = test_f.na.fill(mean_bmi,['bmi']) 
train_f.describe().show() 
test_f.describe().show() 
#Now there is no missing values, Lets work on categorical columns now...
#StringIndexer -> OneHotEncoder -> VectorAssemble
# indexing all categorical columns in the dataset 
from pyspark.ml.feature import StringIndexer 
indexer1 = StringIndexer(inputCol="gender", outputCol="genderIndex") 
indexer2 = StringIndexer(inputCol="ever_married", outputCol="ever_marriedIndex") 
indexer3 = StringIndexer(inputCol="work_type", outputCol="work_typeIndex") 
indexer4 = StringIndexer(inputCol="Residence_type", outputCol="Residence_typeIndex") 
indexer5 = StringIndexer(inputCol="smoking_status", outputCol="smoking_statusIndex") 
# Doing one hot encoding of indexed data 
from pyspark.ml.feature import OneHotEncoderEstimator 
encoder = 
OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Re
sidence_typeIndex","smoking_statusIndex"], 
 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_ty
peVec","smoking_statusVec"]) 
#The next step is to create an assembler, that combines a given list of columns into a single 
vector column to train ML model. I will use the vector columns, that we got after 
one_hot_encoding.
from pyspark.ml.feature import VectorAssembler 
assembler = VectorAssembler(inputCols=['genderVec', 
 'age', 
 'hypertension', 
 'heart_disease', 
 'ever_marriedVec', 
 'work_typeVec', 
 'Residence_typeVec', 
 'avg_glucose_level', 
 'bmi', 
 'smoking_statusVec'],outputCol='features') 
#creating a model 
from pyspark.ml.classification import DecisionTreeClassifier 
dtc = DecisionTreeClassifier(labelCol='stroke',featuresCol='features') 
from pyspark.ml import Pipeline 
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, 
dtc]) 
# splitting training and validation data 
train_data,val_data = train_f.randomSplit([0.7,0.3]) 
# training model pipeline with data 
model = pipeline.fit(train_data) 
# making prediction on model with validation data 
dtc_predictions = model.transform(val_data) 
# Select example rows to display. 
dtc_predictions.select("prediction","probability", "stroke", "features").show(5) 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
# Select (prediction, true label) and compute test error 
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", 
metricName="accuracy") 
dtc_acc = acc_evaluator.evaluate(dtc_predictions) 
print('A Decision Tree algorithm had an accuracy of: {0:2.2f}%'.format(dtc_acc*100)) 
# now predicting the labels for test data 
test_pred = model.transform(test_f) 
test_selected = test_pred.select("id", "features", "prediction","probability") 
test_selected.limit(5).toPandas()
