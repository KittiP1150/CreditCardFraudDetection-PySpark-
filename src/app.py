import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np

load_dotenv()
data_path = os.getenv("DATA_PATH")

spark = SparkSession.builder \
    .appName("CreditCardFraudDetection") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
df = spark.read.csv(data_path, header=True, inferSchema=True)

#----check------
#print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
# missing = df.select([
#     count(when(col(c).isNull() | isnan(c), c)).alias(c)
#     for c in df.columns
# ])
# missing.show()
#--------------

df.groupBy("Class").count() \
  .withColumn("Percentage", F.round(F.col("count") / df.count() * 100, 4)) \
  .show()

df.describe(["Amount", "Time"]).show()

feature_cols = [c for c in df.columns if c != "Class"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)
 
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

pipeline_prep = Pipeline(stages=[assembler, scaler])
prep_model    = pipeline_prep.fit(df)
df_scaled     = prep_model.transform(df)
 
df_scaled.select("features", "Class").show(3, truncate=True)

evaluator = ClusteringEvaluator(
    featuresCol="features",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

best_k, best_score = 2, -1
for k in range(2, 9):
    km    = KMeans(featuresCol="features", k=k, seed=42, maxIter=30)
    model = km.fit(df_scaled)
    pred  = model.transform(df_scaled)
    score = evaluator.evaluate(pred)
    print(f"  k={k}  silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k = score, k
 
print(f"\nBest k = {best_k}  (silhouette={best_score:.4f})")

kmeans = KMeans(
    featuresCol="features",
    k=best_k,
    seed=42,
    maxIter=50,
    distanceMeasure="euclidean"
)
 
km_model   = kmeans.fit(df_scaled)
df_pred    = km_model.transform(df_scaled)  
 
print("\nCluster sizes:")
df_pred.groupBy("prediction").count().orderBy("prediction").show()
 
print("Fraud rate per cluster:")
df_pred.groupBy("prediction") \
    .agg(
        F.count("*").alias("total"),
        F.sum("Class").alias("fraud_count"),
        F.round(F.mean("Class") * 100, 4).alias("fraud_pct")
    ) \
    .orderBy("fraud_pct", ascending=False) \
    .show()
    
centroids = km_model.clusterCenters()          # list of numpy arrays
centroid_bc = spark.sparkContext.broadcast(centroids)
 
def dist_to_centroid(features_vec, cluster_id):
    """Euclidean distance ระหว่าง point กับ centroid ของ cluster ตัวเอง"""
    center = centroid_bc.value[cluster_id]
    diff   = np.array(features_vec.toArray()) - center
    return float(np.sqrt(np.dot(diff, diff)))
 
dist_udf = F.udf(dist_to_centroid, "double")
 
df_with_dist = df_pred.withColumn(
    "dist_to_centroid",
    dist_udf(F.col("features"), F.col("prediction"))
)
 
df_with_dist.select(
    "prediction", "dist_to_centroid", "Class"
).orderBy("dist_to_centroid", ascending=False).show(10)

#------check fraud from outlier------

stats = df_with_dist.groupBy("prediction").agg(
    F.mean("dist_to_centroid").alias("mean_dist"),
    F.stddev("dist_to_centroid").alias("std_dist")
)
 
df_final = df_with_dist.join(stats, on="prediction", how="left")
 
df_final = df_final.withColumn(
    "threshold",
    F.col("mean_dist") + 2.0 * F.col("std_dist")
).withColumn(
    "predicted_fraud",
    (F.col("dist_to_centroid") > F.col("threshold")).cast("double")
)
 
print("\n--- Fraud Prediction (distance-based) ---")
df_final.groupBy("predicted_fraud", "Class").count().orderBy(
    "predicted_fraud", "Class"
).show()

#------check fraud from cluster------

fraud_rate = df_with_dist.groupBy("prediction").agg(
    F.mean("Class").alias("fraud_rate_in_cluster")
)
 
FRAUD_CLUSTER_THRESHOLD = 0.05   #threshold
 
fraud_clusters = fraud_rate.filter(
    F.col("fraud_rate_in_cluster") >= FRAUD_CLUSTER_THRESHOLD
).select("prediction")
 
fraud_cluster_list = [r.prediction for r in fraud_clusters.collect()]
print(f"\nFraud-dominant clusters (fraud_rate >= {FRAUD_CLUSTER_THRESHOLD}):")
print(fraud_cluster_list)
 
df_final2 = df_with_dist.withColumn(
    "predicted_fraud_cluster",
    F.col("prediction").isin(fraud_cluster_list).cast("double")
)
 
print("--- Fraud Prediction (cluster-based) ---")
df_final2.groupBy("predicted_fraud_cluster", "Class").count().orderBy(
    "predicted_fraud_cluster", "Class"
).show()

#------evaluate-------

def evaluate(df_eval, pred_col, label_col="Class", name=""):
    print(f"\n=== {name} ===")
    mc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=pred_col
    )
    for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
        val = mc.evaluate(df_eval, {mc.metricName: metric})
        print(f"  {metric:20s}: {val:.4f}")
        
evaluate(df_final,  "predicted_fraud",         name="Distance-based threshold")
evaluate(df_final2, "predicted_fraud_cluster",  name="Cluster-based label")

#combine model
df_final = df_final.withColumn(
    "predicted_fraud_final",
    (
        (F.col("dist_to_centroid") > F.col("threshold")) |
        (F.col("prediction").isin(fraud_cluster_list))
    ).cast("double")
)

w = Window.partitionBy()

df_final = df_final.withColumn(
    "max_dist", F.max("dist_to_centroid").over(w)
).withColumn(
    "risk_score",
    F.round((F.col("dist_to_centroid") / F.col("max_dist")) * 100, 2)
)
df_final = df_final.withColumn(
    "reason",
    F.when(F.col("dist_to_centroid") > F.col("threshold"),
           "Outlier (far from centroid)") \
     .when(F.col("prediction").isin(fraud_cluster_list),
           "High fraud cluster") \
     .otherwise("Normal")
)

df_final.filter(F.col("predicted_fraud_final") == 1) \
    .orderBy(F.col("risk_score").desc()) \
    .select(
        "Time",
        "Amount",
        "risk_score",
        "reason"
    ).show(20, truncate=False)

evaluate(df_final, "predicted_fraud_final", name="Final Combined Model")
# ------result------
output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

df_final.select(
    "Time", "Amount", "Class",
    "prediction", "dist_to_centroid",
    "threshold", "risk_score", "reason",
    "predicted_fraud_final"
).write.mode("overwrite").parquet(f"{output_dir}/fraud_clustering_results.parquet")
 
print("\nSaved: fraud_clustering_results.parquet")
 
# Save model
km_model.write().overwrite().save(f"{output_dir}/kmeans_fraud_model")
print("Saved: kmeans_fraud_model/")
 
spark.stop()
print("\nDone!")