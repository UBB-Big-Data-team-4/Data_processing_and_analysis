import sys
import shutil
import json
from pathlib import Path
from ultralytics import YOLO
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os
import base64
import uuid
from io import BytesIO
from PIL import Image
from MongoDBImporter import MongoDBImporter

class ModelFactory:
    """
    Factory class to manage YOLO classification pipeline with MongoDB and PySpark.
    """

    def __init__(self, model_size="n", spark_master="local[*]", mongo_batch_size=1000):
        """
        Initialize ModelFactory.

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.spark_master = spark_master
        self.mongo_batch_size = mongo_batch_size
        self.mongo = MongoDBImporter()
        self.spark = self._initialize_spark()
        self.model = None
        self.dataset_path = Path.cwd() / "yolo_dataset"
        self.mongo_temp_dir = Path.cwd() / "mongo_staging"

    def _initialize_spark(self):
        """Initialize PySpark session with adequate memory."""
        return SparkSession.builder \
            .master(self.spark_master) \
            .appName("YOLOModelFactory") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    def load_data_from_mongodb(self, collection_names):
        print(f"Staging data from MongoDB: {collection_names}", flush=True)
        self.mongo_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = self.mongo_temp_dir / "export.jsonl"

        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("people", IntegerType(), True),
            StructField("img", StringType(), True),
        ])

        with open(temp_file_path, "w") as f:
            for coll_name in collection_names:
                try:
                    collection = self.mongo.db[coll_name]
                    cursor = collection.find({}, {"_id": 0, "timestamp": 1, "people": 1, "img": 1})

                    for doc in cursor:
                        record = {
                            "timestamp": str(doc.get("timestamp", "")),
                            "people": int(doc.get("people", 0)),
                            "img": doc.get("img", "")
                        }
                        f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"Error staging {coll_name}: {e}", file=sys.stderr)

        return self.spark.read.schema(schema).json(str(temp_file_path))

    def prepare_training_data(self, collection_names):
        """
        Process images in Spark and write to disk in YOLO classification format.
        """
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        df_spark = self.load_data_from_mongodb(collection_names)
        df_spark = self._balance_dataset(df_spark)
        train_df, val_df = df_spark.randomSplit([0.8, 0.2], seed=42)

        def save_partition(split_name, base_dir):
            def _process(partition):

                for row in partition:
                    try:
                        label = min(int(row.people), 99)
                        if not row.img: continue

                        img_data = base64.b64decode(row.img.encode('utf-8'))
                        img = Image.open(BytesIO(img_data)).convert('RGB')

                        class_dir = os.path.join(base_dir, split_name, str(label))
                        os.makedirs(class_dir, exist_ok=True)
                        img.save(os.path.join(class_dir, f"{uuid.uuid4().hex}.jpg"), "JPEG")
                    except:
                        continue

            return _process

        print("Writing classification dataset to disk...", flush=True)
        train_df.rdd.foreachPartition(save_partition("train", str(self.dataset_path)))
        val_df.rdd.foreachPartition(save_partition("val", str(self.dataset_path)))

        return str(self.dataset_path)

    def load_model(self):
        """
        Load a YOLO classification model (suffix '-cls').
        """
        model_name = f"yolo11{self.model_size}-cls.pt"
        print(f"Loading Classification Model: {model_name}", flush=True)
        self.model = YOLO(model_name)
        return self.model

    def fine_tune(self, collection_names, epochs=50, imgsz=224, batch_size=16, device='mps'):
        """
        Fine-tune with the classification task.
        """
        if self.model is None:
            self.load_model()

        dataset_path_str = self.prepare_training_data(collection_names)

        return self.model.train(
            data=dataset_path_str,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            task='classify',
            batch=batch_size,
        )

    def cleanup_training_data(self):
        for p in [self.dataset_path, self.mongo_temp_dir]:
            if p.exists():
                shutil.rmtree(p)

    def close(self):
        if self.spark:
            self.spark.stop()
        self.cleanup_training_data()

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        self.model.save(path)

    def _balance_dataset(self, df, max_ratio=3.0):
        """
        Downsamples majority classes, allowing them to be at most
        `max_ratio` times larger than the smallest class.
        """
        print("Calculating class distribution for balancing...", flush=True)

        class_counts = df.groupBy("people").count().collect()
        if not class_counts:
            return df

        counts_dict = {row['people']: row['count'] for row in class_counts}
        print(f"Original dataset distribution: {counts_dict}", flush=True)

        min_count = min(counts_dict.values())
        max_allowed = int(min_count * max_ratio)
        print(f"Smallest class has {min_count} images. Capping other classes at {max_allowed} images.", flush=True)

        fractions = {}
        for label, count in counts_dict.items():
            if count <= max_allowed:
                fractions[label] = 1.0
            else:
                fractions[label] = float(max_allowed) / count

        balanced_df = df.stat.sampleBy("people", fractions, seed=42)

        return balanced_df