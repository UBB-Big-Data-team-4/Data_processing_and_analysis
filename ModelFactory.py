import sys
import shutil
import json
from pathlib import Path
from ultralytics import YOLO
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import os
import base64
import uuid
from io import BytesIO
from PIL import Image
from MongoDBImporter import MongoDBImporter


class ModelFactory:
    """
    Factory class to manage YOLO detection pipeline with MongoDB and PySpark.
    """

    def __init__(self, model_size="n", spark_master="local[*]", mongo_batch_size=1000):
        self.model_size = model_size
        self.spark_master = spark_master
        self.mongo_batch_size = mongo_batch_size
        self.mongo = MongoDBImporter()
        self.spark = self._initialize_spark()
        self.model = None
        self.dataset_path = Path.cwd() / "yolo_dataset"
        self.mongo_temp_dir = Path.cwd() / "mongo_staging"
        self.data_yaml_path = self.dataset_path / "data.yaml"

    def _initialize_spark(self):
        return SparkSession.builder \
            .master(self.spark_master) \
            .appName("YOLOModelFactory") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    def load_data_from_mongodb(self, collection_names, limit=None):
        print(f"Staging data from MongoDB: {collection_names}", flush=True)
        self.mongo_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = self.mongo_temp_dir / "export.jsonl"

        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("bboxes", ArrayType(StringType()), True),
            StructField("img", StringType(), True),
        ])

        with open(temp_file_path, "w") as f:
            for coll_name in collection_names:
                try:
                    collection = self.mongo.db[coll_name]

                    if limit is not None:
                        collection_limit = max(1, limit // len(collection_names))
                        pipeline = [
                            {"$sample": {"size": collection_limit}},
                            {"$project": {"_id": 0, "timestamp": 1, "bboxes": 1, "img": 1}}
                        ]
                        cursor = collection.aggregate(pipeline)
                    else:
                        cursor = collection.find({}, {"_id": 0, "timestamp": 1, "bboxes": 1, "img": 1})

                    for doc in cursor:
                        record = {
                            "timestamp": str(doc.get("timestamp", "")),
                            "bboxes": doc.get("bboxes", []),
                            "img": doc.get("img", "")
                        }
                        f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"Error staging {coll_name}: {e}", file=sys.stderr)

        return self.spark.read.schema(schema).json(str(temp_file_path))

    def prepare_training_data(self, collection_names, limit=None):
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)

        for split in ['train', 'val']:
            (self.dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

        df_spark = self.load_data_from_mongodb(collection_names, limit=limit)
        train_df, val_df = df_spark.randomSplit([0.8, 0.2], seed=42)

        def save_partition(split_name, base_dir):
            def _process(partition):
                images_dir = os.path.join(base_dir, 'images', split_name)
                labels_dir = os.path.join(base_dir, 'labels', split_name)

                for row in partition:
                    try:
                        if not row.img: continue

                        file_id = uuid.uuid4().hex

                        img_data = base64.b64decode(row.img.encode('utf-8'))
                        img = Image.open(BytesIO(img_data)).convert('RGB')
                        img.save(os.path.join(images_dir, f"{file_id}.jpg"), "JPEG")

                        bboxes = row.bboxes if row.bboxes else []
                        with open(os.path.join(labels_dir, f"{file_id}.txt"), 'w') as f:
                            f.write('\n'.join(bboxes))

                    except Exception as e:
                        continue

            return _process

        print("Writing detection dataset to disk...", flush=True)
        train_df.rdd.foreachPartition(save_partition("train", str(self.dataset_path)))
        val_df.rdd.foreachPartition(save_partition("val", str(self.dataset_path)))

        self._create_yaml_config()

        return str(self.data_yaml_path)

    def _create_yaml_config(self):
        yaml_content = f"""path: {self.dataset_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: person
"""
        with open(self.data_yaml_path, "w") as f:
            f.write(yaml_content)

    def load_model(self):
        model_name = f"yolo11{self.model_size}.pt"
        print(f"Loading Detection Model: {model_name}", flush=True)
        self.model = YOLO(model_name)
        return self.model

    def fine_tune(self, collection_names, epochs=50, imgsz=640, batch_size=16, device='mps', limit=None):
        if self.model is None:
            self.load_model()

        yaml_path_str = self.prepare_training_data(collection_names, limit=limit)

        return self.model.train(
            data=yaml_path_str,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            task='detect',
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