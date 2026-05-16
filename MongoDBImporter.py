from pymongo import MongoClient
import os
import datetime
import base64
import cv2
from pymongo.errors import PyMongoError

class MongoDBImporter:
    def __init__(
        self,
        host=None,
        port=None,
        db_name=None,
        username=None,
        password=None,
        auth_source=None,
    ):
        host = host or os.getenv('MONGO_HOST', 'localhost')
        port = int(port or os.getenv('MONGO_PORT', '27017'))
        db_name = db_name or os.getenv('MONGO_DB', 'bigdata')
        username = username or os.getenv('MONGO_USER', 'admin')
        password = password or os.getenv('MONGO_PASS', 'securepassword')
        auth_source = auth_source or os.getenv('MONGO_AUTH_SOURCE', 'admin')

        client_kwargs = {'host': host, 'port': port}
        if username and password:
            client_kwargs.update(
                {
                    'username': username,
                    'password': password,
                    'authSource': auth_source,
                }
            )

        self.client = MongoClient(**client_kwargs)
        self.db = self.client[db_name]
        self.auth_source = auth_source

    def test_connection(self):
        try:
            self.client.admin.command("ping")
            return True, "MongoDB connection successful."
        except PyMongoError as exc:
            return False, (
                f"MongoDB connection failed (authSource={self.auth_source}): {exc}"
            )

    def addCollection(self, collectionName, newCollection):
        collection = self.db[collectionName]
        collection.insert_many(newCollection)

    def clearCollection(self, collectionName):
        print(f"Clearing collection {collectionName}", flush=True)
        self.db[collectionName].drop()

    def saveImage(self, img, camera_name):
        if img is None:
            return False, "No image provided"

        safe_camera = str(camera_name).replace(" ", "_")
        now = datetime.datetime.now(datetime.timezone.utc)
        day = now.day
        month = now.month
        year = now.year
        collection_name = f"{safe_camera}_{day}_{month}_{year}"

        try:
            if isinstance(img, (bytes, bytearray)):
                img_bytes = bytes(img)
            else:
                ok, buf = cv2.imencode('.jpg', img)
                if not ok:
                    raise ValueError('cv2.imencode failed to encode image')
                img_bytes = buf.tobytes()

            img_b64 = base64.b64encode(img_bytes).decode('ascii')
            doc = {"timestamp": now, "image": img_b64}
            self.db[collection_name].insert_one(doc)

        except PyMongoError as exc:
            print(f"MongoDB insert failed (authSource={self.auth_source}): {exc}")
        except Exception as exc:
            print(f"saveImage error: {exc}")
