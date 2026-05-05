from pymongo import MongoClient
import os

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
        """Drop collection if it exists, ensuring fresh start for imports."""
        self.db[collectionName].drop()

    def saveImage(self, img, timestamp):
        pass
