from DataImporter import EdinburghDataImporter, UnityImporter
from MongoDBImporter import MongoDBImporter

# for day in ["unity"]:
#     importer = MongoDBImporter()
#     importer.clearCollection(day)
#
# importer = EdinburghDataImporter(days=[day for day in range(1, 2)], imgSize=(640,640))
# importer.importData()

importer = UnityImporter("./data/unity")
importer.import_all(collection_name="unity")