from ModelFactory import ModelFactory

if __name__ == "__main__":
    # importer = EdinburghDataImporter([i for i in range(2,21)], (640, 640))
    # importer.importData()

    factory = ModelFactory(model_size='n')
    results = factory.fine_tune(
        collection_names=[f"day{i:02d}" for i in range(1, 21)],
        device="mps",
        epochs=10,
        batch_size=128
    )

    print("Fine-tuning complete!", flush=True)
    print("Results:", results)

    factory.save_model("models/test_model_v1")
