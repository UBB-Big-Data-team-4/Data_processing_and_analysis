from ModelFactory import ModelFactory

factory = ModelFactory(model_size='n')
results = factory.fine_tune(
    collection_names=[f"day{i:02d}" for i in range(1, 21)]+["unity"],
    device="mps",
    epochs=10,
    batch_size=16,
    limit=25000
)

print("Fine-tuning complete!", flush=True)
print("Results:", results)

factory.save_model("models/test_model_v3.pt")

factory.close()