from ultralytics import YOLO

from ModelFactory import ModelFactory

MODEL_NAME = "models/test_model_v3.pt"

factory = ModelFactory(model_size='n')
#--index-url https://download.pytorch.org/whl/cu121
results = factory.fine_tune(
    collection_names=[f"day{i:02d}" for i in range(1, 21)]+["unity"],
    device=0,
    epochs=10,
    batch_size=16,
    limit=10000
)

print("Fine-tuning complete!", flush=True)
print("Results:", results)

factory.save_model(MODEL_NAME)
model = YOLO("models/test_model_v3.pt")
model.to('cpu')
model.export(format="coreml")

# factory.close()