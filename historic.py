import cv2
import base64
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from MongoDBImporter import MongoDBImporter
from ModelSingleton import ModelSingleton


def analyze_and_play(collection_name, model_name="test_model_v3.pt"):
    print(f"Fetching images and running inference for '{collection_name}'...", flush=True)

    mongo = MongoDBImporter()
    model = ModelSingleton(model_name)
    collection = mongo.db[collection_name]

    cursor = collection.find({}, {"_id": 0, "timestamp": 1, "image": 1, "img": 1}).sort("timestamp", 1)

    records = []
    frames = []

    for doc in cursor:
        img_b64 = doc.get("image") or doc.get("img")
        if not img_b64:
            continue

        clean_b64 = img_b64.split(',')[-1]

        try:
            img_bytes = base64.b64decode(clean_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            count = model.predict(frame)

            records.append({
                "timestamp": str(doc.get("timestamp")),
                "people": max(0, count)
            })

            cv2.putText(frame, f"People Count: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            frames.append(frame)

        except Exception as e:
            print(f"Skipping malformed frame: {e}")
            continue

    if not records:
        print("No valid data found in the collection.")
        return

    print("Generating R code analysis graphs...", flush=True)
    df = pd.DataFrame(records)

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv['df'] = ro.conversion.py2rpy(df)

    with open("data_analysis_updated.R", "r") as file:
        r_script = file.read()

    ro.r(r_script)

    graphs_img = cv2.imread("temp_r_plots.png")
    if graphs_img is None:
        print("Failed to read the generated R graphs.")
        return

    print("Starting combined playback...", flush=True)
    window_name = f"Analysis: {collection_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in frames:
        f_h, f_w = frame.shape[:2]
        g_h, g_w = graphs_img.shape[:2]

        target_width = max(f_w, g_w)

        if f_w != target_width:
            frame = cv2.resize(frame, (target_width, int(f_h * (target_width / f_w))))

        if g_w != target_width:
            current_graphs = cv2.resize(graphs_img, (target_width, int(g_h * (target_width / g_w))))
        else:
            current_graphs = graphs_img

        combined_display = cv2.vconcat([frame, current_graphs])

        cv2.imshow(window_name, combined_display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_and_play("cam0_16_5_2026")