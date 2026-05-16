import cv2
import time
from ultralytics import YOLO

from MongoDBImporter import MongoDBImporter
from camera_feed import ThreadedCamera

model_name = "models/test_model_v3.pt"
model = YOLO(model_name)

MongoDBImporter().clearCollection("cam0_16_5_2026")

def draw_overlays(result, original_frame, idx, cam_id):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        display_frame = result.plot()
        people_pred = len(result.boxes)
    except Exception as e:
        print(f"Overlay error on Cam {cam_id}: {e}", flush=True)
        people_pred = -1
        display_frame = original_frame.copy()

    cv2.putText(display_frame, f"Cam {cam_id} | Frame: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, timestamp, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    color = (0, 128, 255) if people_pred >= 0 else (0, 0, 255)
    text = f"People: {people_pred}" if people_pred >= 0 else "People: ?"
    cv2.putText(display_frame, text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return display_frame


def main():
    print("Initializing cameras...", flush=True)
    cam0 = ThreadedCamera("cam0", device_index=0)
    cam1 = ThreadedCamera("cam1", device_index=1)

    idx = 0
    window_name = "Dual Live Detection Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        last_results = None
        while True:
            ret0, frame0 = cam0.read()
            ret1, frame1 = cam1.read()

            if not ret0 or not ret1:
                continue

            try:
                if idx % 2 == 1:
                    results = model.predict([frame0, frame1], classes=[0], verbose=False)
                    last_results = results
                else:
                    results = last_results

                out0 = draw_overlays(results[0], frame0, idx, cam_id=0)
                out1 = draw_overlays(results[1], frame1, idx, cam_id=1)

            except Exception as e:
                print(f"Batch prediction error: {e}", flush=True)
                out0 = frame0
                out1 = frame1

            h, w = out0.shape[:2]
            out1_resized = cv2.resize(out1, (w, h))

            combined_feed = cv2.hconcat([out0, out1_resized])

            cv2.imshow(window_name, combined_feed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            idx += 1
    finally:
        print("Releasing hardware...", flush=True)
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()