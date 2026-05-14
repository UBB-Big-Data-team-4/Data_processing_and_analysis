import scipy.io as sio
import datetime
import sys
import base64
import io
from pathlib import Path
from PIL import Image

from MongoDBImporter import MongoDBImporter

class EdinburghDataImporter:
    def __init__(self, days, imgSize):
        self.days = days
        self.imgSize = imgSize

    def importData(self):
        for day in self.days:
            self.importDay(day)

    @staticmethod
    def _extract_yolo_boxes(label_matrix, img_width, img_height):
        boxes = []
        for person in label_matrix:
            x_tl, y_tl, w, h = person[0:4]

            if w == 0 and h == 0:
                continue

            x_center = (x_tl + (w / 2.0)) / img_width
            y_center = (y_tl + (h / 2.0)) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            boxes.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
        return boxes

    @staticmethod
    def _parse_timestamp(name):
        tokens = name.replace('.', '_').split('_')
        return datetime.datetime(
            int(tokens[1]), int(tokens[2]), int(tokens[3]),
            int(tokens[4]), int(tokens[5]), int(tokens[6])
        )

    @staticmethod
    def _print_progress(current, total, prefix=''):
        if total <= 0: return
        width = 40
        ratio = current / total
        filled = int(width * ratio)
        bar = '█' * filled + '░' * (width - filled)
        percent = ratio * 100
        end = '\r' if current < total else '\n'
        sys.stdout.write(f"{prefix}[{bar}] {current:6d}/{total:6d} ({percent:5.1f}%){end}")
        sys.stdout.flush()

    def importDay(self, day):
        day = int(day)
        print(f"Importing data for day{day:02d} in Edinburgh.")
        base_dir = Path(__file__).parent / 'data' / 'ed' / f'day{day:02d}'

        mat = sio.loadmat(str(base_dir / f'day{day:02d}.mat'))
        labels = mat['labels'][0]
        mat = sio.loadmat(str(base_dir / f'Name{day:02d}.mat'))
        names = mat['name'][0]
        total_frames = min(len(labels), len(names))

        collection = []
        batch_size = 100
        total_imported = 0
        batches_inserted = 0
        mongo = MongoDBImporter()
        collection_name = f"day{day:02d}"

        for index in range(total_frames):
            active_states = [int(p[4]) for p in labels[index] if int(p[4]) > 0]
            people_present = len(active_states) > 0
            single_sitting = len(active_states) == 1 and active_states[0] == 2

            should_import = True
            if not people_present:
                should_import = index % 20 == 0
            elif single_sitting:
                should_import = index % 5 == 0

            if should_import:
                name = names[index][0]
                img_path = base_dir / 'img' / name

                try:
                    with Image.open(img_path) as img:
                        orig_width, orig_height = img.size
                        yolo_boxes = self._extract_yolo_boxes(labels[index], orig_width, orig_height)
                        resized = img.resize(self.imgSize)
                        buffer = io.BytesIO()
                        resized.save(buffer, format='JPEG', quality=85)
                        buffer.seek(0)
                        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                    collection.append(
                        {
                            'timestamp': self._parse_timestamp(name),
                            'bboxes': yolo_boxes,
                            'img': img_base64,
                        }
                    )
                    total_imported += 1
                except Exception as e:
                    print(f"\nWarning: Failed to process {name}: {e}", file=sys.stderr)
                    continue

                if len(collection) >= batch_size:
                    mongo.addCollection(collection_name, collection)
                    batches_inserted += 1
                    collection = []

            self._print_progress(index + 1, total_frames, prefix=f"day{day:02d} ")

        if collection:
            mongo.addCollection(collection_name, collection)
            batches_inserted += 1

        final_count = mongo.db[collection_name].count_documents({})
        print(
            f"\nImport complete for day{day:02d}. Total frames processed: {total_imported}, documents in DB: {final_count}.",
            flush=True)


class UnityImporter:
    def __init__(self, data_folder, target_size=(640, 640), batch_size=100):
        self.data_folder = Path(data_folder)
        self.target_width, self.target_height = target_size
        self.batch_size = batch_size
        self.mongo = MongoDBImporter()

    def _convert_to_yolo(self, bbox_line):
        parts = [float(p) for p in bbox_line.strip().split()]
        if len(parts) != 4:
            return None

        x_tl, y_tl, w, h = parts

        x_center = (x_tl + (w / 2.0)) / self.target_width
        y_center = (y_tl + (h / 2.0)) / self.target_height
        w_norm = w / self.target_width
        h_norm = h / self.target_height

        return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    def import_all(self, collection_name="unity"):
        txt_files = list(self.data_folder.glob("people_*.txt"))
        if not txt_files:
            print(f"No people_*.txt files found in {self.data_folder}")
            return

        total_inserted = 0

        for file_path in txt_files:
            try:
                n_people = int(file_path.stem.split('_')[1])
            except (IndexError, ValueError):
                print(f"Skipping {file_path.name}: Unable to determine person count from filename.")
                continue

            print(f"Importing {file_path.name} (Chunk size: {n_people + 1} lines)...")
            collection = []

            with open(file_path, 'r') as f:
                while True:
                    img_line = f.readline()
                    if not img_line:
                        break

                    img_base64 = img_line.strip()
                    if not img_base64:
                        continue

                    current_bboxes = []
                    for _ in range(n_people):
                        bbox_line = f.readline()
                        if bbox_line:
                            yolo_box = self._convert_to_yolo(bbox_line)
                            if yolo_box:
                                current_bboxes.append(yolo_box)

                    collection.append({
                        'bboxes': current_bboxes,
                        'img': img_base64
                    })

                    if len(collection) >= self.batch_size:
                        self.mongo.addCollection(collection_name, collection)
                        total_inserted += len(collection)
                        collection = []

            if collection:
                self.mongo.addCollection(collection_name, collection)
                total_inserted += len(collection)

        print(f"Import complete. Inserted {total_inserted} frames into MongoDB collection '{collection_name}'.")