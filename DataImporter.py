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
    def _frame_people_info(label):
        people_count = 0
        active_states = []

        for person in label:
            state = int(person[4])
            if state > 0:
                people_count += 1
                active_states.append(state)

        single_sitting = people_count == 1 and active_states[0] == 2
        return people_count, single_sitting

    @staticmethod
    def _parse_timestamp(name):
        tokens = name.replace('.', '_').split('_')
        year = int(tokens[1])
        month = int(tokens[2])
        day_of_month = int(tokens[3])
        hour = int(tokens[4])
        minute = int(tokens[5])
        second = int(tokens[6])
        return datetime.datetime(year, month, day_of_month, hour, minute, second)

    @staticmethod
    def _print_progress(current, total, prefix=''):
        if total <= 0:
            return

        width = 40
        ratio = current / total
        filled = int(width * ratio)
        bar = '█' * filled + '░' * (width - filled)
        percent = ratio * 100
        end = '\r' if current < total else '\n'
        msg = f"{prefix}[{bar}] {current:6d}/{total:6d} ({percent:5.1f}%)"
        sys.stdout.write(msg + end)
        sys.stdout.flush()

    @staticmethod
    def _image_to_base64(img_path, img_size):
        """Load and encode image as base64 string to save memory."""
        with Image.open(img_path) as img:
            resized = img.resize(img_size)
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def importDay(self, day):
        day = int(day)
        print(f"Importing data for day{day:02d} in Edinburgh.")
        base_dir = Path(__file__).parent / 'data' / 'ed' / f'day{day:02d}'
        
        print(f"Loading metadata...", flush=True)
        mat = sio.loadmat(str(base_dir / f'day{day:02d}.mat'))
        labels = mat['labels'][0]
        mat = sio.loadmat(str(base_dir / f'Name{day:02d}.mat'))
        names = mat['name'][0]
        total_frames = min(len(labels), len(names))
        if len(labels) != len(names):
            print(
                f"Warning: labels ({len(labels)}) and names ({len(names)}) length mismatch. "
                f"Using first {total_frames} frames.",
                file=sys.stderr,
            )

        collection = []
        batch_size = 100
        total_imported = 0
        batches_inserted = 0
        mongo = MongoDBImporter()
        collection_name = f"day{day:02d}"
        
        # Clear existing collection before importing
        print(f"Clearing existing collection '{collection_name}'...", flush=True)
        mongo.clearCollection(collection_name)
        
        for index in range(total_frames):
            people_count, single_sitting = self._frame_people_info(labels[index])

            should_import = True
            if people_count == 0:
                should_import = index % 20 == 0
            elif single_sitting:
                should_import = index % 5 == 0

            if should_import:
                name = names[index][0]
                timestamp = self._parse_timestamp(name)
                img_path = base_dir / 'img' / name
                
                try:
                    img_base64 = self._image_to_base64(img_path, self.imgSize)
                    collection.append(
                        {
                            'timestamp': timestamp,
                            'people': people_count,
                            'img': img_base64,
                        }
                    )
                    total_imported += 1
                except Exception as e:
                    print(f"\nWarning: Failed to process image {name}: {e}", file=sys.stderr, flush=True)
                    continue

                if len(collection) >= batch_size:
                    try:
                        print(f"\nInserting batch {batches_inserted + 1} ({len(collection)} documents)...", flush=True)
                        mongo.addCollection(collection_name, collection)
                        batches_inserted += 1
                        print(f"Batch {batches_inserted} inserted successfully.", flush=True)
                        collection = []
                    except Exception as e:
                        print(f"\nERROR: Failed to insert batch: {e}", file=sys.stderr, flush=True)
                        raise

            self._print_progress(index + 1, total_frames, prefix=f"day{day:02d} ")

        if collection:
            try:
                print(f"\nInserting final batch ({len(collection)} documents)...", flush=True)
                mongo.addCollection(collection_name, collection)
                batches_inserted += 1
                print(f"Final batch inserted successfully.", flush=True)
            except Exception as e:
                print(f"\nERROR: Failed to insert final batch: {e}", file=sys.stderr, flush=True)
                raise

        # Verify import
        final_count = mongo.db[collection_name].count_documents({})
        print(f"Import complete for day{day:02d}. Total frames processed: {total_imported}, total documents in DB: {final_count}.", flush=True)


