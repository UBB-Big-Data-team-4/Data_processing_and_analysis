import time
import os
import csv
import pandas as pd
import DataProcessingUpdated as dp

output_file = "mock_data.csv"

# create file if not exists
if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "people"])


while True:
    # 1. generate new row
    row = dp.getProcessedData()

    # 2. append row
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row["timestamp"], row["people"]])

    # 3. load full dataset
    df = pd.read_csv(output_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 4. keep only last 5 (sau oricate) minutes
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=5)
    df = df[df["timestamp"] >= cutoff]

    # 5. overwrite CSV with filtered data
    df.to_csv(output_file, index=False)

    print(row)

    # 6. wait before next update (se poate modif cu alt timp)
    time.sleep(5)