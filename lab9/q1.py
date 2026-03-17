import numpy as np
import pandas as pd

def main():
    np.random.seed(10)
    n = 100  # number of samples

    data = pd.DataFrame({
        "BP": np.random.normal(80, 5, n),
        "Age": np.random.randint(25, 65, n),
    })

    # Simulated regression target variable (y)
    data["Target"] = 0.5 * data["BP"] + 0.2 * data["Age"] + np.random.normal(0, 2, n)

    print("First 5 rows of dataset:")
    print(data.head())

    # Function to partition dataset based on threshold
    def partition_dataset(df, threshold):
        left_partition = df[df["BP"] <= threshold]
        right_partition = df[df["BP"] > threshold]
        return left_partition, right_partition

    # Threshold values
    thresholds = [80, 78, 82]

    for t in thresholds:
        left, right = partition_dataset(data, t)
        print(f"\nThreshold t = {t}")
        print(f"Left Partition (BP <= {t}): {len(left)} samples")
        print(f"Right Partition (BP > {t}): {len(right)} samples")

if __name__ == '__main__':
    main()