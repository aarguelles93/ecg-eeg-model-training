import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split

def split_eeg_dataset(
    file_path,
    output_train_path,
    output_test_path,
    test_size=0.2,
    random_state=42
):
    # 1) Load raw EEG channels (no header, 32 columns)
    df = pd.read_csv(file_path, header=None)
    
    # 2) Drop any empty “Unnamed” column if present
    if df.shape[1] > 32:
        df = df.iloc[:, :32]
    
    # 3) Create labels: 1 = EEG
    df['label'] = 1
    
    # 4) Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # 5) Save them out
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    print(f"Saved {len(train_df)} train rows to {output_train_path}")
    print(f"Saved {len(test_df)} test rows to {output_test_path}")

split_eeg_dataset(
  'data/eeg_dataset.csv',
  'data/eeg_train.csv',
  'data/eeg_test.csv'
)
