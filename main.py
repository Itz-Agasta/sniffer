#!/usr/bin/env python3
"""
refine_beef.py – turn 5 Mendeley beef CSVs into fridge-ready spoilage dataset
Sensors kept: MQ135, Temperature, Humidity  (exactly what I wired)
Output: 60-s windowed 5-D features + 3-class labels
"""

import os, json, numpy as np, pandas as pd

RAW_FILES = [f'raw dataset/TS{i}.csv' for i in range(1,6)]
# Exclude TS4 (outlier with 99% humidity) and use remaining files
VALID_FILES = [f for f in RAW_FILES if f != 'raw dataset/TS4.csv']  # TS1, TS2, TS3, TS5

TRAIN_FILES = VALID_FILES[:2]  # TS1, TS2
VAL_FILES   = VALID_FILES[2:3]  # TS3
TEST_FILES  = VALID_FILES[3:4]  # TS5

LABEL_MAP = {'excellent':0, 'good':1, 'acceptable':1, 'spoiled':2}
FEAT_COLS = ['MQ135','Temperature','Humidity']
OUT_DIR = 'refined'
os.makedirs(OUT_DIR, exist_ok=True)

def build_features(df):
    WINDOW = 60
    feats, labs = [], []
    for i in range(WINDOW, len(df)):
        # Get the current window
        window_data = df['MQ135'].iloc[i-WINDOW:i+1]
        R_now = df['MQ135'].iloc[i]
        R_prev= df['MQ135'].iloc[i-1]
        T_now = df['Temperature'].iloc[i]
        H_now = df['Humidity'].iloc[i]
        minute= df['minute'].iloc[i]

        # Normalize R_norm within the current window
        R_min = window_data.min()
        R_max = window_data.max()
        R_norm = (R_now - R_min) / (R_max - R_min + 1e-3)
        
        dR_dt  = (R_now - R_prev) / 60.
        # Normalize dR_dt to reasonable range (assuming max change of 1 ohm per second)
        dR_dt_norm = dR_dt / 1.0  # Scale to [-1, 1] approximately
        
        T_comp = max(T_now - 4.0, 0)
        # Normalize T_comp (assuming fridge temperatures 4-40°C, so T_comp 0-36°C)
        T_comp_norm = T_comp / 40.0  # Scale to [0, 0.9] approximately
        
        H_norm = H_now / 100.
        hour   = (minute % 1440) / 1440.

        feats.append([R_norm, dR_dt_norm, T_comp_norm, H_norm, hour])
        labs.append(LABEL_MAP[df['class'].iloc[i]])
    return np.array(feats), np.array(labs)

def load_set(flist):
    X, y = [], []
    for file in flist:
        df = pd.read_csv(file, usecols=['minute','class','MQ135','Temperature','Humidity'])
        fx, fy = build_features(df)
        X.append(fx); y.append(fy)
    return np.vstack(X), np.hstack(y)

def main():
    # ---- main ----
    X_train, y_train = load_set(TRAIN_FILES)
    X_val,   y_val   = load_set(VAL_FILES)
    X_test,  y_test  = load_set(TEST_FILES)

    np.save(os.path.join(OUT_DIR,'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR,'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR,'X_val.npy'),   X_val)
    np.save(os.path.join(OUT_DIR,'y_val.npy'),   y_val)
    np.save(os.path.join(OUT_DIR,'X_test.npy'),  X_test)
    np.save(os.path.join(OUT_DIR,'y_test.npy'),  y_test)

    json.dump({' Fresh':0,'Spoiling':1,'Spoiled':2},
              open(os.path.join(OUT_DIR,'label_map.json'),'w'), indent=2)

    with open(os.path.join(OUT_DIR,'stats.txt'),'w') as f:
        for name, yy in zip(('Train','Val','Test'),(y_train,y_val,y_test)):
            uniq, cnt = np.unique(yy, return_counts=True)
            f.write(f'{name}: {dict(zip(map(int,uniq),map(int,cnt)))}\n')
    print('Refinery done – files in ./refined/')


if __name__ == "__main__":
    main()
