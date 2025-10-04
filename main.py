#!/usr/bin/env python3
"""
refine_beef.py – turn 5 Mendeley beef CSVs into fridge-ready spoilage dataset
Sensors kept: MQ135, Temperature, Humidity  (exactly what I wired)
Output: 60-s windowed 5-D features + 3-class labels
"""

import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW_FILES = [f'raw dataset/TS{i}.csv' for i in range(1,6)]
TRAIN_FILES = RAW_FILES[:2]
VAL_FILES   = RAW_FILES[2:3]
TEST_FILES  = RAW_FILES[3:5]

WINDOW = 60
LABEL_MAP = {'excellent':0, 'good':1, 'acceptable':1, 'spoiled':2}
FEAT_COLS = ['MQ135','Temperature','Humidity']
OUT_DIR = 'refined'
os.makedirs(OUT_DIR, exist_ok=True)

def build_features(df):
    feats, labs = [], []
    r = df['MQ135'].rolling(window=WINDOW)
    r_min, r_max = r.min(), r.max()
    for i in range(WINDOW, len(df)):
        R_now = df['MQ135'].iloc[i]
        R_prev= df['MQ135'].iloc[i-1]
        T_now = df['Temperature'].iloc[i]
        H_now = df['Humidity'].iloc[i]
        minute= df['minute'].iloc[i]

        R_norm = (R_now - r_min.iloc[i]) / (r_max.iloc[i] - r_min.iloc[i] + 1e-3)
        dR_dt  = (R_now - R_prev) / 60.
        T_comp = max(T_now - 4.0, 0)
        H_norm = H_now / 100.
        hour   = (minute % 1440) / 1440.

        feats.append([R_norm, dR_dt, T_comp, H_norm, hour])
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

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    np.save(os.path.join(OUT_DIR,'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR,'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR,'X_val.npy'),   X_val)
    np.save(os.path.join(OUT_DIR,'y_val.npy'),   y_val)
    np.save(os.path.join(OUT_DIR,'X_test.npy'),  X_test)
    np.save(os.path.join(OUT_DIR,'y_test.npy'),  y_test)

    json.dump({' Fresh':0,'Spoiling':1,'Spoiled':2},
              open(os.path.join(OUT_DIR,'label_map.json'),'w'), indent=2)
    json.dump({'scale_':scaler.scale_.tolist(),
               'min_':  scaler.min_.tolist()},
              open(os.path.join(OUT_DIR,'scaler.json'),'w'), indent=2)

    with open(os.path.join(OUT_DIR,'stats.txt'),'w') as f:
        for name, yy in zip(('Train','Val','Test'),(y_train,y_val,y_test)):
            uniq, cnt = np.unique(yy, return_counts=True)
            f.write(f'{name}: {dict(zip(map(int,uniq),map(int,cnt)))}\n')
    print('Refinery done – files in ./refined/')


if __name__ == "__main__":
    main()
