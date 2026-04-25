import pandas as pd 
import numpy as np 

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.x_train = X 
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        most_common = unique[np.argmax(counts)]
        return most_common


class GestureEngine:
    def __init__(self):
        # ==================================================
        # 1. LOAD DATASET
        # ==================================================
        self.data_path = 'dataset/Datafull terakhir test.csv'
        self.df = pd.read_csv(self.data_path, sep=';')
        
        # ==================================================
        # 2. PRE-PROCESSING DATA
        # ==================================================
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        coord_cols = self.df.columns.drop('char')
        self.df[coord_cols] = self.df[coord_cols].apply(pd.to_numeric, errors='coerce')
        
        for col in coord_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
            
        self.df = self.df.drop_duplicates()
        
        self.df = self.df[self.df['char'].notna()]
        self.df['char'] = self.df['char'].astype(str)
        self.df = self.df[self.df['char'].str.match(r'^[A-Za-z]$')]
        self.df['char'] = self.df['char'].str.upper()
        
        # ==================================================
        # 3. ENCODING DATA
        # ==================================================
        unique_chars = sorted(self.df['char'].unique())
        
        self.char_to_int = {char: idx for idx, char in enumerate(unique_chars)}
        self.int_to_char = {idx: char for char, idx in self.char_to_int.items()}
        
        self.df['char_encoded'] = self.df['char'].map(self.char_to_int)
        
        # ==================================================
        # 4. TRAINING DATA PREPARATION
        # ==================================================
        X_raw = self.df.drop(["char", "char_encoded"], axis=1).values
        y = self.df["char_encoded"].values
        
        X_processed = []
        for x in X_raw:
            x = x.reshape(1, -1)
            x_prep = self.preprocess_single_hand(x)
            X_processed.append(x_prep[0])
            
        X = np.array(X_processed)
        
        # ==================================================
        # 5. INISIALISASI MODEL KNN
        # ==================================================
        print("[INFO] Memproses dataset yang sudah dibersihkan...")
        self.classifier = KNN(k=3)
        self.classifier.fit(X, y)
        print("[INFO] Model Engine Siap! Dataset Terverifikasi.")

    @staticmethod
    def preprocess_single_hand(A):
        A = A.copy()
        coords = A.reshape(-1, 2)
        coords -= coords[0]
        
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist != 0:
            coords /= max_dist
            
        ref = coords[9]
        angle = np.arctan2(ref[1], ref[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        coords = coords @ rot.T
        
        return coords.flatten().reshape(1, -1)

    def predict_gesture(self, landmarks):
        """Memproses landmark baru dan mengembalikan huruf tebakan."""
        A = self.preprocess_single_hand(landmarks)
        
        pred = self.classifier.predict(A)
        pred_label = int(pred[0])
        
        result_char = self.int_to_char[pred_label]
        return result_char