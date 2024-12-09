# Import library
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
file_path = r'C:\Users\ASUS\OneDrive\Desktop\stress-level\student_lifestyle_dataset.csv'
data = pd.read_csv(file_path)

# Preprocessing
# Encode kolom 'Stress_Level' menjadi numerik
label_encoder = LabelEncoder()
data['Stress_Level_Encoded'] = label_encoder.fit_transform(data['Stress_Level'])

# Pilih fitur dan target
X = data[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
          'Sleep_Hours_Per_Day', 
          'Physical_Activity_Hours_Per_Day']]
y = data['Stress_Level_Encoded']

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model training dengan cross-validation
model = RandomForestClassifier(random_state=42)

# Gunakan cross-validation untuk mengevaluasi model
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5 fold cross-validation
print("Akurasi Cross-Validation:", cv_scores.mean())

# Hyperparameter tuning menggunakan GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],  # Jumlah pohon
    'max_depth': [None, 10, 20, 30],  # Kedalaman maksimal pohon
    'min_samples_split': [2, 5, 10],  # Minimal sampel untuk memecah node
    'min_samples_leaf': [1, 2, 4]  # Minimal sampel di daun
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Menampilkan hasil tuning terbaik
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluasi model terbaik pada data uji
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Cetak hasil evaluasi
print("Akurasi Model Setelah Tuning:", accuracy)
print("\nLaporan Klasifikasi:\n", report)

# Simpan model dan label encoder
joblib.dump(best_model, 'stress_model_tuned.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'model/scaler.pkl')  # Menyimpan scaler


print("\nModel dan label encoder berhasil disimpan!")
