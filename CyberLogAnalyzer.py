import pandas as pd
import json
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Log Yükleme ve İşleme ---
def load_logs(file_path):
    """
    Log dosyasını yükler (JSON veya CSV destekler).
    """
    if file_path.endswith('.csv'):
        logs = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            logs = pd.DataFrame(json.load(file))
    else:
        raise ValueError("Dosya formatı desteklenmiyor. Lütfen JSON veya CSV kullanın.")
    return logs

# --- 2. Anomali Tespiti ---
def detect_anomalies(logs, feature_columns, contamination_rate=0.05):
    """
    IsolationForest algoritması ile anomali tespiti yapar.
    """
    print("[INFO] Anomali tespiti başlatılıyor...")
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    logs['anomaly'] = model.fit_predict(logs[feature_columns])
    logs['anomaly'] = logs['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return logs

# --- 3. Görselleştirme ---
def visualize_results(logs, feature_columns):
    """
    Anomali tespit sonuçlarını görselleştirir.
    """
    sns.pairplot(logs, hue='anomaly', vars=feature_columns, palette={'Anomaly': 'red', 'Normal': 'blue'})
    plt.title("Anomaly Detection Results")
    plt.show()

# --- 4. JSON ve CSV Raporlama ---
def save_results(logs, output_dir='results'):
    """
    Anomali sonuçlarını JSON ve CSV formatında kaydeder.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logs.to_csv(f'{output_dir}/anomaly_results.csv', index=False)
    logs.to_json(f'{output_dir}/anomaly_results.json', orient='records', lines=True)
    print(f"[INFO] Sonuçlar {output_dir} dizinine kaydedildi.")

# --- Ana Çalışma Akışı ---
if __name__ == "__main__":
    # Örnek dosya yolları
    file_path = 'example_logs.csv'  # Log dosyanızın yolu
    feature_columns = ['response_time', 'request_size', 'status_code']  # Özellik sütunları

    try:
        # 1. Logları yükle
        logs = load_logs(file_path)
        print("[INFO] Loglar başarıyla yüklendi.")

        # 2. Anomali tespiti yap
        logs = detect_anomalies(logs, feature_columns)
        print("[INFO] Anomali tespiti tamamlandı.")

        # 3. Sonuçları görselleştir
        visualize_results(logs, feature_columns)

        # 4. Sonuçları kaydet
        save_results(logs)

    except Exception as e:
        print(f"[ERROR] Bir hata oluştu: {e}")
