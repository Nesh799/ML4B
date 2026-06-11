import pandas as pd
import argparse

def transform_sensor_data(acc_file, gyro_file, output_file, activity_label="unbekannt"):
    """
    Liest Accelerometer- und Gyroscope-Daten ein und formatiert sie
    in das Zielformat.
    """
    
    # 1. Daten einlesen
    df_acc = pd.read_csv(acc_file)
    df_gyro = pd.read_csv(gyro_file)
    
    # 2. Spalten umbenennen
    df_acc = df_acc.rename(columns={
        'time': 'timestamp',
        'x': 'acc_x',
        'y': 'acc_y',
        'z': 'acc_z'
    })
    
    df_gyro = df_gyro.rename(columns={
        'time': 'timestamp',
        'x': 'gyro_x',
        'y': 'gyro_y',
        'z': 'gyro_z'
    })
    
    # 3. Nicht benötigte Spalte 'seconds_elapsed' entfernen
    if 'seconds_elapsed' in df_acc.columns:
        df_acc = df_acc.drop(columns=['seconds_elapsed'])
    if 'seconds_elapsed' in df_gyro.columns:
        df_gyro = df_gyro.drop(columns=['seconds_elapsed'])
        
    # 4. Daten nach Zeitstempel sortieren (zwingend erforderlich für merge_asof)
    df_acc = df_acc.sort_values('timestamp')
    df_gyro = df_gyro.sort_values('timestamp')
    
    # 5. Datensätze anhand der Zeitstempel zusammenführen
    # 'nearest' sucht den zeitlich nächstliegenden Gyroskop-Wert für jeden Beschleunigungswert
    df_combined = pd.merge_asof(df_acc, df_gyro, on='timestamp', direction='nearest')
    
    # 6. Fehlende Spalte 'activity' hinzufügen
    df_combined['activity'] = activity_label
    
    # 7. Spalten in die exakte Reihenfolge der Zieldatei bringen
    target_columns = [
        'timestamp', 
        'acc_x', 'acc_y', 'acc_z', 
        'gyro_x', 'gyro_y', 'gyro_z', 
        'activity'
    ]
    df_combined = df_combined[target_columns]
    
    # 8. Ergebnis als CSV speichern
    df_combined.to_csv(output_file, index=False)
    print(f"Fertig! Datei wurde gespeichert als: {output_file}")

# ==========================================
# Ausführung des Skripts
# ==========================================
if __name__ == "__main__":
    
    # 1. Argument-Parser einrichten
    parser = argparse.ArgumentParser(
        description="Führt Accelerometer- und Gyroscope-Daten zeitlich zusammen."
    )
    
    # 2. Notwendige Argumente definieren (Pflichtfelder)
    parser.add_argument("acc_file", help="Pfad zur Accelerometer-CSV-Datei (z. B. Accelerometer.csv)")
    parser.add_argument("gyro_file", help="Pfad zur Gyroscope-CSV-Datei (z. B. Gyroscope.csv)")
    parser.add_argument("output_file", help="Pfad zur Zieldatei (z. B. formatierte_zieldaten.csv)")
    
    # 3. Optionales Argument definieren (mit Standardwert)
    parser.add_argument("--activity", default="unbekannt", help="Name der Aktivität (z. B. 'gehen', 'sitzen')")
    
    # 4. Eingaben auslesen
    args = parser.parse_args()
    
    # 5. Funktion mit den übergebenen Argumenten starten
    transform_sensor_data(
        acc_file=args.acc_file, 
        gyro_file=args.gyro_file, 
        output_file=args.output_file,
        activity_label=args.activity 
    )