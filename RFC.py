import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

df = pd.read_csv("mfcc_dataset.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y , test_size=0.2, random_state=42)

label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse"}

def class_counts(labels):
    counts = Counter(labels)
    msg = "-> "
    for k in sorted(counts):
        msg += f"{label_map.get(k, 'Unknown')}: {counts[k]}  "
    return msg.strip()

print(f"Loaded: {len(X_scaled)} || " + class_counts(y))
print(f"X_train: {len(X_train)}, y_train: {len(y_train)} || " + class_counts(y_train))
print(f"X_test: {len(X_test)}, y_test: {len(y_test)} || " + class_counts(y_test))

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Whistles", "Clicks", "Burst-Pulses"]))