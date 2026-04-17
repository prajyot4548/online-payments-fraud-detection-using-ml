import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("🚀 Loading dataset...")

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# ✅ USE ALL TYPES (IMPORTANT FIX)
print("Transaction types:", df["type"].unique())

# ⚡ Reduce size for speed
df = df.sample(n=200000, random_state=42)

# =========================
# 🔥 FEATURE ENGINEERING
# =========================

def create_features(df):
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["errorOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["errorDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    return df

df = create_features(df)

# Drop useless
df = df.drop(["nameOrig", "nameDest"], axis=1)

# =========================
# 🔐 ENCODING (IMPORTANT)
# =========================

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Save encoder classes for debug
print("Encoded types:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

FEATURE_COLUMNS = df.drop("isFraud", axis=1).columns.tolist()

X = df[FEATURE_COLUMNS]
y = df["isFraud"]

# =========================
# ✅ SPLIT FIRST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# ⚠️ SMOTE ONLY TRAIN
# =========================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# ✅ SCALING
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🤖 MODEL (TUNED)
# =========================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("🤖 Training...")

model.fit(X_train, y_train)

# =========================
# 📊 EVALUATION
# =========================

y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 💾 SAVE EVERYTHING
# =========================

pickle.dump(model, open("model/fraud_model.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(FEATURE_COLUMNS, open("model/columns.pkl", "wb"))

print("\n✅ Model Saved Successfully!")