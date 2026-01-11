import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import joblib

# 1. Load Data
print("🔄 Loading data...")
try:
    df = pd.read_csv('dataset1.csv')
except FileNotFoundError:
    print("❌ Error: dataset1.csv not found.")
    exit()

X_train = []
y_train = []

# 2. Generate Training Data (English, Tagalog, Bisaya)
print("🧠 Generating training questions...")
for index, row in df.iterrows():
    dept = row['Department']
    role = row['Role']
    intent_label = f"{dept}_{role}"

    # General Questions
    questions = [
        f"Who is the {role} of {dept}?", f"Name of {dept} {role}",
        f"Sino ang {role} ng {dept}?", f"Kinsa ang {role} sa {dept}?"
    ]
    
    # Location Questions (Dean only)
    if role == "Dean":
        loc_qs = [f"Where is {dept}?", f"Saan ang {dept}?", f"Asa dapit ang {dept}?"]
        X_train.extend(loc_qs)
        y_train.extend([f"{dept}_Location"] * len(loc_qs))

    # Org Chart Questions
    org_qs = [f"Show me the org chart of {dept}", f"Struktura ng {dept}", f"Org chart sa {dept}"]
    X_train.extend(org_qs)
    y_train.extend([f"{dept}_OrgChart"] * len(org_qs))

    X_train.extend(questions)
    y_train.extend([intent_label] * len(questions))

# 3. Special Aliases (e.g. CS -> CAS)
print("🔧 Adding aliases...")
cs_qs = ["Who is the CS Dean?", "Saan ang CS?", "Asa ang CS dept?"]
X_train.extend(cs_qs)
y_train.extend(['CAS_Dean'] * len(cs_qs))

# 4. Train & Save
print("🏋️ Training AI...")
model = make_pipeline(CountVectorizer(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000))
model.fit(X_train, y_train)
joblib.dump(model, 'university_model.pkl')
print("✅ Done! 'university_model.pkl' saved.")