import pandas as pd
import json
import os

CSV_FILE = "prompts.csv"

# Load CSV
df = pd.read_csv(CSV_FILE)

print("Total rows:", len(df))

# Normalize columns
df["final_verdict"] = df["final_verdict"].astype(str).str.strip().str.lower()
df["domain"] = df["domain"].astype(str).str.strip().str.lower()

# Split malicious / benign
malicious_df = df[df["final_verdict"] == "block"]
benign_df = df[df["final_verdict"] == "allow"]

print("Malicious prompts:", len(malicious_df))
print("Benign prompts:", len(benign_df))


def fix_domain(value):
    """Convert 'none' or empty domains to 'general'."""
    if value in ["none", "nan", "", "null"]:
        return "general"
    return value


# Build malicious dataset
malicious = []
for i, row in malicious_df.iterrows():
    malicious.append({
        "id": f"mal-{i:05d}",
        "text": str(row["prompt"]),
        "domain": fix_domain(row["domain"])
    })


# Build benign dataset
benign = []
for i, row in benign_df.iterrows():
    benign.append({
        "id": f"ben-{i:05d}",
        "text": str(row["prompt"]),
        "domain": fix_domain(row["domain"])
    })


# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Save JSON files
with open("data/malicious.json", "w", encoding="utf-8") as f:
    json.dump(malicious, f, indent=2)

with open("data/benign.json", "w", encoding="utf-8") as f:
    json.dump(benign, f, indent=2)

print("\nDataset generation complete.")
print(f"[OK] Wrote {len(malicious)} malicious prompts -> data/malicious.json")
print(f"[OK] Wrote {len(benign)} benign prompts -> data/benign.json")