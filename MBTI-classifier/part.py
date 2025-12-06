import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv("mbti_1.csv")

# -----------------------------
# 1️⃣ MBTI 16 類型分布
# -----------------------------
type_counts = df["type"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 10))
plt.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("MBTI Type Distribution (16 classes)")
plt.tight_layout()
plt.savefig("mbti_16types_distribution.png")
plt.show()

print("\n=== 16 types distribution ===")
print(type_counts)
print("-------------------------------\n")


# -----------------------------
# 2️⃣ 四大維度分解 (IE / NS / FT / PJ)
# -----------------------------
df["IE"] = df["type"].apply(lambda x: x[0])
df["NS"] = df["type"].apply(lambda x: x[1])
df["FT"] = df["type"].apply(lambda x: x[2])
df["PJ"] = df["type"].apply(lambda x: x[3])

dimensions = ["IE", "NS", "FT", "PJ"]

for dim in dimensions:
    counts = df[dim].value_counts().sort_index()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"{dim} Distribution")
    plt.tight_layout()
    plt.savefig(f"{dim}_distribution.png")
    plt.show()

    print(f"\n=== {dim} counts ===")
    print(counts)
    print("------------------------\n")
