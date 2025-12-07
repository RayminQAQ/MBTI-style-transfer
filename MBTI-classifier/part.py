import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mbti_1.csv")

# 16 類統計
type_counts = df["type"].value_counts()

# 2.5% 門檻
threshold = 0.025 * len(df)

# 合併小類別
type_counts_adjusted = type_counts.copy()
type_counts_adjusted[type_counts < threshold] = 0

others_sum = type_counts[type_counts < threshold].sum()

# 新的資料（含 Other）
plot_labels = list(type_counts_adjusted[type_counts_adjusted > 0].index) + ["Other"]
plot_sizes = list(type_counts_adjusted[type_counts_adjusted > 0].values) + [others_sum]

plt.figure(figsize=(10, 10))
plt.pie(plot_sizes, labels=plot_labels, autopct="%1.1f%%", pctdistance=0.85)
plt.title("MBTI Type Distribution (Merged Low-Frequency Types)")
plt.tight_layout()
plt.savefig("mbti_16types_merged.png")
plt.show()
