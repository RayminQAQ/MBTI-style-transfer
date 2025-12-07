import pandas as pd
import re

# 讀取 dataset
df = pd.read_csv("mbti_1.csv")

url_pattern = r"(http://|https://)\S+"

total_posts = 0

short_posts = 0
posts_with_url = 0

short_examples = []
url_examples = []

for _, row in df.iterrows():
    posts = row["posts"].split("|||")

    for p in posts:
        text = p.strip()
        total_posts += 1

        # -----------------------------
        # 1) 字串太短 (< 10 字)
        # -----------------------------
        if len(text) < 15:
            short_posts += 1
            if len(short_examples) < 5:
                short_examples.append(text)

        # -----------------------------
        # 2) 含 URL
        # -----------------------------
        if re.findall(url_pattern, text):
            posts_with_url += 1
            if len(url_examples) < 5:
                url_examples.append(text)


# 統計結果
short_ratio = short_posts / total_posts * 100
url_ratio = posts_with_url / total_posts * 100

print("📌 Dataset 統計結果")
print(f"➡️ 貼文總數：{total_posts}")
print(f"➡️ 字串 < 15 的貼文數量：{short_posts}")
print(f"➡️ 字串 < 15 的占比：{short_ratio:.2f}%")
print(f"➡️ 含 URL 的貼文數量：{posts_with_url}")
print(f"➡️ 含 URL 的占比：{url_ratio:.2f}%")

print("\n🔍 字串太短 (<15) 的貼文（前 5 筆）：")
for s in short_examples:
    print("-", repr(s))

print("\n🔗 含 URL 的貼文（前 5 筆）：")
for s in url_examples:
    print("-", repr(s))
