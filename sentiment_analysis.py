# ============================================================
#  PRODIGY INFOTECH – Data Science Internship
#  Task 04: Sentiment Analysis – Social Media Data
#  Author : Abdul Saif
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# Sentiment analysis
from textblob import TextBlob

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.autolayout": True})

# ============================================================
# 1.  LOAD DATA
# ============================================================
print("=" * 55)
print("  SENTIMENT ANALYSIS – SOCIAL MEDIA DATA")
print("=" * 55)

# Using the Twitter Entity Sentiment dataset from the sample link
# Falls back to a reliable Kaggle-mirrored CSV via URL
url = ("https://raw.githubusercontent.com/nicholasgasior/twitter-"
       "sentiment-analysis/main/data/twitter_training.csv")

try:
    df = pd.read_csv(url, header=None,
                     names=["id", "topic", "sentiment", "text"])
    print(f"\n── Loaded dataset from URL. Shape: {df.shape}")
except Exception:
    # Fallback: build a representative sample dataset
    print("\n── URL unavailable. Using built-in sample dataset.")
    import random, string
    random.seed(42)
    topics    = ["Apple", "Google", "Microsoft", "Twitter", "Amazon"]
    sentiments = ["Positive", "Negative", "Neutral", "Irrelevant"]
    samples = {
        "Positive": [
            "I love this product so much!", "Amazing experience overall!",
            "Best service I have ever used.", "Really happy with my purchase.",
            "Totally recommend this to everyone!", "Fantastic quality and fast delivery.",
            "Great customer support team.", "Exceeded all my expectations!",
            "Super impressed with the new update.", "Five stars, absolutely wonderful."
        ],
        "Negative": [
            "This is the worst product ever.", "Terrible customer service.",
            "I want a refund immediately.", "Very disappointed with the quality.",
            "Do not buy this, complete waste of money.", "Broken on arrival, very frustrating.",
            "Support team was rude and unhelpful.", "Never buying from here again.",
            "Product stopped working after one day.", "Awful experience from start to finish."
        ],
        "Neutral": [
            "The product arrived today.", "I used the app this morning.",
            "The update was released yesterday.", "Package delivered as scheduled.",
            "I have been using this for a week.", "The price is similar to competitors.",
            "Got an email confirmation.", "The website is currently under maintenance.",
            "New features were announced today.", "The store opens at 9am."
        ],
        "Irrelevant": [
            "The weather is nice today.", "I had pizza for lunch.",
            "Just watched a great movie.", "Going for a walk later.",
            "My cat knocked over my coffee.", "Can't believe it's already Friday.",
            "The traffic was bad this morning.", "Reading a good book right now.",
            "Planning a vacation next month.", "Listening to music while working."
        ]
    }
    rows = []
    for i in range(800):
        t  = random.choice(topics)
        s  = random.choice(sentiments)
        tx = random.choice(samples[s])
        rows.append([i, t, s, tx])
    df = pd.DataFrame(rows, columns=["id", "topic", "sentiment", "text"])

print(f"\n── Columns : {list(df.columns)}")
print(f"── Shape   : {df.shape}")
print(f"\n── Sentiment distribution:\n{df['sentiment'].value_counts()}")

# ============================================================
# 2.  DATA CLEANING
# ============================================================
# Drop rows with missing text or sentiment
df.dropna(subset=["text", "sentiment"], inplace=True)
df["text"] = df["text"].astype(str)

# Remove duplicates
df.drop_duplicates(subset="text", inplace=True)

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", "", txt)      # URLs
    txt = re.sub(r"@\w+", "", txt)                 # mentions
    txt = re.sub(r"#\w+", "", txt)                 # hashtags
    txt = re.sub(r"[^a-z\s]", "", txt)             # special chars
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

df["clean_text"] = df["text"].apply(clean_text)

# ── TextBlob sentiment scores ────────────────────────────────
df["polarity"]    = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"]= df["clean_text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Map labels to 3 classes for cleaner analysis
sentiment_map = {
    "Positive": "Positive", "Negative": "Negative",
    "Neutral": "Neutral",   "Irrelevant": "Neutral"
}
df["sentiment_3"] = df["sentiment"].map(sentiment_map).fillna("Neutral")

print(f"\n── After cleaning, shape: {df.shape}")
print(f"\n── Polarity stats:\n{df['polarity'].describe().round(3)}")

# ============================================================
# 3.  VISUALISATIONS
# ============================================================
COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}

# ── Fig 1 · Sentiment Distribution ──────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Sentiment Distribution Overview",
              fontsize=14, fontweight="bold")

# Bar chart
sc = df["sentiment_3"].value_counts()
axes[0].bar(sc.index, sc.values,
            color=[COLORS[s] for s in sc.index],
            edgecolor="white", width=0.5)
axes[0].set_title("Sentiment Count")
axes[0].set_ylabel("Number of Posts")
for bar, val in zip(axes[0].patches, sc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5, str(val),
                 ha="center", fontweight="bold")

# Pie chart
axes[1].pie(sc.values, labels=sc.index,
            autopct="%1.1f%%", startangle=90,
            colors=[COLORS[s] for s in sc.index])
axes[1].set_title("Sentiment Share")

plt.savefig("fig1_sentiment_distribution.png")
plt.show()
print("✔  Saved fig1_sentiment_distribution.png")

# ── Fig 2 · Sentiment by Topic ───────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6))
topic_sent = (df.groupby(["topic", "sentiment_3"])
                .size().unstack(fill_value=0))
topic_sent_pct = topic_sent.div(topic_sent.sum(axis=1), axis=0)
topic_sent_pct[["Positive", "Neutral", "Negative"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=["#2ecc71", "#3498db", "#e74c3c"],
    edgecolor="white", width=0.6)
ax.set_title("Sentiment Distribution by Topic",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Topic")
ax.set_ylabel("Proportion")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.savefig("fig2_sentiment_by_topic.png", bbox_inches="tight")
plt.show()
print("✔  Saved fig2_sentiment_by_topic.png")

# ── Fig 3 · Polarity Distribution ───────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Polarity & Subjectivity Analysis",
              fontsize=14, fontweight="bold")

for sentiment, color in COLORS.items():
    subset = df[df["sentiment_3"] == sentiment]["polarity"]
    if len(subset) > 1:
        subset.plot.kde(ax=axes[0], label=sentiment,
                        color=color, linewidth=2)
axes[0].set_title("Polarity Distribution by Sentiment")
axes[0].set_xlabel("Polarity Score (-1 to +1)")
axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[0].legend()

axes[1].scatter(df["polarity"], df["subjectivity"],
                c=df["sentiment_3"].map(COLORS),
                alpha=0.4, edgecolors="none", s=15)
axes[1].set_title("Polarity vs Subjectivity")
axes[1].set_xlabel("Polarity")
axes[1].set_ylabel("Subjectivity")
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=s) for s, c in COLORS.items()]
axes[1].legend(handles=legend_els)

plt.savefig("fig3_polarity_subjectivity.png")
plt.show()
print("✔  Saved fig3_polarity_subjectivity.png")

# ── Fig 4 · Word Clouds ──────────────────────────────────────
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle("Word Clouds by Sentiment",
              fontsize=14, fontweight="bold")

wc_colors = {"Positive": "Greens", "Negative": "Reds", "Neutral": "Blues"}
for ax, sentiment in zip(axes, ["Positive", "Negative", "Neutral"]):
    words = " ".join(df[df["sentiment_3"] == sentiment]["clean_text"])
    if words.strip():
        wc = WordCloud(width=400, height=300, background_color="white",
                       colormap=wc_colors[sentiment],
                       max_words=80).generate(words)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(sentiment, fontsize=12, fontweight="bold",
                 color=COLORS[sentiment])

plt.savefig("fig4_wordclouds.png", bbox_inches="tight")
plt.show()
print("✔  Saved fig4_wordclouds.png")

# ── Fig 5 · Top Words per Sentiment ─────────────────────────
fig5, axes = plt.subplots(1, 3, figsize=(15, 5))
fig5.suptitle("Top 10 Words per Sentiment",
              fontsize=14, fontweight="bold")

STOPWORDS = {"the","a","an","is","it","in","of","and","to",
             "this","for","i","my","was","be","with","that","on","at","are"}

for ax, sentiment in zip(axes, ["Positive", "Negative", "Neutral"]):
    words = " ".join(df[df["sentiment_3"] == sentiment]["clean_text"]).split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    top = Counter(words).most_common(10)
    if top:
        labels, counts = zip(*top)
        ax.barh(list(labels)[::-1], list(counts)[::-1],
                color=COLORS[sentiment], edgecolor="white")
        ax.set_title(f"{sentiment}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")

plt.savefig("fig5_top_words.png", bbox_inches="tight")
plt.show()
print("✔  Saved fig5_top_words.png")

# ============================================================
# 4.  KEY INSIGHTS
# ============================================================
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)

total = len(df)
for s in ["Positive", "Negative", "Neutral"]:
    pct = (df["sentiment_3"] == s).sum() / total
    print(f"• {s:10s} posts : {pct:.1%}")

print(f"\n• Avg polarity  (Positive) : "
      f"{df[df['sentiment_3']=='Positive']['polarity'].mean():.3f}")
print(f"• Avg polarity  (Negative) : "
      f"{df[df['sentiment_3']=='Negative']['polarity'].mean():.3f}")
print(f"• Avg subjectivity         : {df['subjectivity'].mean():.3f}")

if "topic" in df.columns:
    best  = df.groupby("topic")["polarity"].mean().idxmax()
    worst = df.groupby("topic")["polarity"].mean().idxmin()
    print(f"\n• Most positive topic  : {best}")
    print(f"• Most negative topic  : {worst}")

print("\n── Analysis complete. All figures saved as PNG files.")
