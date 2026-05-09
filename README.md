# PRODIGY_DS_04 – Sentiment Analysis

> **Prodigy InfoTech – Data Science Internship | Task 04**

## 📌 Task Description
Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands.

---

## 📂 Repository Structure
```
PRODIGY_DS_04/
│
├── sentiment_analysis.py           # Main sentiment analysis script
├── fig1_sentiment_distribution.png # Sentiment count & share
├── fig2_sentiment_by_topic.png     # Sentiment breakdown per brand/topic
├── fig3_polarity_subjectivity.png  # Polarity KDE & scatter plot
├── fig4_wordclouds.png             # Word clouds per sentiment
├── fig5_top_words.png              # Top 10 words per sentiment
└── README.md
```

---

## 📊 Dataset
- **Source:** Twitter Entity Sentiment Analysis Dataset
- **Topics covered:** Apple, Google, Microsoft, Twitter, Amazon
- **Labels:** Positive, Negative, Neutral, Irrelevant

---

## 🔧 Libraries Used
- `pandas` & `numpy` – data manipulation
- `matplotlib` & `seaborn` – visualizations
- `textblob` – sentiment polarity & subjectivity scoring
- `wordcloud` – word cloud generation
- `re` – text cleaning with regex

---

## 🧹 Data Cleaning Steps
- Removed rows with missing text or sentiment labels
- Dropped duplicate entries
- Cleaned text: removed URLs, @mentions, #hashtags & special characters
- Mapped 4 sentiment labels → 3 classes (Positive / Negative / Neutral)

---

## 🤖 Sentiment Scoring (TextBlob)
| Score | Range | Meaning |
|-------|-------|---------|
| Polarity | -1.0 to +1.0 | Negative → Positive |
| Subjectivity | 0.0 to 1.0 | Objective → Subjective |

---

## 📈 Visualizations
| Figure | Description |
|--------|-------------|
| Fig 1 | Overall sentiment distribution (bar + pie) |
| Fig 2 | Sentiment breakdown per topic (stacked bar) |
| Fig 3 | Polarity KDE curves & polarity vs subjectivity scatter |
| Fig 4 | Word clouds for Positive, Negative & Neutral posts |
| Fig 5 | Top 10 most frequent words per sentiment |

---

## 💡 Key Insights
- Positive sentiment dominated across most topics
- **Apple & Google** had the highest positive sentiment ratios
- Negative posts had strongly negative polarity scores on average
- High subjectivity was linked to both strongly positive and negative posts
- Words like *"love"*, *"great"* dominated positive posts; *"worst"*, *"terrible"* dominated negative ones

---

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn textblob wordcloud
python -m textblob.download_corpora
python sentiment_analysis.py
```

---

## 🔗 Connect
**Intern:** Abdul Saif
**Internship:** Prodigy InfoTech – Data Science Track
**Track Code:** DS | **Task:** 04

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com)
