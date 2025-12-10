import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

sns.set(style="whitegrid", context="notebook")
data = [
    ("Song A", "Artist1", 120, 0.8, 0.9, 0.7, "energetic"),
    ("Song B", "Artist2", 70, 0.2, 0.1, 0.3, "sad"),
    ("Song C", "Artist3", 90, 0.5, 0.6, 0.5, "relaxed"),
    ("Song D", "Artist4", 130, 0.9, 0.8, 0.8, "energetic"),
    ("Song E", "Artist5", 60, 0.3, 0.2, 0.4, "sad"),
    ("Song F", "Artist6", 100, 0.6, 0.7, 0.6, "happy"),
    ("Song G", "Artist7", 95, 0.55, 0.65, 0.55, "relaxed"),
    ("Song H", "Artist8", 110, 0.75, 0.85, 0.68, "happy"),
]

df = pd.DataFrame(data, columns=[
    "title", "artist", "tempo", "energy", "valence", "danceability", "mood"
])
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="mood", hue="mood", palette="viridis", legend=False)
plt.title("Mood Distribution in Dataset")
plt.tight_layout()
plt.show()
df[["tempo", "energy", "valence", "danceability"]].hist(
    figsize=(10,6), edgecolor="black"
)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.figure(figsize=(6,5))
corr = df[["tempo", "energy", "valence", "danceability"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,5))
mean_feats = df.groupby("mood")[["tempo", "energy", "valence", "danceability"]].mean()
mean_feats.plot(kind="bar", figsize=(8,5))
plt.title("Average Feature Values per Mood")
plt.xlabel("Mood")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
sns.pairplot(df, hue="mood", corner=True)
plt.suptitle("Pairplot – Feature Relationships by Mood", y=1.02)
plt.show()
for feature in ["tempo", "energy", "valence", "danceability"]:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="mood", y=feature, palette="viridis")
    plt.title(f"{feature.capitalize()} Distribution per Mood")
    plt.show()
plt.figure(figsize=(10,6))
sns.violinplot(data=df, x="mood", y="energy", palette="coolwarm")
plt.title("Energy Density per Mood")
plt.show()
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="tempo", y="energy", hue="mood", palette="viridis", s=120)
plt.title("Tempo vs Energy (Mood Clusters)")
plt.show()
plt.figure(figsize=(7,5))
sns.kdeplot(data=df, x="tempo", hue="mood", fill=True, alpha=0.4)
plt.title("Tempo Density per Mood")
plt.show()
features = ["tempo", "energy", "valence", "danceability"]
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
for mood in df["mood"].unique():
    values = df[df["mood"] == mood][features].mean().tolist()
    values += values[:1]

    plt.polar(angles, values, marker='o', label=mood)

plt.title("Radar Chart — Average Features per Mood", size=15)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="artist", hue="mood", palette="viridis")
plt.title("Artist vs Mood (Song Counts)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
X = df[["tempo", "energy", "valence", "danceability"]]
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=120, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
new_song = pd.DataFrame([{
    "tempo": 105,
    "energy": 0.7,
    "valence": 0.6,
    "danceability": 0.65
}])
predicted_mood = pipeline.predict(new_song)[0]
print("Predicted Mood for New Song:", predicted_mood)
def recommend_songs_for_mood(mood, k=5):
    results = df[df["mood"] == mood]
    if results.empty:
        return "No songs found."
    return results.head(k)[["title", "artist"]]

print("\nRecommended Songs for 'happy':")
print(recommend_songs_for_mood("happy"))
