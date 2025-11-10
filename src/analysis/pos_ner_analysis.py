import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import spacy

# === NLTK Downloads ===
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# === Load spaCy model ===
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If not installed, install and load
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# === Load Cleaned Data ===
print("Loading cleaned reviews...")
df = pd.read_csv("data/processed/cleaned_reviews.csv")

# Handle possible different column names
text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
df = df.dropna(subset=[text_col])
reviews = df[text_col].tolist()

# === POS Tagging ===
print("Performing POS tagging...")
pos_counts = Counter()
all_pos_tags = []

for text in reviews:
    if not isinstance(text, str) or not text.strip():
        continue
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    all_pos_tags.extend(tags)
    for _, tag in tags:
        pos_counts[tag] += 1

# === Display Top POS Tags ===
print("\nTop 10 POS Tags:")
for tag, count in pos_counts.most_common(10):
    print(f"{tag}: {count}")

# === Extract adjectives and verbs for analysis ===
adjectives = [word for word, tag in all_pos_tags if tag.startswith('JJ')]
verbs = [word for word, tag in all_pos_tags if tag.startswith('VB')]

print(f"\nCommon adjectives describing product: {Counter(adjectives).most_common(10)}")
print(f"Common verbs used: {Counter(verbs).most_common(10)}")

# === NER (Named Entity Recognition) using spaCy ===
print("\nPerforming Named Entity Recognition with spaCy...")
entities = []

for text in reviews[:200]:  # sample 200 to save time
    if not isinstance(text, str) or not text.strip():
        continue
    doc = nlp(text)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

# === Show Top Named Entities ===
entity_counts = Counter([label for _, label in entities])
print("\nEntity Type Counts:", entity_counts)

print("\nSample Entities Found:")
for entity, label in entities[:15]:
    print(f"{entity} ({label})")

# === Visualization ===
plt.figure(figsize=(8, 4))
plt.bar([x for x, _ in pos_counts.most_common(10)],
        [y for _, y in pos_counts.most_common(10)])
plt.title("Top 10 POS Tags in Reviews")
plt.xlabel("POS Tag")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === Save POS counts to CSV ===
pos_df = pd.DataFrame(pos_counts.items(), columns=["POS_Tag", "Count"])
pos_df.to_csv("data/processed/pos_counts.csv", index=False)

# === Save Named Entities to CSV ===
ner_df = pd.DataFrame(entities, columns=["Entity", "Type"])
ner_df.to_csv("data/processed/ner_entities.csv", index=False)
