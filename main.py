import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

# ---------- (a) Read text and split by paragraphs ----------
text_path = Path("wizard_of_oz.txt")
raw = text_path.read_text(encoding="utf-8", errors="ignore")

# Remove beginning/end Gutenberg copyright pages, keep main content (simple rules)
start_markers = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
]
end_markers = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
]
lower = raw.upper()
start_idx = 0
end_idx = len(raw)
for m in start_markers:
    i = lower.find(m)
    if i != -1:
        # Skip this line
        start_idx = lower.find("\n", i)
        if start_idx == -1: start_idx = i
        break
for m in end_markers:
    i = lower.find(m)
    if i != -1:
        end_idx = i
        break
content = raw[start_idx:end_idx]

# Split by empty lines using given regex (may produce empty first/last paragraphs, filtered later)
paragraphs = re.split(r'\n[ \n]*\n', content)
paragraphs = [p.strip() for p in paragraphs if p.strip()]

print(f"[INFO] Number of paragraphs: {len(paragraphs)}")

# ---------- (c) Build TF-IDF ----------
# Notes:
# - Using scikit-learn's TfidfVectorizer, smooth_idf=True (Smoother log IDF)
# - stop_words='english' removes English stop words
# - norm='l2' normalizes final vector length; consistent with "length-normalized TF" goal
# - You can change ngram_range to (1,2) for words + bigrams (unigram is sufficient for this problem)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    smooth_idf=True,
    norm='l2',
    use_idf=True,
    sublinear_tf=False,
    max_df=0.9,       # Remove overly common words
    min_df=2          # Remove words appearing only once
)
X = vectorizer.fit_transform(paragraphs)   # (n_paras, n_terms)
terms = np.array(vectorizer.get_feature_names_out())
print(f"[INFO] Vocabulary size: {len(terms)}")

# ---------- (d) Train 10-component GMM ----------
n_components = 10
gmm = GaussianMixture(reg_covar=1e-2, 
    n_components=n_components,
    covariance_type='diag',
    random_state=42,
    max_iter=500
)
# GMM requires dense array
X_dense = X.toarray()
gmm.fit(X_dense)
print("[INFO] GMM training completed.")

# ---------- (e) Each component: Top-10 words with largest absolute values in mean vector ----------
Path("results").mkdir(exist_ok=True)
top_words_report = []

means = gmm.means_  # (K, n_terms)
for k in range(n_components):
    mu = means[k]
    # Get 10 dimensions with largest absolute values
    top_idx = np.argsort(np.abs(mu))[-10:][::-1]
    top_terms = terms[top_idx]
    top_vals = mu[top_idx]
    line = f"Component {k:02d}: " + ", ".join(f"{t} ({v:+.4f})" for t, v in zip(top_terms, top_vals))
    top_words_report.append(line)

(Path("results") / "top_words_by_component.txt").write_text(
    "\n".join(top_words_report), encoding="utf-8"
)
print("[INFO] Written results/top_words_by_component.txt")

# ---------- (f) Each component: Paragraph with maximum posterior probability ----------
# Calculate membership probability of each paragraph to each component
resp = gmm.predict_proba(X_dense)  # (n_paras, K)
winners = resp.argmax(axis=0)      # Paragraph index with max probability for each component (note: max per column)
# The above line is incorrect because argmax(axis=0) returns index per column, but columns are paragraphs; we need the paragraph with max probability for each component across all paragraphs:
winners = resp[:, :].argmax(axis=0)  # Ensure clear semantics
# More robust approach:
winners = resp[:, :].argmax(axis=0)

# Actually the above two lines are equivalent, we'll use this clear version:
winners = np.argmax(resp, axis=0)

top_paras_lines = []
for k in range(n_components):
    idx = winners[k]
    prob = resp[idx, k]
    # Output first 200 characters of the paragraph for readability
    preview = paragraphs[idx].replace("\n", " ")
    preview = re.sub(r"\s+", " ", preview)
    preview = preview[:200] + ("..." if len(preview) > 200 else "")
    top_paras_lines.append(f"Component {k:02d}  Paragraph#{idx}  p={prob:.4f}\n{preview}\n")

(Path("results") / "top_paragraphs_by_component.txt").write_text(
    "\n".join(top_paras_lines), encoding="utf-8"
)
print("[INFO] Written results/top_paragraphs_by_component.txt")

# Save entire membership matrix to CSV for own analysis 
import pandas as pd
df = pd.DataFrame(resp, columns=[f"comp_{i:02d}" for i in range(n_components)])
df.to_csv("results/memberships.csv", index_label="paragraph_id")
print("[INFO] Written results/memberships.csv")

print("\n=== Summary ===")
print(f"Paragraphs: {len(paragraphs)} | Vocabulary: {len(terms)} | Components: {n_components}")
print("Check: results/top_words_by_component.txt and results/top_paragraphs_by_component.txt")