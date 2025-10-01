# Paragraph Clustering for *The Wonderful Wizard of Oz*

This repository contains a complete, reproducible pipeline to cluster **paragraphs** from L. Frank Baum’s *The Wonderful Wizard of Oz* using a **TF–IDF vector space** and a **10‑component Gaussian Mixture Model (GMM)**.

> Goal: split the book into paragraphs, build a length‑normalized TF + smoothed‑IDF representation, fit a GMM, and report (e) top words per component and (f) the most representative paragraph per component — then briefly discuss results.

---

## Contents

```
.
├── main.py
├── wizard_of_oz.txt            # Gutenberg text (downloaded by you)
├── results/
│   ├── top_words_by_component.txt
│   ├── top_paragraphs_by_component.txt
│   └── memberships.csv         # posterior probabilities (paragraph × component)
├── .gitignore
├── requirements.txt            # optional, pinned by `pip freeze`
└── README.md
```

---

## Quick Start

```bash
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install numpy scipy scikit-learn pandas

# 3) Download the book (UTF-8)
curl -L -o wizard_of_oz.txt https://www.gutenberg.org/files/55/55-0.txt

# 4) Run the pipeline
python main.py
```

**Expected console summary** (your numbers may vary slightly by preprocessing choices):

```
[INFO] Number of paragraphs: 1140
[INFO] Vocabulary size: 1489
[INFO] GMM training completed.
[INFO] Written results/top_words_by_component.txt
[INFO] Written results/top_paragraphs_by_component.txt
[INFO] Written results/memberships.csv

=== Summary ===
Paragraphs: 1140 | Vocabulary: 1489 | Components: 10
```

---

## Method

1. **Paragraph splitting** — using the regex `\n[ \n]*\n` to split on blank lines, then trimming/removing empty pieces.  
2. **Vectorization** — `TfidfVectorizer(lowercase=True, stop_words='english', norm='l2', smooth_idf=True, max_df=0.9, min_df=2)`.  
   - *Length-normalized TF* via `norm='l2'` (row‑wise L2 norm).  
   - *Smoothed logarithmic IDF* via `smooth_idf=True`.  
3. **Clustering** — 10‑component **GaussianMixture**, `covariance_type='diag'`, with a small `reg_covar` to stabilize estimation.  
4. **Reporting** —  
   - (e) For each component: top 10 words with largest absolute values in the **mean** vector.  
   - (f) For each component: the **paragraph** with the **highest posterior probability** (membership) for that component.  
   - (optional) Entire posterior matrix saved to `results/memberships.csv` for analysis.

---

## Results (samples)

### (e) Top words per component (first 5 of 10)
*(from `results/top_words_by_component.txt`)*
```
Component 00: queen (+0.2631), mice (+0.2526), little (+0.0740), truck (+0.0556), saving (+0.0535), ...
Component 01: woodman (+0.2383), tin (+0.2006), heart (+0.1259), said (+0.0769), shall (+0.0344), ...
Component 02: city (+0.2640), emerald (+0.2218), oz (+0.0856), dorothy (+0.0554), said (+0.0520), ...
Component 03: dorothy (+0.0593), said (+0.0390), little (+0.0270), answered (+0.0226), kansas (+0.0219), ...
Component 04: asked (+0.4781), dorothy (+0.1221), girl (+0.0531), scarecrow (+0.0514), shall (+0.0428), ...
```

These are highly interpretable: field‑mice queen, Tin Woodman, Emerald City, Dorothy/Kansas/Toto, Q&A/dialogue… Other components (not shown above) clearly map to **Scarecrow**, **Wicked Witch (East/West, silver shoes)**, **Cowardly Lion/courage**, etc.

### (f) Representative paragraph per component (sample)
*(from `results/top_paragraphs_by_component.txt`)*
```
Component 00  Paragraph#388  p=1.0000
Chapter IX The Queen of the Field Mice

Component 01  Paragraph#188  p=1.0000
Chapter V The Rescue of the Tin Woodman

Component 02  Paragraph#64  p=1.0000
“Oz himself is the Great Wizard,” ... “He lives in the City of Emeralds.”
```
*(Files include representatives for all 10 components.)*

### Posterior memberships
The full **paragraph × component** posterior matrix is stored in `results/memberships.csv`.  
Each row sums to ~1. Example rows (rounded):

```
paragraph_0: c09=0.9514, c06=0.0471, c01=0.0004, ...
paragraph_7: c09=0.9408, c01=0.0107, c06=0.0466, ...
paragraph_26: c09=0.7533, c03=0.1910, c06=0.0082, ...
```

---

## Reproduce / Modify

- Adjust n‑grams for more topical phrases:
  ```python
  TfidfVectorizer(..., ngram_range=(1,2))
  ```
- Tune mixture size:
  ```python
  GaussianMixture(n_components=8)  # or 12, etc.
  ```
- Soften responsibilities (more “soft” clusters): increase `reg_covar` or try `BayesianGaussianMixture`.

---

## Notes

- We keep **results/** in version control so reviewers can see outputs without re‑running.  
- `.venv/`, `__pycache__/`, and editor/system files are ignored via `.gitignore`.  
- Text source: Project Gutenberg (book #55). See the Wikipedia entry for background reading.

---

## Citation

- L. Frank Baum, *The Wonderful Wizard of Oz*, Project Gutenberg, eBook #55.  
- scikit‑learn: Pedregosa et al., JMLR 12, 2011.

---
