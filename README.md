# Representation-Space Analysis of NER Errors in Kazakh–Russian Public Transport Complaints

This repository contains the training pipeline and a follow-up analysis for the NER model from:

> A. Ussenbayeva, D. Khussainova, D. Otepova, T. Orazov, D. Rakhimzhanov. *Heterogeneous Entity Extraction for Geotagging and Spatio-Temporal Analysis of Public Transport Complaints.* 2026 IEEE 6th International Conference on Smart Information Systems and Technologies (SIST), pp. 1–6. DOI: [10.1109/SIST61674.2026.11596166](https://doi.org/10.1109/SIST61674.2026.11596166)

In the paper, STREET_NAME was the weakest entity type (F1 = 0.28). We attributed this to class scarcity and a train–test distribution shift (about 34 complaints with street names in train vs. 167 in test), and listed language mixing among the difficulties of street names. This follow-up tests these explanations in the encoder's representation space:

1. Does the position of a complaint in representation space predict NER errors beyond simple baselines (length, language, model confidence)?
2. Is the train–test shift for STREET_NAME real, or an artifact of small samples? Is it caused by the data, by fine-tuning, or by language mixing?

**Status: work in progress.** Results come from the original data split; see [Limitations](#limitations) and [Next steps](#next-steps).

## Summary of findings

- **The train–test shift for street names is real.** Measured with the sliced Wasserstein distance (SWD) on entity-level representations, it is larger than under random re-splits of the same data (permutation test) and larger than for stop names at the same sample size (size-matched null).
- **Fine-tuning strongly amplifies it.** In the pre-trained encoder the shift is small (about 1.3× the permutation null at the last layer); after fine-tuning it is about 5×. This is consistent with the model memorizing the 46 training street mentions instead of generalizing.
- **Language mixing is not the main cause.** The language composition of complaints with street names is nearly identical in train and test, and the shift remains significant within Russian-only complaints.
- **Geometry predicts errors only modestly.** Complaints with errors are *not* farther from their language centroid (the correlation has the opposite sign). Geometric features alone predict errors about as well as complaint length, and add a small gain on top of model confidence.

## Contents

| Path | Description |
|---|---|
| `notebooks/ner_representation_analysis.ipynb` | Training, evaluation and analysis (Parts 1–5) |
| `scripts/make_stratified_split.py` | Re-splits the data 70/30 with multi-label iterative stratification |
| `scripts/representation_shift_checks.py` | Statistical controls for the SWD analysis (same code as notebook cell 17) |

## Data

The dataset (2,750 manually annotated complaints in Russian, Kazakh and mixed Kazakh–Russian, six entity types) is **not included** because of data-sharing restrictions. The notebook expects `train.jsonl` and `test.jsonl` with pre-tokenized `input_ids`, `attention_mask` and `labels` (XLM-R tokenizer; label on the first subword of each word, other subwords set to -100).

## How to run

1. Open the notebook in Google Colab with a GPU runtime (a T4 is enough).
2. Upload `train.jsonl` and `test.jsonl` (cell 4 finds them in `/content`, `/` or `/content/drive/MyDrive/ner_data`).
3. Optional: create a re-stratified split and set `TRAIN_FILE` / `TEST_FILE` in cell 2:
   ```bash
   python scripts/make_stratified_split.py --train train.jsonl --test test.jsonl \
       --out_train train_strat.jsonl --out_test test_strat.jsonl
   ```
4. Run the cells in order. Outputs can be kept in the committed notebook, but make sure none of them shows complaint texts (the notebook as provided only prints counts, metrics and plots).

## Setup

- Model: `xlm-roberta-base`, token classification with BIO tags.
- AdamW (lr 2e-5, weight decay 0.01), linear schedule without warm-up, batch size 8, 9 epochs (the paper used 10).
- Class-weighted cross-entropy (inverse frequency; STREET_NAME weight × 1.6).
- Entity-level exact-match evaluation with `seqeval`.

## Results (original split)

### 1. NER performance

Per-type F1 on the test set: the latest run and the range over three runs of the same code (GPU training is not fully deterministic).

| Entity type | Precision | Recall | F1 (latest run) | F1 range, 3 runs |
|---|---|---|---|---|
| ROUTE_NUM | 0.874 | 0.961 | 0.916 | 0.916–0.922 |
| BUS_ID | 0.899 | 0.639 | 0.747 | 0.735–0.751 |
| PLATE_NUM | 0.534 | 0.854 | 0.657 | 0.643–0.657 |
| STOP_NAME | 0.526 | 0.830 | 0.644 | 0.633–0.644 |
| STREET_NAME | 0.316 | 0.207 | 0.250 | 0.218–0.294 |
| STREET_INTERSECTION | 0 | 0 | 0 | 0 (2 train / 6 test) |

Micro-F1 = 0.727. STOP_NAME has high recall but low precision, consistent with street names being tagged as stops (the paper reports 50 such confusions).

### 2. Representation geometry vs. errors

Each complaint is represented by its mean-pooled hidden states. 59.5% of test complaints contain at least one error.

**Distance to language centroid.** Hypothesis: erroneous complaints lie farther from the centroid of their language (computed on train). Spearman correlation between distance and complaint-level F1:

| Layer | 0 | 4 | 8 | 12 |
|---|---|---|---|---|
| ρ | +0.113 | +0.171 | +0.196 | +0.087 |

The sign is positive at every layer, so the hypothesis is **not supported**: erroneous complaints are slightly *closer* to the centroid. A likely explanation is a length confound: complaints with many entities are more likely to contain at least one error, and they are also more typical.

**Predicting errors** (logistic regression, 5-fold cross-validation, ROC-AUC):

| Features | AUC |
|---|---|
| length + number of entities | 0.683 |
| + language | 0.695 |
| + model confidence | 0.801 |
| geometry only | 0.693 |
| geometry + all baselines | 0.829 |

Geometry alone is no better than length, but adds +0.028 AUC on top of model confidence. Geometry-only AUC within languages: ru 0.705 (n = 576), mixed 0.665 (n = 167), kk 0.805 (n = 82, too small to be reliable).

### 3. Train–test shift (complaint level)

SWD between train and test embeddings at layer 12:

| Comparison | n (train / test) | SWD |
|---|---|---|
| complaints with STOP_NAME | 1616 / 462 | 0.042 |
| complaints with STREET_NAME | 34 / 167 | 0.137 |
| *reference:* Russian complaints, train vs. test | | 0.043 |
| *reference:* Russian vs. Kazakh complaints (test) | | 0.072 |

The shift for street-name complaints is about three times the shift for stop names. Because SWD is inflated at small sample sizes, this comparison alone is not conclusive; see the controls below.

### 4. Statistical controls (entity level)

Each street or stop mention is represented by the mean of its subword hidden states (46 street mentions in train, 208 in test). For each encoder and layer: the observed street SWD, its mean under 500 random re-splits of the pooled mentions (permutation null), and the 95th percentile of stop-name SWD on 200 subsamples of the same size as the street set (size-matched null).

| Encoder | Layer | Street SWD | Permutation null | p | Stop, size-matched 95% |
|---|---|---|---|---|---|
| fine-tuned | 0 | 0.046 | 0.040 | 0.010 | 0.042 |
| fine-tuned | 4 | 0.114 | 0.089 | < 0.002 | 0.095 |
| fine-tuned | 8 | 0.282 | 0.096 | < 0.002 | 0.115 |
| fine-tuned | 12 | 0.669 | 0.132 | < 0.002 | 0.203 |
| frozen | 0 | 0.046 | 0.040 | 0.012 | 0.041 |
| frozen | 4 | 0.110 | 0.088 | 0.004 | 0.092 |
| frozen | 8 | 0.113 | 0.091 | < 0.002 | 0.098 |
| frozen | 12 | 0.026 | 0.021 | < 0.002 | 0.022 |

(p < 0.002 is the smallest value resolvable with 500 permutations. SWD values are comparable within a layer, not across layers.)

- The street shift exceeds both nulls for both encoders at every layer, so it is not a small-sample artifact.
- In the frozen encoder it is modest (about 1.3× the permutation null at layer 12); after fine-tuning it grows to about 5×. Layer 0 is identical for both encoders, as expected, since fine-tuning barely changes the token embeddings.
- The shift is already present at layer 0, which reflects mostly lexical content. This suggests that the test set contains different street names than the training set (to be verified; see Next steps).

**Language.** Language composition of complaints with street names: train ru 71% / mixed 21% / kk 9% (n = 34); test ru 73% / mixed 22% / kk 5% (n = 167). Within Russian-only complaints the shift remains significant (layer 12: SWD = 0.138 vs. permutation null 0.058, p < 0.002).

### Relation to the paper

The paper explained the low STREET_NAME score by class scarcity and a train–test shift. This analysis **confirms the shift statistically**, shows that **fine-tuning amplifies it**, and finds **no support for language mixing** as its main cause.

## Limitations

- **Uneven split.** Although the published split was meant to be stratified, only 34 complaints with street names are in train and 167 in test. The shift measured here is a property of this split; `scripts/make_stratified_split.py` creates a stratified split for re-running the analysis.
- **Single split, few runs.** Per-class F1 varies between runs; the analysis in sections 2–4 comes from one run.
- **No validation set.** The number of epochs and the STREET_NAME weight were effectively tuned on the test set, so test scores are likely optimistic.
- **Language labels** come from a heuristic based on Kazakh-specific letters. Kazakh words without such letters are counted as Russian, and the "mixed" label often corresponds to Russian text containing a Kazakh toponym.
- **Error definition.** A complaint counts as erroneous if any of its entities is wrong, which ties errors to the number of entities.

## Next steps

- Check whether test street names are new strings unseen in training (lexical novelty).
- Re-run all analyses on the re-stratified split and over several seeds.
- Analyze errors at the entity level instead of the complaint level.

## References

- A. Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. (XLM-RoBERTa.)
- H. Nakayama. *seqeval: A Python framework for sequence labeling evaluation.* 2018. https://github.com/chakki-works/seqeval
- Y. Lin et al. *Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis.* EMNLP 2024. (Representation-space analysis approach adapted in section 2.)
- L. McInnes, J. Healy, J. Melville. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426, 2018.
- N. Bonneel, J. Rabin, G. Peyré, H. Pfister. *Sliced and Radon Wasserstein Barycenters of Measures.* Journal of Mathematical Imaging and Vision 51, 2015. (Sliced Wasserstein distance.)
- R. Flamary et al. *POT: Python Optimal Transport.* JMLR 22(78), 2021.
- K. Sechidis, G. Tsoumakas, I. Vlahavas. *On the Stratification of Multi-Label Data.* ECML PKDD 2011.

## Contact

Ayana Ussenbayeva · ayannnna31@gmail.com · ORCID [0009-0004-1344-8477](https://orcid.org/0009-0004-1344-8477)
