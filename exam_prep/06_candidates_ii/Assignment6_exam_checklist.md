# Assignment 6: Candidate Test 2022 Analysis Part 2 - Exam Checklist Map

This file maps the official exam overview topics to the Assignment 6 notebook.

Sources:

- `Materials/exam/Exam_information_and_assignments_overview.pdf`
- Notebook: `MAL1-LAB/Answers/Assignment6/Assignment6.ipynb`
- Notes: `MAL1-LAB/Answers/Assignment6/Assignment6_notebook_explanation.md`
- Speech notes: `MAL1-LAB/Answers/Assignment6/Assignment6_speech_notes.md`

---

## Official Assignment-Specific Checklist From Exam Overview

Official topic for Assignment 6: **Candidate Test 2022 Analysis Part 2**

- Dimensionality reduction algorithms
- PCA
- Choice of PC and their meaning
- Clustering algorithms
- K-means
- DBSCAN
- Hierarchical clustering

General exam expectation from the overview:

- Explain the underlying theory.
- Explain the reasoning behind code decisions.
- Present and interpret results.
- Understand the code and how it applies the theory.

Supporting notebook topics:

- Data loading and feature preparation
- Question mapping for interpretability
- Scaling / normalization
- Average party positions by question
- Elected candidates political landscape
- Agreement and disagreement using distances
- Limitations of exploratory unsupervised analysis

---

## Official Topic To Notebook Map

| Official exam topic | Where in notebook | How it is covered |
| --- | --- | --- |
| Dimensionality reduction algorithms | Cells 20-28 | PCA reduces 49 response variables to 2 components for visualization and interpretation. |
| PCA | Cells 21-23 | Standardized responses are transformed into PC1 and PC2; explained variance is reported. |
| Choice of PC and their meaning | Cells 23-28, 33 | The first two PCs are used for plotting, and loadings are used to explain what PC1 and PC2 represent politically. |
| Clustering algorithms | Cells 34-53 | K-Means, hierarchical clustering and DBSCAN are compared. |
| K-Means | Cells 34-45 | Different `k` values are tested; `k = 2` is selected by silhouette score and `k = 5` is exploratory. |
| DBSCAN | Cells 49-53 | Different `eps` values are tested; results show unstable/noisy or merged clusters. |
| Hierarchical clustering | Cells 46-48 | Party-average profiles are clustered and visualized with a dendrogram. |

---

## 1. Data Loading And Feature Preparation

**Status:** covered

**Where in notebook**

- Cells 4-8: load datasets, check shapes, preview data
- Cells 9-13: define colors, clean age metadata, separate metadata from response features

**What happens**

- Load `alldata.xlsx`, `electeddata.xlsx`, `drq.xlsx`, and `tv2q.xlsx`.
- Check dataset shapes.
- Preview candidate and question tables.
- Define party colors for consistent plots.
- Replace age value `0` with missing.
- Separate metadata columns from the 49 response feature columns.
- Check missing values and response range.

**Why it matters**

The analysis needs a clean response matrix where only political answers are used as features.
Metadata like name, party, region and age should help interpretation, but should not drive PCA or clustering.

**What to say**

```text
I first loaded the candidate response data and question metadata.
Then I separated descriptive metadata from the actual response features.
The PCA and clustering are based only on the 49 political answer columns.
```

**Possible question**

```text
Why do you not use party as a feature?
```

**Answer**

```text
Because party is used for interpretation.
If I used party as an input feature, the clustering would partly reproduce party labels instead of discovering structure from the answers.
```

---

## 2. Question Mapping And Interpretability

**Status:** covered

**Where in notebook**

- Cells 14-19: build DR mapping, TV2 mapping, combined `question_map`, preview mapped questions

**What happens**

- Technical DR and TV2 question IDs are mapped to readable question text.
- DR and TV2 mappings are combined into one `question_map`.
- The final question map is ordered according to `feature_cols`.

**Why it matters**

PCA and clustering operate on numerical columns, but exam interpretation requires political meaning.
The question map makes it possible to explain which concrete political questions define PCA axes and party differences.

**What to say**

```text
The original response columns are technical IDs.
I map them back to the original question text so that later PCA loadings and party averages can be interpreted politically.
```

**Possible question**

```text
Why is question mapping necessary?
```

**Answer**

```text
Because otherwise I would only know that a technical feature ID is important.
To explain the PCA axis, I need to know the actual political question behind that feature.
```

---

## 3. Scaling / Normalization

**Status:** covered

**Where in notebook**

- Cell 21: `StandardScaler`
- Cells 23, 68: interpretation and limitations of scaling

**What happens**

- The 49 response features are standardized before PCA and clustering.
- The fitted scaler is reused later for elected candidates.

**Why it matters**

PCA, K-Means, DBSCAN and Euclidean distances are sensitive to feature scale and variance.
Even though answers are all between `-2` and `2`, some questions can vary more than others.
Standardization prevents high-variance questions from dominating the analysis.

**What to say**

```text
I standardize the responses before PCA and clustering.
The answers already have the same range, but questions may still have different variance.
Scaling gives each question equal weight in the variance-based and distance-based methods.
```

**Possible question**

```text
Why scale if all answers are already on the same scale?
```

**Answer**

```text
Because same range does not mean same variance.
PCA and distance-based clustering can still be dominated by questions where candidates vary more.
```

**Possible question**

```text
What is the trade-off of scaling?
```

**Answer**

```text
Scaling treats all questions as equally important.
That is good for balanced analysis, but it removes the natural influence of high-variance questions.
```

---

## 4. PCA Dimensionality Reduction

**Status:** covered

**Where in notebook**

- Cell 21: fit PCA with two components
- Cell 22: PCA scatter plot
- Cell 23: interpretation of explained variance

**What happens**

- PCA reduces 49 standardized response features to two components.
- Candidates are plotted in the `PC1` / `PC2` space.
- PC1 explains about `41.7%` of variance.
- PC2 explains about `10.3%` of variance.
- Together they explain about `52%` of variance.

**Why it matters**

PCA gives a visual summary of the political landscape.
Candidates close together in the PCA plot have similar answer patterns.
But PCA is only an approximation, because two components do not capture all information from 49 questions.

**What to say**

```text
PCA compresses the 49 response variables into two main dimensions.
This makes it possible to visualize the political landscape.
The first two components explain about 52 percent of the variance, so the plot is useful but incomplete.
```

**Possible question**

```text
What does it mean that two candidates are close in the PCA plot?
```

**Answer**

```text
It means they gave similar answer patterns across the candidate test questions, at least according to the main PCA dimensions.
```

**Possible question**

```text
Can you fully interpret politics from the PCA plot?
```

**Answer**

```text
No. The plot is a two-dimensional approximation.
It captures an important part of the structure, but some information is lost.
```

---

## 5. PCA Loadings And Axis Interpretation

**Status:** covered

**Where in notebook**

- Cells 24-28: calculate loadings, top PC1 features, top PC2 features, loading plots
- Cell 33: interpretation of PC1 and PC2

**What happens**

- PCA loadings are calculated for each original question.
- Questions with the largest absolute loadings are selected.
- These questions are used to interpret PC1 and PC2 politically.

**Why it matters**

PCA axes should not be interpreted only by looking at where parties appear on a plot.
Loadings provide a more defensible way to explain what each axis represents.

**What to say**

```text
I interpret PCA using loadings.
The questions with the largest absolute loadings contribute most strongly to each component.
This lets me explain what PC1 and PC2 mean politically.
```

**Possible question**

```text
What is a PCA loading?
```

**Answer**

```text
A loading shows how strongly an original feature contributes to a principal component.
A large absolute loading means the question is important for that axis.
```

**Possible question**

```text
Does the positive side of PC1 always mean left-wing or right-wing?
```

**Answer**

```text
No. The sign of a PCA component is arbitrary.
Only the contrast between opposite directions matters.
```

---

## 6. Average Party Positions By Question

**Status:** covered

**Where in notebook**

- Cells 29-33: average party responses, selected PCA questions, party response plots

**What happens**

- Average response is calculated for each party and question.
- Important questions from PCA loadings are selected.
- Party average plots are created for those selected questions.

**Why it matters**

This connects the PCA interpretation back to concrete party-level behavior.
It helps check whether high-loading questions actually separate parties in a meaningful way.

**What to say**

```text
After identifying important PCA questions, I plot average party responses for them.
This helps verify that the PCA axes correspond to real political differences between parties.
```

**Possible question**

```text
Why not show all 49 questions?
```

**Answer**

```text
Because that would make the notebook too noisy.
I focus on the questions that contribute most to the PCA axes.
```

---

## 7. K-Means And Silhouette Score

**Status:** covered

**Where in notebook**

- Cells 34-37: test K-Means from `k = 2` to `k = 14`, compute silhouette score
- Cells 38-41: fit and plot `k = 2`
- Cells 42-45: exploratory `k = 5`

**What happens**

- K-Means is tested for multiple values of `k`.
- Silhouette score is calculated for each `k`.
- The highest score is for `k = 2`.
- `k = 2` is treated as the most defensible clustering solution.
- `k = 5` is shown as exploratory detail, not as the best model.

**Why it matters**

The assignment asks whether the data supports many clusters corresponding to parties or whether a reduction is needed.
The silhouette score suggests reduction: the strongest structure is a broad two-cluster split.

**What to say**

```text
I use silhouette score to compare different numbers of K-Means clusters.
The best score is for two clusters, which suggests that the strongest structure is a broad ideological split rather than one cluster per party.
```

**Possible question**

```text
What does silhouette score measure?
```

**Answer**

```text
It measures how well-separated and compact the clusters are.
Higher values mean points are closer to their own cluster and farther from other clusters.
```

**Possible question**

```text
Why do you also show five clusters?
```

**Answer**

```text
Five clusters are shown as exploratory analysis.
They reveal smaller subgroups, but the silhouette score supports two clusters more strongly.
```

---

## 8. Hierarchical Clustering

**Status:** covered

**Where in notebook**

- Cells 46-48: party profiles and dendrogram

**What happens**

- Average response profiles are calculated for each party.
- Hierarchical clustering is applied to party averages.
- A dendrogram visualizes party-level similarity.

**Why it matters**

This provides another clustering perspective.
Unlike K-Means and DBSCAN, this section clusters parties, not individual candidates.

**What to say**

```text
Here I cluster average party profiles.
This shows which parties are similar based on their average answers.
The result supports the same broad-bloc structure seen in PCA and K-Means.
```

**Possible question**

```text
Is this candidate-level or party-level clustering?
```

**Answer**

```text
This is party-level clustering.
Each party is represented by its average response vector.
```

---

## 9. DBSCAN

**Status:** covered

**Where in notebook**

- Cells 49-53: coarse and fine `eps` search for DBSCAN

**What happens**

- DBSCAN is tested with several `eps` values.
- Small `eps` creates many noise points.
- Larger `eps` merges most candidates into one cluster.
- No stable meaningful multi-cluster structure is found.

**Why it matters**

DBSCAN is useful when data contains dense, separated groups.
The result suggests that candidate responses form a continuous political landscape instead.

**What to say**

```text
DBSCAN was not very suitable here.
With small epsilon values it labels many candidates as noise, and with larger values it merges almost everyone into one cluster.
This suggests that the political space is continuous rather than density-separated.
```

**Possible question**

```text
What does `eps` mean in DBSCAN?
```

**Answer**

```text
`eps` is the neighbourhood radius.
It controls how close points must be to count as neighbours.
```

**Possible question**

```text
Why does DBSCAN fail here?
```

**Answer**

```text
Because the candidates do not form clearly separated dense groups.
The data is more like a continuous political landscape.
```

---

## 10. Elected Candidates Political Landscape

**Status:** covered

**Where in notebook**

- Cells 54-57: transform elected candidates into the existing PCA space and plot them

**What happens**

- Elected candidates are scaled using the previously fitted scaler.
- Elected candidates are transformed using the previously fitted PCA.
- They are plotted in the same PC1/PC2 coordinate system.

**Why it matters**

Using `transform` instead of refitting keeps the PCA axes comparable.
This allows elected candidates to be interpreted relative to the political landscape built from all candidates.

**What to say**

```text
I transform elected candidates into the same PCA space instead of fitting PCA again.
This keeps the axes comparable to the earlier analysis of all candidates.
```

**Possible question**

```text
Why use `transform` instead of `fit_transform` here?
```

**Answer**

```text
Because the PCA space has already been defined using all candidates.
Using `transform` places elected candidates into that same coordinate system.
If I fitted PCA again, the axes could change.
```

---

## 11. Agreement And Disagreement Among Elected Candidates

**Status:** covered

**Where in notebook**

- Cells 58-62: pairwise distances, most similar pairs, most different pairs
- Cells 63-66: internal party disagreement

**What happens**

- Pairwise Euclidean distances are calculated between elected candidates.
- Small distances identify similar questionnaire responses.
- Large distances identify different questionnaire responses.
- Internal party disagreement is measured using within-party distances.

**Why it matters**

This answers the assignment requirement to highlight which elected candidates agree or disagree most and which parties have significant internal disagreement.

**What to say**

```text
I measure agreement using Euclidean distance on standardized response vectors.
Small distance means similar answers across the 49 questions, while large distance means different answer patterns.
I also calculate within-party distances to identify parties with more internal disagreement.
```

**Possible question**

```text
Does distance mean real political cooperation?
```

**Answer**

```text
No. It only measures similarity in questionnaire answers.
It does not prove that candidates cooperate or vote the same way in parliament.
```

**Possible question**

```text
Why should small parties be interpreted carefully?
```

**Answer**

```text
Because with few elected candidates, one unusual candidate can strongly affect the mean or maximum internal distance.
```

---

## 12. Limitations

**Status:** covered

**Where in notebook**

- Cells 67-68: conclusion and limitations
- Also discussed in Cells 23, 37, 45, 57 and 62

**Main limitations**

- PCA is only a two-dimensional approximation.
- The first two components explain about `52%` of variance, not all information.
- K-Means depends on the chosen `k`.
- DBSCAN depends heavily on `eps`.
- Scaling treats all questions as equally important.
- The elected candidate dataset is incomplete.
- Distance measures questionnaire similarity, not full political behaviour.

**Why it matters**

The analysis is exploratory.
It is useful for finding patterns, but it should not be presented as exact ground truth about Danish politics.

**What to say**

```text
I interpret the results carefully because this is exploratory unsupervised analysis.
PCA simplifies the data, clustering depends on parameters, and the elected dataset is incomplete.
The results show useful patterns, not exact political categories.
```

**Possible question**

```text
What is the biggest limitation of the PCA plot?
```

**Answer**

```text
It reduces 49 questions to two dimensions.
The first two components explain about half of the variance, so the plot is useful but incomplete.
```

---

## Quick Oral Checklist

- Start by saying this is exploratory unsupervised learning.
- Mention there is no target variable and no train/test split because this is not predictive supervised modelling.
- Say party is metadata used for interpretation, not an input feature.
- Explain why question mapping is needed.
- Explain why scaling is used even though all responses are from `-2` to `2`.
- Mention the trade-off of scaling.
- Define PCA as dimensionality reduction.
- Mention that PC1 and PC2 explain about `52%` of variance.
- Interpret PCA axes using loadings.
- Say that PCA sign is arbitrary.
- Explain party-average plots as support for PCA interpretation.
- Explain K-Means and silhouette score.
- Say `k = 2` is the most defensible K-Means result.
- Say `k = 5` is exploratory.
- Distinguish candidate-level clustering from party-level hierarchical clustering.
- Explain why DBSCAN is less suitable.
- Explain why elected candidates use `transform`, not `fit_transform`.
- Explain Euclidean distance as questionnaire similarity.
- Mention internal party disagreement and small-party caution.
- End with limitations: PCA information loss, parameter dependence, incomplete elected data.

---

## One-Minute Summary

```text
This assignment explores Danish candidate test responses using unsupervised learning.
I use PCA to reduce 49 answer variables to two main dimensions and interpret the axes using loadings.
The PCA map shows a broad political landscape where parties form regions but are not perfectly separated.

For clustering, K-Means with two clusters is the most defensible result based on silhouette score.
Five clusters are useful only as exploratory detail.
Hierarchical clustering of party averages supports the same broad-bloc structure.
DBSCAN is less suitable because the data is continuous rather than density-separated.

For elected candidates, I project them into the same PCA space and use distances to find the most similar and most different answer patterns.
Overall, the analysis shows broad ideological blocs, not one clean cluster per party.
```
