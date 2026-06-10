# Assignment 6: Candidate Test 2022 Analysis Part 2 - Oral Exam Notes

## Opening Speech

```text
Good morning. In this assignment I worked with candidate test data from the Danish 2022 election.
The goal was to explore the political landscape using unsupervised learning methods.

The dataset contains candidate answers to 49 political questions from DR and TV2.
All answers are on the same scale from -2 to 2, where negative and positive values represent disagreement and agreement.

I started by loading and checking the candidate and question datasets.
Then I cleaned simple metadata issues, mapped technical question IDs back to readable question text, and prepared the response matrix.

For the main analysis, I standardized the response data and used PCA to project candidates into two dimensions.
I interpreted the PCA axes using the questions with the strongest loadings.

After that, I compared clustering methods: K-Means, hierarchical clustering, and DBSCAN.
K-Means and PCA showed that the strongest structure is a broad ideological split, not one clean cluster per party.

Finally, I focused on elected candidates.
I projected them into the same PCA space and measured which elected candidates answered most similarly or most differently.
The main limitation is that PCA is only a two-dimensional approximation and clustering results depend on preprocessing and parameter choices.
```

---

## Cells 0-3: Title And Imports

**What to say**

```text
At the beginning I define the topic and import the libraries needed for the analysis.
Pandas and NumPy are used for tabular data handling, Matplotlib and Seaborn for plots, and Scikit-learn and SciPy for PCA, clustering and distance calculations.
```

**Plain English**

This is only setup. No analysis has been done yet.

**Most important**

- `pandas` - loading and working with tables.
- `numpy` - numerical operations.
- `matplotlib`, `seaborn` - visualizations.
- `StandardScaler` - standardizes the response features.
- `PCA` - reduces 49 response variables to 2 main axes.
- `KMeans`, `DBSCAN` - clustering algorithms.
- `silhouette_score` - evaluates K-Means cluster separation.
- `linkage`, `dendrogram` - hierarchical clustering.
- `pdist`, `squareform` - pairwise distances between elected candidates.

**Question**

```text
Why do you use unsupervised learning here?
```

**Answer**

```text
Because the goal is not to predict a known label, but to explore structure in the candidate responses.
I want to see whether candidates naturally form political groups or blocs.
```

---

## Cells 4-8: Load Datasets

**What to say**

```text
Here I load the candidate response data and the question metadata.
The main candidate dataset contains all candidates, while the elected dataset contains only elected candidates.
The DR and TV2 question files are used later to interpret the technical question IDs.
```

**Plain English**

I load the raw data and quickly check that the shapes and first rows look reasonable.

**Most important**

- `all_df` contains all candidate responses.
- `elected_df` contains elected candidate responses.
- `drq_df` and `tv2q_df` contain question text.
- The data is loaded from the local `data` folder.
- The first previews are sanity checks, not modelling steps.

**Question**

```text
Why do you display the first rows?
```

**Answer**

```text
To check that the files loaded correctly and to understand the structure of the columns before cleaning or modelling.
```

---

## Cells 9-13: Party Colors And Feature Preparation

**What to say**

```text
I define party colors for consistent visualizations.
Then I separate metadata columns from response columns.
The response columns are the actual features used for PCA and clustering.
```

**Plain English**

Candidate names, parties, regions and age are descriptive information. The political answers are the features.

**Most important**

- `meta_cols` are not used as clustering features.
- `feature_cols` are the 49 political response columns.
- Age value `0` is replaced with missing because it is not realistic.
- The notebook checks missing values and response range.
- The responses are already numeric from `-2` to `2`.

**Question**

```text
Why do you not use party as a feature?
```

**Answer**

```text
Because party is metadata used for interpretation.
If I used party as a feature, the clustering would partly reproduce the party labels instead of discovering structure from answers.
```

**Question**

```text
Why replace age 0 with missing?
```

**Answer**

```text
Because age 0 is not meaningful for election candidates, so it is likely a placeholder or data issue.
Age is not used in the PCA or clustering features, but cleaning it makes the metadata more honest.
```

---

## Cells 14-19: Question Mapping

**What to say**

```text
The response columns have technical identifiers, so I map them back to the original question text.
I build one mapping for DR questions and one for TV2 questions, then combine them into a single question map.
This makes the PCA loadings and party averages interpretable.
```

**Plain English**

Without this mapping, I would only know that feature `530` is important, but not what political question it represents.

**Most important**

- DR question IDs are converted to strings to match feature names.
- TV2 questions are filtered to the TV2 response columns.
- `question_map` keeps `feature`, `source`, `topic`, and `question`.
- The final map is ordered according to `feature_cols`.
- This step is for interpretation, not for model fitting.

**Question**

```text
Why do you need the original question text?
```

**Answer**

```text
Because PCA and clustering only produce numerical patterns.
To explain what a PCA axis means politically, I need to connect important features back to the actual questions.
```

---

## Cells 20-23: PCA Political Landscape

**What to say**

```text
Before PCA I standardize the response matrix.
Even though all answers are on the same scale from -2 to 2, different questions can have different variances.
Scaling gives each question equal weight in PCA and clustering.

Then I fit PCA with two components and plot candidates in the PC1-PC2 space.
The first two components explain about 52% of the total variance, so the map is useful but incomplete.
```

**Plain English**

PCA compresses 49 answers into two coordinates so we can visualize the political landscape.

**Most important**

- `fit_transform` is used on all candidates because this is exploratory unsupervised analysis.
- `PC1` is the direction of largest variation.
- `PC2` is the second largest independent direction of variation.
- Similar points mean similar answer patterns.
- The sign of a PCA axis is arbitrary.

**Question**

```text
Why do you scale if all answers are already from -2 to 2?
```

**Answer**

```text
Because PCA is variance-based.
Even on the same scale, some questions may have higher variance than others.
Scaling prevents high-variance questions from dominating the components.
```

**Question**

```text
What is the trade-off of scaling?
```

**Answer**

```text
The trade-off is that all questions are treated as equally important.
If a high-variance question is politically meaningful, scaling reduces its natural influence.
```

---

## Cells 24-28: PCA Axis Interpretation

**What to say**

```text
To interpret PCA, I calculate loadings.
Loadings show how strongly each original question contributes to each principal component.
I inspect the questions with the largest absolute loadings for PC1 and PC2.
```

**Plain English**

This tells me which questions define each PCA axis.

**Most important**

- Large absolute loading means strong influence on that component.
- PC1 and PC2 are interpreted using question text, not only the plot.
- Positive and negative signs indicate opposite directions on the same axis.
- The sign itself is arbitrary, so the interpretation should focus on contrast.

**Question**

```text
How do you know what PC1 means?
```

**Answer**

```text
I look at the questions with the strongest absolute PC1 loadings.
Those questions contribute most to the PC1 direction, so they help explain the political meaning of the axis.
```

---

## Cells 29-33: Average Party Positions By Question

**What to say**

```text
I calculate average party responses for each question.
Then I select important questions from the PCA loadings and plot average party positions for those questions.
This checks whether the PCA interpretation matches real party-level response patterns.
```

**Plain English**

If a question is important for PCA, I want to see how parties differ on that question.

**Most important**

- Grouping is done by party.
- The average response is still on the original response scale.
- Selected questions come from important PCA loadings.
- These plots support the political interpretation of PC1 and PC2.

**Question**

```text
Why not plot all 49 questions?
```

**Answer**

```text
Because plotting all questions would be too much for a clear notebook.
I focus on questions that are most important for the PCA axes.
```

---

## Cells 34-37: K-Means Cluster Selection

**What to say**

```text
I test K-Means for different numbers of clusters and calculate the silhouette score.
The silhouette score measures how separated and compact the clusters are.
The best score is for 2 clusters, so the strongest structure is a broad two-bloc division.
```

**Plain English**

The data does not naturally split into one clean cluster per party. It mostly splits into two larger political blocs.

**Most important**

- K-Means is run on standardized response data.
- `k` is tested from 2 to 14.
- Silhouette score is higher when clusters are better separated.
- `k = 2` is the most defensible choice according to the metric.
- Higher `k` values are weaker and more exploratory.

**Question**

```text
Why do you use silhouette score?
```

**Answer**

```text
Because there is no true cluster label in unsupervised learning.
Silhouette score gives an internal measure of how well-separated the clusters are.
```

---

## Cells 38-41: K-Means With Two Clusters

**What to say**

```text
I fit K-Means with 2 clusters because that had the best silhouette score.
Then I compare the clusters with party membership and plot the clusters on the PCA map.
The result mostly separates candidates into two broad ideological blocs.
```

**Plain English**

K-Means finds a broad political split, not exact parties.

**Most important**

- `best_k = 2` is chosen from silhouette score.
- The crosstab shows which parties dominate each cluster.
- The PCA plot visualizes where the clusters are located.
- The clusters mostly follow PC1.

**Question**

```text
Does K-Means discover the parties?
```

**Answer**

```text
No. It mainly discovers broader ideological blocs.
That means party labels are more detailed than the strongest natural cluster structure in the responses.
```

---

## Cells 42-45: Exploratory K-Means With Five Clusters

**What to say**

```text
I also try 5 clusters as an exploratory analysis.
This gives a more detailed structure than 2 clusters, but it is not the best solution according to silhouette score.
I use it only to see whether smaller political subgroups appear inside the broader blocs.
```

**Plain English**

`k = 5` is interesting, but not as defensible as `k = 2`.

**Most important**

- `k = 5` is exploratory.
- It gives more detail but weaker separation.
- It should not be presented as the best model.
- It helps discuss subgroups inside broad political blocs.

**Question**

```text
Why do you show 5 clusters if 2 is best?
```

**Answer**

```text
Because the assignment asks whether there is room for more clusters or if reduction is needed.
The 5-cluster result helps explore that, but I still state that 2 clusters is more defensible by silhouette score.
```

---

## Cells 46-48: Hierarchical Clustering Of Parties

**What to say**

```text
Here I perform hierarchical clustering on average party profiles.
This is different from K-Means, which was done on individual candidates.
The dendrogram shows party-level similarity based on average responses.
```

**Plain English**

This answers: which parties are similar to each other on average?

**Most important**

- The input is `party_profiles`, not individual candidates.
- Each party is represented by its average response vector.
- Ward linkage groups parties by similarity.
- The dendrogram supports the broad bloc interpretation.

**Question**

```text
Why cluster party averages instead of candidates here?
```

**Answer**

```text
Because hierarchical clustering is easier to read at party level.
It gives a compact overview of which parties have similar average response profiles.
```

---

## Cells 49-53: DBSCAN Clustering

**What to say**

```text
I test DBSCAN with several epsilon values.
DBSCAN looks for dense regions separated by lower-density space.
In this dataset it is unstable: small epsilon values create many noise points, while larger epsilon values merge almost everyone into one cluster.
```

**Plain English**

DBSCAN is not a good fit here because the political landscape looks continuous, not like separate dense islands.

**Most important**

- `eps` controls how close points must be to count as neighbours.
- `min_samples = 5` is kept fixed.
- Many noise points mean `eps` is too strict.
- One giant cluster means `eps` is too loose.
- The result supports the idea of a continuous political landscape.

**Question**

```text
Why does DBSCAN perform poorly here?
```

**Answer**

```text
Because candidate responses do not form clearly separated dense groups.
They form a more continuous political space, so DBSCAN either marks many points as noise or merges most points together.
```

---

## Cells 54-57: Elected Candidates Political Landscape

**What to say**

```text
I project elected candidates into the same PCA space as all candidates.
I use the already fitted scaler and PCA model, so elected candidates are compared in the same coordinate system.
Then I visualize elected candidates by party.
```

**Plain English**

This is the same political map, but focused only on elected candidates.

**Most important**

- `transform` is used instead of fitting PCA again.
- This keeps the PCA axes comparable to the earlier map.
- The plot shows party regions among elected candidates.
- Parties are visible but not perfectly separated.

**Question**

```text
Why do you use transform instead of fit_transform for elected candidates?
```

**Answer**

```text
Because I want to place elected candidates into the same PCA space that was already learned from all candidates.
If I fitted PCA again, the axes could change and the plots would not be directly comparable.
```

---

## Cells 58-62: Agreement And Disagreement Among Elected Candidates

**What to say**

```text
I calculate pairwise Euclidean distances between elected candidates using their standardized response vectors.
Small distance means two candidates answered the 49 questions similarly.
Large distance means their response patterns are very different.
```

**Plain English**

This finds the most similar and most different pairs of elected candidates.

**Most important**

- Distances are computed on standardized answers.
- The distance is based on all 49 questions.
- `most_similar` shows the smallest distances.
- `most_different` shows the largest distances.
- This measures questionnaire agreement, not personal or parliamentary agreement.

**Question**

```text
What does a small distance mean?
```

**Answer**

```text
It means that two elected candidates gave similar answers across the candidate test questions.
```

**Question**

```text
Is this the same as political cooperation?
```

**Answer**

```text
No. It only measures similarity in questionnaire responses.
It does not prove that candidates cooperate or vote the same way in parliament.
```

---

## Cells 63-66: Internal Disagreement Within Parties

**What to say**

```text
I calculate internal party disagreement by measuring distances between elected candidates from the same party.
For each party, I compute the mean and maximum internal distance.
Higher values mean that elected candidates from that party are more spread out in their answers.
```

**Plain English**

This checks which parties are internally more unified or more diverse.

**Most important**

- The calculation uses only elected candidates.
- Parties with fewer than two candidates are skipped.
- Mean internal distance measures typical disagreement.
- Maximum internal distance shows the biggest internal gap.
- Results should be interpreted carefully for parties with few elected candidates.

**Question**

```text
Why is party size important here?
```

**Answer**

```text
Because a party with only a few elected candidates can have unstable distance estimates.
One unusual candidate can strongly affect the mean or maximum distance.
```

---

## Cells 67-68: Conclusion And Limitations

**What to say**

```text
In the conclusion I summarize the main finding: the data mostly shows broad ideological blocs rather than one clean cluster per party.
PCA, K-Means and hierarchical clustering support this.
DBSCAN is less suitable because the data looks continuous rather than density-separated.

The limitations are important: PCA loses information, clustering depends on parameters and preprocessing, and the elected dataset does not contain every elected member.
```

**Plain English**

The notebook gives a useful exploratory political map, but it should not be treated as an exact or final truth.

**Most important**

- PCA is an approximation.
- Two components explain only part of the variance.
- K-Means depends on `k`.
- DBSCAN depends strongly on `eps`.
- Scaling treats all questions as equally important.
- Elected candidates data is incomplete.

**Question**

```text
What is the main conclusion?
```

**Answer**

```text
The strongest structure in the candidate responses is a broad ideological split.
The data does not naturally form one clear cluster per party.
```

**Question**

```text
What is the biggest limitation?
```

**Answer**

```text
The biggest limitation is that PCA and clustering simplify the data.
They reveal useful patterns, but they do not capture the full political complexity of all 49 questions.
```

---

## Exam Checklist

- Explain the goal as exploratory unsupervised learning.
- Say clearly that there is no train/test split because this is not predictive supervised modelling.
- Explain why party is metadata, not a feature.
- Explain why scaling is used before PCA and clustering.
- Explain the trade-off of scaling.
- Explain PCA as dimensionality reduction from 49 questions to 2 axes.
- Interpret PCA axes using loadings, not by guessing from the plot.
- Mention that PCA sign is arbitrary.
- Mention that the first two PCs explain about 52% of variance.
- Explain silhouette score and why `k = 2` is most defensible.
- Say that `k = 5` is exploratory.
- Distinguish candidate-level clustering from party-level hierarchical clustering.
- Explain why DBSCAN is not suitable here.
- Explain why elected candidates are transformed into the existing PCA space.
- Explain Euclidean distance as questionnaire similarity.
- Mention limitations: incomplete elected data, PCA information loss, parameter dependence, scaling assumptions.
