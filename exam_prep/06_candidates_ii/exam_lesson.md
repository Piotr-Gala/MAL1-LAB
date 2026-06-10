# Assignment 6: Candidates II - ADHD-Friendly Oral Exam Lesson

## How To Use This File

To jest wersja do nauki ustnej: najpierw masz normalny kawałek lekcji, a dopiero pod spodem wyjaśnienia słów z tej sekcji.

Schemat:

```text
section
lesson text
term explanations
exam sentences
where in code
```

Main exam pattern:

```text
What does it mean?
Where is it in your code?
Why did you do it?
```

---

# 0. Big Picture

Assignment 6 uses Danish candidate test data from the 2022 election.

The goal is to explore political structure in candidate answers.

This is unsupervised learning.

There is no target variable to predict.

The notebook uses 49 political answer columns, standardizes them, reduces them with PCA, and compares clustering methods:

```text
K-Means
DBSCAN
Hierarchical clustering
```

The main conclusion is that the strongest structure is a broad ideological split, not one clean cluster per party.

## Terms From This Section

`unsupervised learning`

```text
Learning without a target label.
The goal is to find patterns, groups or structure.
```

`target variable`

```text
The value a supervised model tries to predict.
In A6, there is no target variable for PCA or clustering.
```

`political structure`

```text
Patterns in how candidates answer political questions.
```

`clustering`

```text
Grouping observations based on similarity.
```

`broad ideological split`

```text
The data separates mostly into large political blocs, not exact parties.
```

## Exam Sentences

```text
This assignment uses unsupervised learning.
The goal is not to predict a known label, but to explore structure in candidate questionnaire answers.
Party is used for interpretation, not as a target variable.
```

---

# 1. Data Loading And Feature Preparation

The notebook loads candidate response data and question metadata.

Main files:

```text
alldata.xlsx
electeddata.xlsx
drq.xlsx
tv2q.xlsx
```

The candidate data contains answers to 49 political questions.

The answers are on a scale from `-2` to `2`.

Metadata columns such as name, party, region and age are separated from the response features.

Only the 49 response columns are used for PCA and clustering.

## Terms From This Section

`metadata`

```text
Descriptive information about candidates.
Examples: name, party, region, age.
```

`response features`

```text
The actual political answer columns used as input.
In A6, there are 49 response features.
```

`feature matrix`

```text
The table of input features used by PCA and clustering.
```

`response scale`

```text
The numerical answer scale from -2 to 2.
Negative and positive values represent disagreement/agreement.
```

`party as metadata`

```text
Party is used for coloring and interpretation, not as an input feature.
```

## Exam Sentences

```text
I load the candidate response data and question metadata.
Then I separate descriptive metadata from the actual response features.
The PCA and clustering are based only on the 49 political answer columns.
```

## Where In Code

```text
Cells 4-8: load datasets and preview them.
Cells 9-13: define party colors, clean age metadata and separate metadata from response features.
```

---

# 2. Why Party Is Not A Feature

Party labels are not used as input features.

They are used only for interpretation and plotting.

If party was used as a feature, the clustering would partly reproduce party labels instead of discovering structure from the answers.

This is important because the assignment is exploratory unsupervised learning.

## Terms From This Section

`party label`

```text
The candidate's political party.
```

`interpretation`

```text
Using labels/colors after analysis to understand the result.
```

`input feature`

```text
A variable used by the algorithm.
Party is not used as an input feature in PCA or clustering.
```

`discovering structure`

```text
Finding patterns from the answer data itself.
```

## Exam Sentences

```text
Party is metadata used for interpretation, not an input feature.
If I used party as a feature, the clustering would partly reproduce party labels instead of discovering structure from the answers.
```

## Where In Code

```text
Cells 9-13: metadata columns are separated from the 49 response feature columns.
```

---

# 3. Question Mapping

The response columns have technical IDs.

The notebook maps those IDs back to readable political question text.

It builds one mapping for DR questions and one for TV2 questions, then combines them into one `question_map`.

This is necessary for interpretation.

Without question mapping, the notebook could say that feature `530` is important, but not what political question it represents.

## Terms From This Section

`technical ID`

```text
A column name or code that is not meaningful by itself.
```

`question_map`

```text
A table that connects feature IDs to readable question text.
```

`interpretability`

```text
The ability to explain what a result means.
```

`PCA loading interpretation`

```text
Using important original questions to explain what PC1 and PC2 mean.
```

## Exam Sentences

```text
The original response columns are technical IDs.
I map them back to the original question text so that PCA loadings and party averages can be interpreted politically.
```

## Where In Code

```text
Cells 14-19: build DR mapping, TV2 mapping and combined question_map.
```

---

# 4. Scaling / Standardization

Before PCA and clustering, the notebook standardizes the 49 response features.

Even though all answers are already between `-2` and `2`, different questions can still have different variance.

Scaling gives each question equal weight in PCA and distance-based clustering.

The trade-off is that scaling also removes the natural influence of high-variance questions.

## Terms From This Section

`scaling`

```text
Putting features on comparable numeric scales.
```

`standardization`

```text
Transforming features to have mean 0 and standard deviation 1.
```

`variance`

```text
How much values differ from each other.
Some questions may vary more than others.
```

`distance-based clustering`

```text
Clustering methods that depend on distances between points.
Examples: K-Means, DBSCAN, Euclidean distances.
```

`trade-off of scaling`

```text
Scaling gives equal weight to questions, but it removes the natural influence of higher-variance questions.
```

## Exam Sentences

```text
I standardize the responses before PCA and clustering.
The answers already have the same range, but questions may still have different variance.
Scaling gives each question equal weight in variance-based and distance-based methods.
```

## Where In Code

```text
Cell 21: StandardScaler is used before PCA and clustering.
```

---

# 5. PCA Dimensionality Reduction

PCA reduces the 49 standardized response features to two main dimensions:

```text
PC1
PC2
```

This makes it possible to visualize candidates in a two-dimensional political map.

In the notebook:

```text
PC1 explains about 41.7% of variance
PC2 explains about 10.3% of variance
Together they explain about 52% of variance
```

This means the PCA plot is useful, but incomplete.

It is a simplification of the original 49-dimensional answer data.

## Terms From This Section

`PCA`

```text
Principal Component Analysis.
A dimensionality reduction method.
```

`dimensionality reduction`

```text
Reducing many features to fewer dimensions while keeping important structure.
```

`principal component`

```text
A new axis created as a weighted combination of original features.
```

`PC1`

```text
The first principal component.
It captures the largest amount of variance.
```

`PC2`

```text
The second principal component.
It captures the next largest independent direction of variance.
```

`explained variance`

```text
How much of the original variation is captured by a component.
```

## Exam Sentences

```text
PCA compresses the 49 response variables into two main dimensions.
This makes it possible to visualize the political landscape.
The first two components explain about 52% of the variance, so the plot is useful but incomplete.
```

## Where In Code

```text
Cells 20-23: standardization, PCA with two components, PCA scatter plot and explained variance.
```

---

# 6. Principal Components And Orthogonality

A principal component is not one original question.

It is a weighted combination of many original questions.

PC1 captures the strongest direction of variation in the data.

PC2 captures the next strongest direction, while being orthogonal to PC1.

Orthogonal means the components capture different, non-overlapping directions of variation.

## Terms From This Section

`weighted combination`

```text
Many original features are combined with different weights.
```

`orthogonal`

```text
At a right angle in feature space.
In practice: components are uncorrelated and capture different variance.
```

`uncorrelated directions`

```text
The components do not repeat the same information.
```

`feature space`

```text
The high-dimensional space defined by all input features.
Here: 49 answer dimensions.
```

## Exam Sentences

```text
A principal component is a linear combination of original features.
PC1 captures the most variance, and PC2 captures the next most variance while being orthogonal to PC1.
This means PC1 and PC2 summarize different patterns in candidate responses.
```

## Where In Code

```text
Cells 21-23: PCA creates PC1 and PC2 from the standardized response matrix.
```

---

# 7. PCA Loadings And Axis Interpretation

To interpret PCA, the notebook calculates loadings.

Loadings show how strongly each original question contributes to each principal component.

The notebook selects questions with the largest absolute loadings for PC1 and PC2.

Those questions are used to interpret the political meaning of the axes.

The sign of a PCA component is arbitrary.

That means the important thing is the contrast between opposite directions, not whether positive means "good" or "bad".

## Terms From This Section

`loading`

```text
How strongly an original feature contributes to a principal component.
```

`absolute loading`

```text
The size of the loading without caring about positive or negative sign.
Large absolute loading means strong influence.
```

`axis interpretation`

```text
Explaining what PC1 or PC2 means using the original questions.
```

`arbitrary sign`

```text
PCA axes can be flipped.
Positive and negative directions are meaningful as opposites, but the sign itself is not fixed.
```

## Exam Sentences

```text
I interpret PCA using loadings.
The questions with the largest absolute loadings contribute most strongly to each component.
The sign of a PCA component is arbitrary, so I focus on the contrast between opposite directions.
```

## Where In Code

```text
Cells 24-28: calculate loadings and identify top PC1 and PC2 questions.
Cell 33: interpretation of PC1 and PC2.
```

---

# 8. Average Party Positions

After identifying important PCA questions, the notebook calculates average party responses.

For each party and selected question, it calculates the average answer.

This connects the PCA interpretation back to concrete political differences between parties.

The notebook does not plot all 49 questions because that would be too noisy.

It focuses on questions that contribute most to the PCA axes.

## Terms From This Section

`average party response`

```text
The mean answer for one party on one question.
```

`selected PCA questions`

```text
Questions chosen because they have high loadings on PC1 or PC2.
```

`party-level behavior`

```text
Patterns in average answers by party.
```

`too noisy`

```text
Too much information to read clearly.
Plotting all 49 questions would be hard to interpret.
```

## Exam Sentences

```text
After identifying important PCA questions, I plot average party responses for those questions.
This helps verify that the PCA axes correspond to real political differences between parties.
```

## Where In Code

```text
Cells 29-33: average party responses and plots for important PCA questions.
```

---

# 9. K-Means And Silhouette Score

K-Means is a clustering algorithm.

It requires choosing the number of clusters, called `k`.

The notebook tests K-Means for different values of `k` from 2 to 14.

For each `k`, it calculates the silhouette score.

The best silhouette score is for:

```text
k = 2
```

So `k = 2` is the most defensible clustering solution.

This suggests that the strongest structure is a broad two-bloc split, not one cluster per party.

## Terms From This Section

`K-Means`

```text
A clustering algorithm that groups points around k centroids.
```

`k`

```text
The chosen number of clusters.
```

`centroid`

```text
The center of a cluster.
K-Means assigns points to the nearest centroid.
```

`silhouette score`

```text
Measures how compact and separated clusters are.
Higher usually means better cluster structure.
```

`defensible solution`

```text
A choice you can justify with a metric or reasoning.
Here, k = 2 is defensible because it has the best silhouette score.
```

## Exam Sentences

```text
I use silhouette score to compare different numbers of K-Means clusters.
The best score is for two clusters, which suggests that the strongest structure is a broad ideological split rather than one cluster per party.
```

## Where In Code

```text
Cells 34-37: test K-Means from k = 2 to k = 14 and compute silhouette scores.
Cells 38-41: fit and plot k = 2.
```

---

# 10. K-Means With Five Clusters

The notebook also tests K-Means with:

```text
k = 5
```

This is exploratory.

It gives a more detailed view of possible subgroups inside the broad two-bloc structure.

But it is not presented as the best solution, because the silhouette score supports `k = 2` more strongly.

## Terms From This Section

`exploratory analysis`

```text
Analysis used to investigate patterns, not to claim a final best model.
```

`subgroup`

```text
A smaller group inside a broader cluster.
```

`weaker separation`

```text
Clusters are less clearly separated.
```

## Exam Sentences

```text
I also show five clusters as exploratory detail.
It may reveal smaller political subgroups, but the silhouette score supports two clusters more strongly.
So k = 5 is interesting, but k = 2 is more defensible.
```

## Where In Code

```text
Cells 42-45: exploratory K-Means with five clusters.
```

---

# 11. Hierarchical Clustering

The notebook also uses hierarchical clustering.

This part clusters parties, not individual candidates.

Each party is represented by its average response vector across the 49 questions.

The result is shown as a dendrogram.

This gives a compact party-level view of similarity.

## Terms From This Section

`hierarchical clustering`

```text
A clustering method that builds a tree of similarities.
```

`agglomerative clustering`

```text
Starts with each point as its own cluster and repeatedly merges the closest clusters.
```

`party profile`

```text
The average response vector for one party.
```

`dendrogram`

```text
A tree diagram showing how clusters merge.
```

`linkage`

```text
The rule for measuring distance between clusters.
```

## Exam Sentences

```text
In hierarchical clustering, I cluster average party profiles rather than individual candidates.
The dendrogram shows which parties have similar average response patterns.
This supports the broad-bloc interpretation.
```

## Where In Code

```text
Cells 46-48: party profiles and dendrogram.
```

---

# 12. DBSCAN

DBSCAN is a density-based clustering algorithm.

It groups dense regions and labels sparse points as noise.

The notebook tests different `eps` values.

The results are not stable:

```text
small eps -> many noise points
large eps -> most candidates merge into one cluster
```

This suggests that the data is more continuous than density-separated.

DBSCAN is less suitable here.

## Terms From This Section

`DBSCAN`

```text
Density-based clustering.
It finds dense groups and marks sparse points as noise.
```

`eps`

```text
Neighbourhood radius.
Controls how close points must be to count as neighbours.
```

`min_samples`

```text
Minimum number of nearby points needed to form a dense region.
```

`noise point`

```text
A point not assigned to any cluster by DBSCAN.
```

`continuous political landscape`

```text
Candidates are spread gradually, not separated into dense isolated groups.
```

## Exam Sentences

```text
DBSCAN was less suitable here.
With small epsilon values, many candidates are labelled as noise.
With larger epsilon values, almost all candidates merge into one cluster.
This suggests that the political space is continuous rather than density-separated.
```

## Where In Code

```text
Cells 49-53: DBSCAN with different eps values.
```

---

# 13. Elected Candidates In The Same PCA Space

The notebook then focuses on elected candidates.

It places elected candidates into the same PCA space as all candidates.

It uses the already fitted scaler and PCA model.

This means it uses `transform`, not `fit_transform`.

If PCA were fitted again only on elected candidates, the axes could change and the plots would not be directly comparable.

## Terms From This Section

`elected candidates`

```text
Candidates who were elected to parliament.
```

`same PCA space`

```text
The same PC1 and PC2 coordinate system learned earlier.
```

`transform`

```text
Apply an already fitted scaler/PCA to new data.
```

`fit_transform`

```text
Fit a new transformation and apply it.
Not used here because it would create new PCA axes.
```

`comparable axes`

```text
The plotted coordinates mean the same thing across plots.
```

## Exam Sentences

```text
I transform elected candidates into the same PCA space instead of fitting PCA again.
This keeps the axes comparable to the earlier analysis of all candidates.
If I fitted PCA again, the axes could change.
```

## Where In Code

```text
Cells 54-57: transform elected candidates using the existing scaler and PCA model.
```

---

# 14. Distances Between Elected Candidates

The notebook calculates pairwise Euclidean distances between elected candidates.

The distances are calculated using standardized responses across all 49 questions.

Small distance means similar questionnaire answers.

Large distance means different answer patterns.

This measures questionnaire similarity, not real political cooperation.

## Terms From This Section

`pairwise distance`

```text
Distance calculated between every pair of candidates.
```

`Euclidean distance`

```text
Straight-line distance in feature space.
```

`standardized response vector`

```text
A candidate's 49 answers after scaling.
```

`questionnaire similarity`

```text
Similarity based only on candidate test answers.
```

`most similar / most different`

```text
Pairs with smallest or largest distances.
```

## Exam Sentences

```text
I measure agreement using Euclidean distance on standardized response vectors.
Small distance means similar answers across the 49 questions, while large distance means different answer patterns.
This does not prove political cooperation; it only measures questionnaire similarity.
```

## Where In Code

```text
Cells 58-62: pairwise distances, most similar pairs and most different pairs.
```

---

# 15. Internal Party Disagreement

The notebook also measures internal disagreement within parties.

For each party with at least two elected candidates, it calculates distances between candidates from the same party.

Mean internal distance shows typical disagreement.

Maximum internal distance shows the largest internal gap.

Small parties should be interpreted carefully because one unusual candidate can strongly affect the result.

## Terms From This Section

`internal disagreement`

```text
How spread out candidates from the same party are in their answers.
```

`within-party distance`

```text
Distance between candidates from the same party.
```

`mean internal distance`

```text
The average disagreement inside a party.
```

`maximum internal distance`

```text
The largest gap between two candidates in the same party.
```

`small-party caution`

```text
With few candidates, one unusual person can strongly affect averages.
```

## Exam Sentences

```text
I calculate internal party disagreement using within-party distances.
Mean internal distance shows typical disagreement, while maximum internal distance shows the largest internal gap.
I interpret small parties carefully because few candidates can make the estimate unstable.
```

## Where In Code

```text
Cells 63-66: internal party disagreement among elected candidates.
```

---

# 16. Limitations

This assignment is exploratory.

The results should not be treated as exact political truth.

Main limitations:

```text
PCA reduces 49 questions to 2 dimensions.
The first two PCs explain about 52% of variance, not all information.
K-Means depends on k.
DBSCAN depends strongly on eps.
Scaling treats all questions as equally important.
The elected candidate dataset is incomplete.
Distances measure questionnaire similarity, not full political behaviour.
```

## Terms From This Section

`exploratory`

```text
Used to discover patterns, not to prove final truth.
```

`information loss`

```text
Some detail is lost when reducing 49 dimensions to 2.
```

`parameter dependence`

```text
Results change depending on choices like k or eps.
```

`incomplete dataset`

```text
The elected candidate data does not include every elected member.
```

`not ground truth`

```text
Clusters are patterns created by algorithms, not official political categories.
```

## Exam Sentences

```text
I interpret the results carefully because this is exploratory unsupervised analysis.
PCA simplifies the data, clustering depends on parameters, and the elected dataset is incomplete.
The results show useful patterns, not exact political categories.
```

## Where In Code

```text
Cells 67-68: conclusion and limitations.
Also discussed in Cells 23, 37, 45, 57 and 62.
```

---

# 17. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Data loading:
Cells 4-8, where I load all candidate data, elected candidates and question metadata.

Metadata and feature preparation:
Cells 9-13, where I separate metadata from the 49 response features.

Question mapping:
Cells 14-19, where I map technical IDs to readable DR and TV2 questions.

Scaling:
Cell 21, where I use StandardScaler.

PCA:
Cells 21-23, where I reduce 49 features to PC1 and PC2 and report explained variance.

PCA loadings:
Cells 24-28, where I calculate important questions for PC1 and PC2.

Average party positions:
Cells 29-33.

K-Means and silhouette score:
Cells 34-37.

K-Means with two clusters:
Cells 38-41.

Exploratory K-Means with five clusters:
Cells 42-45.

Hierarchical clustering:
Cells 46-48.

DBSCAN:
Cells 49-53.

Elected candidates in PCA space:
Cells 54-57.

Distances between elected candidates:
Cells 58-62.

Internal party disagreement:
Cells 63-66.

Conclusion and limitations:
Cells 67-68.
```

---

# 18. A6 In 30 Seconds

```text
Assignment 6 explores Danish candidate test responses using unsupervised learning.
There is no target variable; party is used for interpretation, not prediction.

I use the 49 political answer columns as features.
First, I map technical question IDs to readable question text, then I standardize the response data.

I use PCA to reduce the 49 answers to two principal components.
The first two PCs explain about 52% of the variance, so the PCA map is useful but incomplete.

Then I compare clustering methods.
K-Means with two clusters is the most defensible result based on silhouette score.
Five clusters are exploratory, hierarchical clustering supports broad party blocs, and DBSCAN is less suitable because the data is continuous rather than density-separated.

Finally, I project elected candidates into the same PCA space and use distances to compare their questionnaire similarity.
```

---

# 19. Emergency Speaking Pattern

If you forget a formal definition, use this pattern:

```text
[Term] means [simple meaning].
In my assignment, I used it for [specific thing].
The reason is [why].
```

Example:

```text
PCA means reducing many features to fewer dimensions.
In my assignment, I used it to reduce 49 political answers to PC1 and PC2.
The reason is that this makes the political landscape easier to visualize and interpret.
```

---

# 20. Top Words To Memorize

```text
unsupervised learning = no target label
metadata = descriptive columns like party/name/age
feature = one political answer column
scaling = putting features on comparable scale
PCA = dimensionality reduction
principal component = new axis made from original features
loading = contribution of original question to PC
explained variance = how much information a PC captures
K-Means = centroid-based clustering
k = number of clusters
silhouette score = cluster quality measure
DBSCAN = density-based clustering
eps = DBSCAN neighbourhood radius
hierarchical clustering = clustering shown as dendrogram
dendrogram = tree of cluster merges
Euclidean distance = distance between answer profiles
transform = place data into existing PCA space
```

Final survival sentence:

```text
The key point in this assignment is that I use standardized candidate answer profiles, reduce them with PCA for interpretation, and compare clustering methods to explore political structure without treating clusters as absolute truth.
```
