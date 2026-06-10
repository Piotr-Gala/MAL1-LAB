# Assignment 6: Candidate Test 2022 Analysis Part 2 - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with Danish candidate test data from the 2022 election.
The goal was to explore the political landscape of candidates using unsupervised machine learning.

The data contains answers from DR and TV2 candidate tests.
Each answer is on a scale from -2 to 2, so the features represent political agreement or disagreement with different statements.

I started by loading the candidate datasets and the question metadata.
Then I cleaned simple metadata issues, separated response features from metadata, and mapped technical question IDs back to readable question text.

For the main analysis, I standardized the response features and used PCA to reduce 49 political questions to two main dimensions.
I interpreted the PCA axes using the strongest loadings, meaning the questions that contributed most to each component.

Then I compared clustering methods: K-Means, hierarchical clustering and DBSCAN.
The main result is that candidates form broad ideological blocs rather than one clean cluster per party.

Finally, I analyzed elected candidates separately by placing them in the same PCA space and measuring which elected candidates agreed or disagreed the most based on their questionnaire answers.
```

## Cells 0-3: Title And Imports

```text
At the beginning I import the libraries needed for the analysis.
Pandas and NumPy are used for tabular data and numerical operations.
Matplotlib and Seaborn are used for visualization.
Scikit-learn is used for scaling, PCA, K-Means, DBSCAN and silhouette score.
SciPy is used for hierarchical clustering and pairwise distances.
```

## Cells 4-8: Load Datasets

```text
Here I load the data from the local data folder.
The main dataset contains all candidates, and the elected dataset contains the candidates who were elected to parliament.
I also load the DR and TV2 question files, because the response columns use technical IDs.

After loading, I check the shapes and preview the candidate and question tables.
This is a sanity check to confirm that the files loaded correctly and that I understand the structure.
```

## Cells 9-13: Party Colors And Feature Preparation

```text
I define a color dictionary for parties so that the same party has the same color in all plots.
Then I prepare the data for analysis.

I treat name, party, region and age as metadata.
The remaining 49 columns are the response features used for PCA and clustering.

I replace age values equal to zero with missing values, because age zero is not meaningful for candidates.
The response data itself is already clean: there are no missing response values and all responses are within the expected range from -2 to 2.
```

## Cells 14-19: Question Mapping

```text
In this section I map technical response column names back to the actual political questions.
I create one mapping for DR questions and one mapping for TV2 questions.
Then I combine them into one question map ordered like the response features.

This is important because PCA and clustering work with numbers, but for interpretation I need to know which political questions those numbers represent.
Without this mapping, I could say that feature 530 is important, but I could not explain what political topic it refers to.
```

## Cells 20-23: PCA Political Landscape

```text
Before PCA I standardize the response features.
Even though all answers use the same -2 to 2 scale, some questions may still have higher variance than others.
Scaling prevents those high-variance questions from dominating PCA and clustering.

Then I fit PCA with two components and create a two-dimensional political map.
Each point is a candidate.
Candidates close to each other gave similar answer patterns.

The first two principal components explain about 52 percent of the total variance.
So the PCA plot captures an important part of the political structure, but it is still only a simplified two-dimensional view.
```

## Cells 24-28: PCA Axis Interpretation

```text
To interpret the PCA axes, I calculate loadings.
Loadings show how strongly each original question contributes to PC1 and PC2.

I focus on the questions with the largest absolute loadings, because they define the axes most strongly.
This is how I avoid guessing the meaning of the PCA plot only by looking at party positions.

PC1 is mainly connected to welfare, inequality, redistribution and climate-related questions.
PC2 is more connected to issues such as EU, immigration, defence and specific political controversies.

The sign of a PCA component is arbitrary.
So I interpret positive and negative directions as opposite sides of an axis, not as inherently good or bad.
```

## Cells 29-33: Average Party Positions By Question

```text
Here I calculate average party responses for each question.
Then I select important questions from the PCA loadings and plot average party positions for those questions.

This helps validate the PCA interpretation.
If a question has a strong loading, the party average plot should show meaningful political differences between parties.

I do not plot all 49 questions because that would make the notebook too noisy.
Instead, I focus on the questions that are most important for the PCA axes.
```

## Cells 34-37: K-Means Cluster Selection

```text
In this section I test K-Means with different numbers of clusters.
For each value of k, I calculate the silhouette score.

The silhouette score measures how compact and separated the clusters are.
The best score is for k equal to 2.

This means that the strongest natural structure in the data is a broad two-group split.
It does not support one clean cluster per party.
```

## Cells 38-41: K-Means With Two Clusters

```text
I fit K-Means with two clusters because this was the best choice according to silhouette score.
Then I compare the cluster labels with party membership using a crosstab.
I also plot the clusters on the PCA map.

The result shows that the clusters mostly follow PC1.
This supports the interpretation that the main structure is a broad ideological division rather than exact party separation.
```

## Cells 42-45: K-Means With Five Clusters

```text
I also try K-Means with five clusters as an exploratory analysis.
This gives a more detailed view of possible subgroups inside the broad two-bloc structure.

However, I do not present five clusters as the best solution.
The silhouette score is lower than for two clusters, so five clusters are useful for interpretation but less defensible as the main clustering result.
```

## Cells 46-48: Hierarchical Clustering Of Parties

```text
Here I use hierarchical clustering on average party profiles.
This is party-level clustering, not candidate-level clustering.

Each party is represented by its average response vector across the 49 questions.
The dendrogram shows which parties have similar average response patterns.

The result again supports the idea of two broad political blocs.
This is consistent with the PCA and K-Means results.
```

## Cells 49-53: DBSCAN Clustering

```text
In this section I test DBSCAN with different epsilon values.
DBSCAN looks for dense groups of points separated by lower-density areas.

The results are not stable.
With small epsilon values, many candidates are classified as noise.
With larger epsilon values, almost all candidates merge into one cluster.

This suggests that the candidate responses form a continuous political landscape rather than clearly separated dense groups.
So DBSCAN is less suitable for this dataset.
```

## Cells 54-57: Elected Candidates Political Landscape

```text
Now I focus only on elected candidates.
I use the already fitted scaler and PCA model to transform elected candidates into the same PCA space.

This is important because I want the elected candidate map to be comparable to the earlier map of all candidates.
If I fitted PCA again only on elected candidates, the axes could change.

The plot shows that elected candidates form visible party-based regions, but parties are not perfectly separated.
```

## Cells 58-62: Agreement And Disagreement Among Elected Candidates

```text
Here I calculate pairwise Euclidean distances between elected candidates.
The distances are computed using standardized responses across all 49 questions.

A small distance means that two elected candidates answered the candidate test similarly.
A large distance means their answer patterns are very different.

The most similar pairs are often from the same party.
The most different pairs are mainly between left-wing parties and Liberal Alliance.

This should be interpreted as agreement in questionnaire responses, not necessarily personal agreement or parliamentary cooperation.
```

## Cells 63-66: Internal Disagreement Within Parties

```text
In this section I measure internal disagreement within parties.
For each party with at least two elected candidates, I calculate pairwise distances between candidates from the same party.

The mean internal distance shows typical disagreement within the party.
The maximum internal distance shows the largest internal gap.

Parties with higher internal distance have elected candidates who are more spread out in their questionnaire answers.
These results should be interpreted carefully for small parties, because a few candidates can strongly affect the average.
```

## Cells 67-68: Conclusion And Limitations

```text
The main conclusion is that the candidate responses mostly form broad ideological blocs.
The data does not naturally split into one clean cluster per party.

PCA, K-Means and hierarchical clustering all support this broad-bloc interpretation.
DBSCAN is less suitable because the data looks continuous rather than density-separated.

The limitations are important.
PCA only shows a two-dimensional approximation.
The first two components explain about half of the variance, so some information is lost.
Clustering results also depend on preprocessing choices and parameters.
Finally, the elected candidate dataset does not contain every elected member, so it is not a complete representation of parliament.
```

## PCA Speech

```text
PCA is used here to reduce 49 political response variables into two main dimensions.
It finds directions of maximum variance in the standardized response data.

PC1 explains the largest amount of variation.
PC2 explains the second largest independent amount of variation.

I interpret the axes using loadings.
The questions with the highest absolute loadings contribute most to each component.

The PCA map is useful because it gives a visual overview of the political landscape.
However, it is only an approximation because the first two components explain about 52 percent of the total variance.
```

## Clustering Speech

```text
I compare three clustering approaches.

K-Means is useful when I want to test a fixed number of clusters.
The silhouette score suggests that two clusters are the most defensible solution.

Hierarchical clustering is used at party level.
It shows which parties have similar average response profiles.

DBSCAN tries to find dense separated groups.
It does not work very well here because the data forms a continuous political space.

Together, these methods suggest that the strongest structure is a broad ideological split, not one separate cluster per party.
```

## Limitations Speech

```text
I would be careful not to overinterpret the plots.
PCA reduces 49 questions to only two dimensions, so some information is lost.

The clustering results depend on preprocessing and parameter choices.
For example, K-Means depends on k, and DBSCAN depends heavily on epsilon.

Scaling also has a trade-off.
It gives each question equal weight, but it removes the natural influence of questions with higher variance.

Finally, the elected candidate dataset is incomplete because not every elected member participated in the candidate tests.
So the elected-candidate analysis should be treated as exploratory rather than a full description of parliament.
```

## Final Sentence

```text
Overall, this analysis shows that candidate test responses are useful for visualizing broad political structure, but the results should be interpreted as exploratory patterns rather than exact political categories.
```
