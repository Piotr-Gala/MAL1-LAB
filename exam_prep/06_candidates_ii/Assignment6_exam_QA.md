# Assignment 6: Candidates II - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment6.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This assignment uses unsupervised learning. The goal is not to predict a known label, but to explore structure in candidate questionnaire answers using dimensionality reduction and clustering.

**How used in the assignment:** The notebook loads candidate answers, standardizes response features, uses PCA to reduce 49 answer dimensions to two principal components, and compares K-Means, DBSCAN, and hierarchical clustering.

## 1. Supervised vs Unsupervised Learning

**Question:** Why is this unsupervised learning?

**Theory answer:** In supervised learning, the model learns from labels. In unsupervised learning, there is no target label for prediction; the goal is to find patterns, groups, or lower-dimensional structure.

**How used in the assignment:** Party labels are mainly used for interpretation and plotting, not as the target for training PCA or clustering.

## 2. High Dimensionality

**Question:** Why is high dimensionality difficult?

**Theory answer:** High-dimensional data is hard to visualize, distances can become less meaningful, and models can become harder to interpret. Dimensionality reduction helps summarize structure.

**How used in the assignment:** Candidate answers contain 49 response variables. PCA projects them into two dimensions for visualization and interpretation.

## 3. Scaling

**Question:** Why standardize before PCA and clustering?

**Theory answer:** PCA and distance-based clustering are sensitive to feature scale. Standardization gives features mean 0 and standard deviation 1, so no feature dominates only because of its numeric scale.

**How used in the assignment:** The notebook uses `StandardScaler` before PCA, K-Means, DBSCAN, and distance calculations.

## 4. PCA

**Question:** What is PCA?

**Theory answer:** PCA is a dimensionality reduction method that finds new orthogonal axes called principal components. The first component captures the most variance, the second captures the next most variance under the constraint of being orthogonal to the first.

**How used in the assignment:** PCA reduces 49 political answer features to PC1 and PC2, allowing a two-dimensional plot of candidates.

## 5. Principal Components

**Question:** What does a principal component mean?

**Theory answer:** A principal component is a linear combination of original features. It is not one original question, but a weighted direction through the feature space.

**How used in the assignment:** The notebook examines PCA loadings to interpret which political questions contribute most to PC1 and PC2.

## 6. Orthogonality

**Question:** Why are principal components orthogonal?

**Theory answer:** PCA chooses components that are uncorrelated directions. Orthogonality means each component captures different variance instead of repeating the same direction.

**How used in the assignment:** PC1 and PC2 summarize different patterns in candidate responses, which makes the 2D plot more informative.

## 7. Explained Variance

**Question:** What is explained variance?

**Theory answer:** Explained variance tells how much of the original data variation is captured by each principal component. The explained variance ratio expresses this as a percentage.

**How used in the assignment:** The notebook reports explained variance for PC1 and PC2 and treats the 2D PCA plot as a useful simplification, not a perfect representation.

## 8. Choosing Number Of PCs

**Question:** How do you choose the number of principal components?

**Theory answer:** Common methods include keeping enough PCs to capture a chosen percentage of variance, using the elbow/shoulder method, or choosing two PCs for visualization.

**How used in the assignment:** The notebook uses two PCs mainly for visualization and interpretation of the political landscape.

## 9. PCA Limitations

**Question:** What is lost in PCA?

**Theory answer:** PCA is lossy when we keep fewer components than original dimensions. Some variance and detail are discarded.

**How used in the assignment:** The 2D plot is an approximation of 49-dimensional candidate answers. It is useful for overview, but not the full truth.

## 10. Clustering

**Question:** What is clustering?

**Theory answer:** Clustering groups observations based on similarity without using target labels. Different algorithms define similarity and clusters differently.

**How used in the assignment:** The notebook clusters candidates or party profiles to see whether political answer patterns form natural groups.

## 11. K-Means

**Question:** How does K-Means work?

**Theory answer:** K-Means chooses `k` centroids, assigns each point to the nearest centroid, recomputes centroids, and repeats until assignments stabilize.

**How used in the assignment:** The notebook tests K-Means with different `k` values and compares cluster quality using silhouette score.

## 12. Choosing k

**Question:** How do you choose `k` in K-Means?

**Theory answer:** `k` can be chosen using domain knowledge, inertia/elbow method, or silhouette score. There is no single automatic truth.

**How used in the assignment:** The notebook selects `k = 2` based on silhouette score and also tests `k = 5` as an exploratory political grouping.

## 13. Silhouette Score

**Question:** What is silhouette score?

**Theory answer:** Silhouette score measures how similar a point is to its own cluster compared with other clusters. Higher values usually mean better separated clusters.

**How used in the assignment:** The notebook uses silhouette score to compare K-Means cluster choices.

## 14. DBSCAN

**Question:** How does DBSCAN work?

**Theory answer:** DBSCAN is density-based clustering. It groups dense regions and marks sparse points as noise. It uses `eps` for neighborhood radius and `min_samples` for density.

**How used in the assignment:** The notebook tests DBSCAN with different `eps` values and observes that results can be unstable, noisy, or merged.

## 15. DBSCAN Parameters

**Question:** What happens if `eps` is too small or too large?

**Theory answer:** If `eps` is too small, many points become noise. If `eps` is too large, separate clusters can merge into one cluster.

**How used in the assignment:** The notebook compares DBSCAN outputs to show that parameter choice strongly affects clustering.

## 16. Hierarchical Clustering

**Question:** What is hierarchical clustering?

**Theory answer:** Agglomerative hierarchical clustering starts with each point as its own cluster, then repeatedly merges closest clusters. The result can be shown as a dendrogram.

**How used in the assignment:** The notebook clusters party-average profiles and visualizes the result with a dendrogram.

## 17. Linkage

**Question:** What does linkage mean?

**Theory answer:** Linkage defines how distance between clusters is calculated. Common methods include single, complete, average, and Ward linkage.

**How used in the assignment:** The hierarchical clustering section uses linkage to merge similar party profiles and interpret political closeness.

## 18. Distances Between Elected Candidates

**Question:** Why compute distances between elected candidates?

**Theory answer:** Distance measures similarity in feature space. Smaller distance means more similar answer profiles; larger distance means more disagreement.

**How used in the assignment:** The notebook computes pairwise distances among elected candidates to find who answered most similarly or differently.

## 19. Limitations

**Question:** What are the limitations?

**Theory answer:** Unsupervised results depend heavily on scaling, selected features, distance metrics, and hyperparameters. Clusters are exploratory and should not be treated as ground truth.

**How used in the assignment:** The notebook presents PCA and clustering as exploratory political-landscape analysis, not definitive party classification.

## Fast Last-Minute Answers

- **Main task:** unsupervised exploration.
- **Dimensionality reduction:** PCA from 49 answers to 2 PCs.
- **Clustering:** K-Means, DBSCAN, hierarchical clustering.
- **Key preprocessing:** standardization.
- **Best one-sentence defense:** I use standardized candidate answer profiles, reduce them with PCA for interpretation, and compare clustering methods to explore political structure without treating clusters as absolute truth.
