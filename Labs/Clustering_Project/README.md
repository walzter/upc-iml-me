These are the objectives: 

- Load & Read the .arff files 
- Implement OPTICS from sklearn
	- Test 3 different distance metrics 
		- Euclidean, Cosine, L1/L2 (Lasso/Ridge)
- Implement own K-Means
- Implement one of the following: 
	- K-Modes
	- M-Medoids
	- K-Prototypes
- Implement Fuzzy Computing:
	- Fuzzy C-Means (FCM)
	- Possibilistic C-Means (PCM)
- Analyze the algorithms on 3 datasets: 
	- 2 of them should be mixed dtypes (numerical, categorical)
	- 1 of them with all numbers

-------------------------------------------
So far the plan is to: 
- make a dataframe with the number of (cat, num) cols in each dataset [x]
- preprocess all the dataframes (appropriate methods for each dtype).
- use OPTICS on the datasets and test different distance metrics.

