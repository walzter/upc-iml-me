This is going to be the summary for the IBL project: 

It will contain information exactly from the Work 3. Lazy Learning Excercise document. 

*Structure* 

IBL
- data -> contains all the dataset with the 10-fold CV 
- Resources -> all the papers needed to understand this project
- Resources/Summary --> Summaries for the papers (important). 

--> Rest of the structure will be added as it is being built <-- 


*Random Info* 
Lazy Learning --> Deffering majority of the computation to the inference section and not the training. 


*Steps*
1. Load the data, with the classes TrainMatrix, TestMatrix
	- Normalize the values in [0,1]
	- Most used representation is a flat structure 
	- Feature vector ( set of attribute-value pair AKA feature-value pair)
2. Write a py function which does this for all 10-fold cross-validation files.
3.1. Implement the IBL algorithm: 
	- IB1, IB2 and IB3
	- Hyperparameters / Fine-tuning --> We chose & Justify why we use them. 
	- PERFORMANCE: 
		- Correctly Classified Instances (Confusion Matrix) --> pickle
		- Efficiency (Time and Memory consumption of each algorithm). 

3.2. We will use the Euclidean similarity measure (we can also test different ones). 
	- Identify the best one based on correctly classified and efficiency. 
	- There most-likely will be a trade-off between the two 
	- Justify why we chose one over the other. 

4.1. From the best IBL algorithm, write a py function using a K-NN algorithm. Which will then return the k most similar instances. 
	- Justify the implementation and all the references used for this. 
	- We can test different distance metrics: Euclidean, Manhattan, Cosine, Clark, Canberra, HVDM or IVDM. We should adapt these as neccesary whenever needed. 
	- The kIBLAlgorithm returns the k most similar cases to q (our unknown sample)
	-> Basically spitting out which values are most similar based on a distance formula. 
	- We also need to include a voting method. And how to break ties, if there are any. 

4.2. At this point we would have a K-IBL Algorithm with: 
	- Four similarity measures 
	- Different values for the k-parameter 
	- three policies for deciding the solution to q. 

5. Modify the k-IBL algorithm such that it includes a preprocessing for selecting the best set of features from the training set. 
	- "SelectionkIBLAlgorithm"
	- Select two algorithms (Filter or Wrapper)	
	- From there we will select the most important features. 

6. Compare the results from SelectionkIBLAlgorithm to the kIBLAlgorithm in terms of classification accuracy. 

==============================================================================

- We need to vectorize the entire computation as well as use numba to see if we can increase the processing. 
- Take advantage of multiprocessing and threading if possible. 

