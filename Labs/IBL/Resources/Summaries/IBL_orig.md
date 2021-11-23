Summary for the paper for the Instance based learning algorithms IB1, IB2, and IB3. 


The goal of the project is to be able to implement these 3 algorithms and be able to measure the performance and memory consumption of each of these models. 


*Abstract:* 
- Generates classification predictions using only specific instances. 
- They do not maintain abstraction from specific instances. 
- Similar to the nearest neighbor algorithm, however this one would: 
	- Reduce the storage requirements 
	- minor sacrifices in the learning rate 
	- minor sacrifice in the classification accuracy
- Performance is affected when the level of noise is higher. 
	- Performance is proportional to the amount of noise 
	- Decrease or reduce the noise we can control the performance. 
--> That is why it was extended to distinguish noisy instances. Causing it to degrade slower when more noise is present. 


*Introduction:*
- We can use various different models in order to achieve expert dcision making. 
- IBL algs are incremental, and their goals is argmax(classification_acc) on the subsequent presented instances. 
	- Basically we want to maximize the classification accuracy when we see new instances. 
- 



