Paper: Deep Kalman Filters (https://arxiv.org/abs/1511.05121) 
Deep Markov Models: https://github.com/clinicalml/dmm
Structured Inference Networks for Nonlinear State Space Models: https://github.com/clinicalml/structuredinference
re-implementation and test on paper Deep Kalman Filter: https://github.com/GalaxyFox/DS-GA-3001-Deep_Kalman_Filter

Other Filter methods:
 1. State Space Models and Imputation via Expectation Maximization
 2. Particle Filter (Sequential Monte Carlo Methods) - https://aleksandarhaber.com/clear-and-concise-particle-filter-tutorial-with-python-implementation-part-3-python-implementation-of-particle-filter-algorithm/#google_vignette 
 3. Wiener Filter
 4. Moving Average Filter
 5. Linear and spline interpolation
 6. Kalman filters: https://medium.com/@ab.jannatpour/kalman-filter-with-python-code-98641017a2bd 

​
Summer Research:

These are control chart pattern recognition papers that I published with my students and PhD advisor :

https://www.sciencedirect.com/science/article/pii/S0360835214000308
https://www.sciencedirect.com/science/article/abs/pii/S0957417424005487 (we used Cytovance data as a use case)
https://www.sciencedirect.com/science/article/pii/S0957417420301007

Here is the literature review paper on online learning for time series data.
https://www.sciencedirect.com/science/article/abs/pii/S0925231221006706


Feature Ranking methods: Border Count Ranking.


Outline fpr Paper:

Introduction: 
	talk about why the problem is important (give statistics and literature)
	Literature revoew for using ML for Biomanufacturing
		Pros/Cons of works
		gaps
	what you propose/suggest to fill in the Gap - Novelty)
	Organization of the paper.

Data:
	data size/features
	Data preparation (deal with missing values)
	data visualization/types of patterns
		- Correlaton mstrices
		- missing data

Methodology:
	SVM
	Kalman filters
	.
	.
	Performance metrics

Results/Discussion:
	- Train/Test split
	- CV-fold for hyperparameter tuning
	- Results -> comparative analysis based RMSE (figures/tables)
	- Discuss the findings

Conclusion
	- Summarise your findings
	- Future directions