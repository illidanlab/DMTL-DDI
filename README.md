# Identifying Drug-Drug Interactions via Deep Multi-Task Learning

## Abstract

Adverse drug-drug interactions (DDIs) remain a leading cause of morbidity and mortality around the world. Computational approaches that leverage the available big data to identify potential DDIs can greatly facilitate pre-market targeted clinical drug safety testings as well as post-market drug safety surveillance. However, most existing computational approaches only focus on binary prediction of the occurrence of DDIs, i.e., whether there is an existing DDI or not. Prediction of the actual DDI types will help us understand specific DDI mechanism and identify proper early prevention strategies. In addition, the nonlinearity and heterogeneity in drug features are rarely explored in most existing works. In this paper, we propose a deep learning model to predict fine-grained DDI types. It captures nonlinearity with nonlinear model structures and through layers of abstraction, and resolves the structural heterogeneity via projecting different subsequences of fingerprint features to the same hidden states that indicate a particular interaction pattern. Moreover, we proposed a multi-task deep model to perform simultaneous prediction for a focused task (i.e. a DDI type) along with its auxiliary tasks, thus to exploit the relatedness among multiple DDI types and improve the prediction performance. Experimental results demonstrated the effectiveness and usefulness of the proposed model. 

## This site

This website includes the algorithm implementation and experimental data. 
