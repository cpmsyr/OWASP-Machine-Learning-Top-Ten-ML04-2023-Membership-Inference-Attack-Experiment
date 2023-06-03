# privacy_preservation
This projects purpose is to experiment with OWASP Machine Learning Top Ten - "ML04:2023 Membership Inference Attack" as a proof of concept and also show 
how differential privacy is supposed to work in Tensorflow. 

The code was trained with three saved models. 
-The first model is a straight forward classification model on the data set. 
-The second model is trained leaving out the target classification group so a delta in confidence scores can be derived. 
-The third model experiments with the TensorFlow differential privacy module to see it's impact on accuracy and a malicious actors ability to conduct membership infrerence attacks. 

#please see my youtube channel for a complete overview of the project and how to work with the code. 


