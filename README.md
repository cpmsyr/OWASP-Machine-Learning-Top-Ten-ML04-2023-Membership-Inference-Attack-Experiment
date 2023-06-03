# privacy_preservation
This projects purpose is to experiment with OWASP Machine Learning Top Ten - "ML04:2023 Membership Inference Attack" as a proof of concept and also show 
how differential privacy is supposed to work in Tensorflow. 

The code was trained with three saved models.

1. The first model is a straight forward classification model on the data set. 
2. The second model is trained leaving out the target classification group so a delta in confidence scores can be derived. 
3. The third model experiments with the TensorFlow differential privacy module to see it's impact on accuracy and a malicious actors ability to conduct membership infrerence attacks. 

#Please see my youtube channel for a complete overview of the project and how to work with the code. 


