# LDA Evaluation Scripts  

## Overview  

This script is used to test an LDA model which has been trained. This is useful to be able to compare various models and ensure that the model meets requirements needed. With the LDA training script there are some initial evaluations which may be enough for model comparison, but this script dives deeper.  

## Inputs  

* **Model Directory** - Select the directory which the model and everything is saved in. It should contain the model, corpus, dictionary, where to save output files  
* **Number of Topics** - Enter the Number of topics the model was trained on, as an integer  
* **File Path for Sample Text** - Select a directory for a collection of text files which the script should evaultate the model to.  

## Outputs  

* **csv File** - csv file with all of the metrics saved in comparison to the text files the model was evaluated on
* **csv Sample**
![alt text](image.png)  

## Metrics Explained  

* Dominate Topic Probability - This is the highest percentage which a topic contributes to the documents, close to 1.0 is good for one topics but lower values such as 0.3-0.5 mean multiple topics contribute  
* KL Divergence - This is the deviation from the nominal topics distribution, range of roughly (3.0 to 0.5 for short texts), a higher value means a few topics dominate and a lower means multiple topics are used  
* Perplexity - This is the measurement of how well the model predicts unseen words. Looking for a generally lower number (more negative explains the model explains text well)  this does not correlate directly to human readability  
* Coherence u_mass - Measurement the semantic similarity of high-probability works within each topic, generally how interpretable the topic is. The closer to zero the more coherent topics. (usually negative)
* Entropy - This measures the uncertainty of text to model and how spread out topics are. Low number is focused a few topics and high number is covering many topics  
* Max Topic Probability - This is the single largest topics which dominates the topic
* Gini Index - A measure of inequalities among the topic probabilities, how concentrated vs spread out the topics are. 0 is dominated by one topic and 1 is evenly spread topics.  
* Number of Effective Topic - The number of topics that meaningfully contribute to the document, 0 is one dominant topic and 1.6 ~ 5-6 topics  
* Topic Diversity - Measurement of unique top words in the topics, if the topics are unique or repeated. Looking for non repeated topics  
* Topic # - Explains the weight each topic has on the document.
