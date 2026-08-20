README 

author: Bryce Shiver 
email: beshiver@crimson.ua.edu
The University of Alabama Libraries 

---------lda_model_training-------------

Overview: 
This script is used to train a Latent Dirichlet Allocation (LDA) model, which can be used to analyze texts in various forms. Usually to identify hidden associations and catergorize texts into various topics. This script uses a wikipedia database of 4.9 million articles to build this lda model. This is a general model training for the english language, it is being used for sorting newspaper articles into topics. This uses the gensim module for tokenization and lda model training. The LDA is controlled through the GUI and the parameters listed, more is explained about parameters below. 

Steps: 
1. Build Corpus and Dictionary (Build Corpus Button)
2. Input Parameters & Train LDA Model (Train LDA Button)
3. Evaluate Model using text file and plots
4. Use Model or Rerun 

Inputs/GUI: 
Model Directory = File Path where the model, corpus, logging, evaluations, and plots will save 
Number of Topics = Number of topics the lda will be trained to, reccommended is around 50 - 300. A list can also be input and a seperate model will be trained for each value in list 
Chunk Size = Number of documents (corpus) loaded in to train the model at one time 
Passes = How many times the model is trained on the corpus, the model will train through with whatever value is given 
Iterations = How often a certain loop is repeated 
Update Every = How many chunks of the corpus are loaded in before the model is updated 
*refer to parameter explaination below* 


Outputs: 
evaluation file
coherence comparison plot 
logging file 
lda model 
corpus 
dictionary 

Important: 
-The corpus must be created first, if corpus is made at a earlier time it must keep the naming convention 
-Gensim currently uses Numpy <2.0 so a python enviroment may be needed if on newer versions 
-The coherence score and the convergence are rather important, decreasing convergence score is good and a coherence score (u_mass) lower is better (generally not my strong suit)

LDA Parameters Explained: 

Number of Topics - This controls how many topics the lda has, it is an effect on topic coherence in general. As the number of topics increases too much it leads to improper and confusing topics. The increase in topics # also increasing training time. 

Chunk Size - This controls the number of documents the model is trained on at once. The chunk size directly effects the computing ram needed so if ram limited than decreasing chunk size may be useful. As for the model itself the chuck size effects the coherence of the model and usually helps to develop more coherent topics with larger chunks (in my exp). Increasing chunk size will decrease training time, the goal is to balance chunk size between ram use, coherence, and time. 

Passes - This controls how many times the model is trained on the corpus. As this increases it increases the time like a multipler directly. Increasing also increase model coherence, model convergence, and human readalibity for topics. Try to increase this slightly until you reach maximum time wanted because the more passes over corpus the model will be better. 

Iterations - This is a diffcult metric to explain but important with convergence of the model and the understanding of topics, keep around 50 - 100. Used mostly in the actual calculations of LDA. 

Update Every - This works in tandem with chunk size, how many chucks the training session is run on. So a update every 2 at 100 chucks is a the same as a 200 chunk size. This can help reduce ram usage with chunk size and decrease training time



Functions: 
setup_logging - used to log everything running into console and a text file

preprocess - handoff function for the gensim token tokenization of text can be altered by changing the tokenizing filters

build_corpus - the function to load in the wiki text, parse through, tokenize, and then build both the dictionary and corpus. If needed the dictionary can be filtered but seemed to work fine with the full dictionary for me, filtering reduces word count and therefore reduces time to train 

train_model - calls the gensim lda model trainer and runs the function, is fed by the parameters input on the GUI, more is mentioned in Parameters. 

eval - this is a function to evaluate a trained lda model and return some metrics of the model, such a corpus length, dictionary length, # of corpus tokens, displays the topics in the lda model, and evaluate the coherence of the model. All of these metrics are written to a text file saved in the same directory as the lda model. This also include a html file which has a lda visualization item which can be used to compare the models. 

eval_plot - a function to plot the coherence scores vs number of topics for the lda model, useful when comparing the various number of topics in one training session. 

sample_corpus_streaming - function used to stream the corpus (which has already been built) to the evaluation metrics. The streamed corpus allows for the process to run much faster than loading in the entire corpus and also allowed my pc to not run out of ram. 

LDAGui (Class) - this handles all of the GUI, parameter inputs, and control of running said processes when the buttons are pushed. Generally straight forward as it runs both the build_corpus and train_model as a thread that can be monitored. The _build_corpus_thread is the button on the GUI to build the corpus and handles such. The _train_lda_thread is used for the button on the GUI controlling the LDA training. 

Misc: 
 - There is a sample corpus function which can be commented out to test out the process without the lengthly time, please comment out the main function if needed to test 
 - There is a logging text file and evaluation text file that are important and can be used to understand LDA training better
 - Took my pc 28hrs to generate a [5,10,20,50,75,200] number of topic models


Details to Read More on LDA models: 
https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
https://radimrehurek.com/gensim/wiki.html
https://miningthedetails.com/blog/python/lda/GensimLDA/
https://groups.google.com/g/gensim

