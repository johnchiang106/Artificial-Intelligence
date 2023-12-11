CS440/ECE448 MP1

![](clinton-word-cloud.png)

CS440/ECE448 Fall 2022  
MP1: Naive Bayes
-----------------------------------------

### Due date: Monday September 11th, 11:59pm

Suppose that we're building an app that recommends movies. We've scraped a large set of reviews off the web, but (for obvious reasons) we would like to recommend only movies with positive reviews. In this assignment, you will use the Naive Bayes algorithm to train a binary sentiment classifier with a dataset of movie reviews. The task is to learn a bag of words model that will classify a review as positive or negative based on the words it contains.

General guidelines
------------------

This assignment is written in **Python 3**. You should be using **Python version 3.8** for all assignments in this course, due to compatability with Gradescope's autograder. If you have never used Python before, a good place to start is the [Python tutorial](https://docs.python.org/3/tutorial/).

Your code may import modules that are part of the [standard python library](https://docs.python.org/3/library/). You should also import nltk, numpy, and tqdm. You can install modules locally using the [**pip3**](https://pip.pypa.io/en/stable/) tool.

We will load nltk so that we can use its tokenizer and the Porter Stemmer. You may not use any other utilities from the nltk package unless you get permission from the instructor. More information about nltk can be found at the [NLTK web site](https://www.nltk.org/).

The tqdm package provides nice progress displays for the main loops in the algorithm, because they can be somewhat slow when handling this amount of data.

For general instructions, see the [main MP page](../index.html) and the [syllabus](../../syllabus.html).

Provided Code
-------------

We have provided a ([zip package](template.zip)) containing all the code to get you started on your MP, as well as the training and development datasets.

Submit only your **naive\_bayes.py** file on gradescope.

The main program mp1.py is provided to help you test your program. The autograder does not use it. So feel free to modify the default values of tunable parameters (near the end of the file). Do not modify reader.py.

To run the main program, type **python3 mp1.py** in your terminal. This should load the provided datasets and run the naiveBayes function on them. Sadly, it's not doing the required training and just returns the label -1 for all the reviews. Your job is to write real code for naiveBayes, returning a list of 0's (Negative) and 1's (Positive).

The main program mp1.py accepts values for a number of tunable parameters. To see the details, type **python3 mp1.py -h** in your terminal. Note that you can and should change the parameters as necessary to achieve good performance.

Submitting to Gradescope
------------------------

Submit this assignment by uploading `naive_bayes.py` to Gradescope. You can upload other files with it, but only `naive_bayes.py` will be retained by the autograder. We strongly encourage you to submit to Gradescope early and often as you will be able to see your final score there.

Policies
--------

You are expected to be familiar with the general policies on the course syllabus (e.g. academic integrity) and on the top-level MP page (e.g. code style). In particular, notice that this is an individual assignment.

Dataset
-------

The dataset in your template package consists of 10000 positive and 3000 negative movie reviews. It is a subset of the [Stanford Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which was originally introduced by [this paper](https://www.aclweb.org/anthology/P11-1015). We have split this data set for you into 5000 development examples and 8000 training examples. The autograder also has a hidden set of test examples, generally similar to the development dataset.

Background
----------

The bag of words model in NLP is a simple unigram model which considers a text to be represented as a bag of independent words. That is, we ignore the position the words appear in, and only pay attention to their frequency in the text. Here, each review consists of a group of words. Using Bayes theorem, you need to compute the probability of a review being positive given the words in the review. Thus you need to estimate the posterior probabilities: \\\[ P( \\mathrm{Type} = \\mathrm{Positive} | \\mathrm{Words}) = \\frac{P(\\mathrm{Type}=\\mathrm{Positive})}{P(\\mathrm{Words})} \\prod\_{\\mathrm{All}~\\mathrm{words}} P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{Positive}) \\\] \\\[ P( \\mathrm{Type} = \\mathrm{Negative} | \\mathrm{Words}) = \\frac{P(\\mathrm{Type}=\\mathrm{Negative})}{P(\\mathrm{Words})} \\prod\_{\\mathrm{All}~\\mathrm{words}} P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{Negative}) \\\]

Notice that P(words) is the same in both formulas, so you can omit it (set term to 1).

Unigram Model
-------------

**Training Phase:** Use the training set to build a bag of words model using the reviews. Note that you will already be provided with the labels (positive or negative review) for the training set and the training set is already pre-processed for you, such that the training set is a list of lists of words (each list of words contains all the words in one review). The purpose of the training set is to help you calculate \\(P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{positive})\\) and \\(P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{negative})\\) during the testing (development) phase.

For example \\(P(\\mathrm{Word = tiger}|\\mathrm{Type}=\\mathrm{positive})\\) is the probability of encountering the word "tiger" in a positive review. After the training phase, you should be able to quickly look up \\(P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{positive})\\) and \\(P(\\mathrm{Word}|\\mathrm{Type}=\\mathrm{negative})\\) for any word (whether or not it was in your training data).

**Development Phase:** In the development phase, you will calculate the \\(P(\\mathrm{Type}=\\mathrm{positive}|\\mathrm{Words})\\) and \\(P(\\mathrm{Type}=\\mathrm{negative}|\\mathrm{Words})\\) for each review in the development set. You will classify each review in the development set as a positive or negative review depending on which posterior probability is of higher value. You should return a list containing labels for each of the reviews in the development set (label order should be the same as the document order in the given development set, so we can grade correctly). Note that your code should use only the training set to learn the individual probabilities. Do not use the development data or any external sources of information.

The prior probability \\(P(\\mathrm{Type}=\\mathrm{Positive})\\) is provided as an input paramater. You can adjust its value using the command-line options to mp1.py. Inspect the development dataset to determine the actual distribution of reviews in the development data. **Adjust your definition of naiveBayes so that the default value for pos\_prior is appropriate for the development dataset.** Our autograder tests will pass in appropriate values for our hidden tests. \\(P(\\mathrm{Type}=\\mathrm{Negative})\\) can be computed easily from \\(P(\\mathrm{Type}=\\mathrm{Positive})\\).

Making the details work
-----------------------

Consider Python's Counter data structure.

**Use the log of the probabilities to prevent underflow/precision issues.** Apply log to both sides of the equation and convert multiplication to addition. Be aware that the standard python math functions are faster than the corresponding numpy functions, when applied to individual numbers.

Zero values in the naive Bayes equations will prevent the classification from working right. Therefore, you must smooth your calculated probabilities so that they are never zero. In order to accomplish this task, use Laplace smoothing. See the lecture notes for details. The Laplace smoothing parameter \\(\\alpha\\) is passed as an argument to naiveBayes and you can adjust its value using the command-line arguments to mp1.py.

Tune the values of the Laplace smoothing constant using the command-line arguments to mp1.py. When you are happy with the result on the development set, edit the default values for these parameters in the definition of the function naiveBayes. Some of our tests will use your default settings and some tests will pass in new values.

You can experiment with other methods that might (or might not) improve performance. The command line options will let you transform the input words by converting them all to lowercase and/or running them through the Porter Stemmer. If you wish to turn either of these on for your autograder tests, edit the default values in the function load\_data.

You could also try removing stop words from the reviews before you process them. You can add this to load\_data or to the start of your naiveBayes function. You will need to find a suitable list of stop words and write a short python function to modify the input data.

No guarantees about what changes will make the accuracy better or worse. You need to figure that out by experimenting.
