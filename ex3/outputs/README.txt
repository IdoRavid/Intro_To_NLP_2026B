Exercise 3 - HMM POS Tagger

Files
-----
- ex3.py: implementation of the models and experiments.
- outputs/results.txt: error rates for all models.
- outputs/confusion_matrix.csv: confusion matrix for the final model.
- outputs/top_confusions.txt: most frequent confusion-matrix errors.

How to run
----------
python ex3.py

Preprocessing
-------------
- Used the Brown corpus, category: news.
- Used the last 10% of sentences as the test set.
- Complex Brown tags were simplified by cutting at the first '+' or '-'.
- Unknown words are test-set words that did not appear in the training set.

Models implemented
------------------
1. Most likely tag baseline: each known word receives its most frequent training tag; unknown words receive NN.
2. Bigram HMM with MLE emission probabilities.
3. Bigram HMM with Add-one smoothed emission probabilities.
4. Bigram HMM with pseudo-words and MLE emission probabilities.
5. Bigram HMM with pseudo-words and Add-one smoothed emission probabilities.

Pseudo-word rules
-----------------
Low-frequency training words and unknown test words were mapped into categories such as:
<NUM>, <CONTAINS_DIGIT>, <HAS_HYPHEN>, <ALL_CAPS>, <CAPITALIZED>,
<ENDS_ING>, <ENDS_ED>, <ENDS_LY>, <ENDS_ION>, <ENDS_ER>, <ENDS_EST>, <ENDS_S>, <RARE>.

Results
-------
Model                                         |  Known err | Unknown err | Total err
------------------------------------------------------------------------------------
Most likely tag baseline                      |     0.0704 |      0.7500 |    0.1481
Bigram HMM MLE                                |     0.0471 |      0.7500 |    0.1274
Bigram HMM Add-one emissions                  |     0.1453 |      0.7124 |    0.2101
Bigram HMM pseudo-words + MLE                 |     0.0782 |      0.3348 |    0.1075
Bigram HMM pseudo-words + Add-one             |     0.1451 |      0.3671 |    0.1704

Most frequent errors in the final model
---------------------------------------
- true NN, predicted NP: 95 tokens
- true JJ, predicted NN: 55 tokens
- true NN, predicted JJ: 48 tokens
- true NNS, predicted NN: 47 tokens
- true NP, predicted NN: 46 tokens
- true JJ, predicted NP: 36 tokens
- true VB, predicted NN: 29 tokens
- true NNS, predicted NP: 28 tokens
- true NN, predicted AT: 27 tokens
- true VBN, predicted JJ: 27 tokens

Models explanations
------------------
Most likely tag baseline:
This model assigns each known word its most frequent tag in the training set.
For unknown words, it always predicts NN, as required in the exercise.
It performs well on known words, but very poorly on unknown words because it has no way to use word shape or context.

Bigram HMM MLE:
This model uses both emission probabilities and transition probabilities between tags.
It improves the known-word error because it uses sentence context, not only the word itself.
However, unknown-word performance stays poor because unseen words are still not handled in an informative way.

Bigram HMM Add-one emissions:
This model applies Add-one smoothing to the emission probabilities.
The goal is to avoid zero probabilities for unseen word-tag combinations.
In practice, this hurt performance because Add-one smoothing spreads probability mass too broadly and weakens useful emission patterns.

Bigram HMM pseudo-words plus MLE:
This model replaces rare training words and unknown test words with pseudo-word categories.
This gives the model useful clues from word shape, such as capitalization, digits, suffixes, and hyphens.
It gives the best total result because it greatly improves unknown-word tagging while still preserving useful HMM context.

Bigram HMM pseudo-words plus Add-one:
This model combines pseudo-words with Add-one smoothed emissions.
Although it improves unknown-word handling compared to the basic models, it performs worse than pseudo-words with MLE.
This suggests that pseudo-words are helpful, but Add-one smoothing over-smooths the emission probabilities in this setting.

Notes
-----
- Viterbi decoding was implemented manually.
- Probabilities are computed in log-space to avoid numerical underflow.
- Add-one smoothing was applied to emission probabilities.
- Transition probabilities were estimated by maximum likelihood.

Usage of AI
-----------
AI assistance was used during the work on this exercise as a supporting tool.
It was used to clarify the assignment requirements, understand how to load and process the Brown corpus with NLTK, plan the structure of the implementation, and discuss how to compute and interpret the confusion matrix.
All code and results were reviewed and tested by us before submission.