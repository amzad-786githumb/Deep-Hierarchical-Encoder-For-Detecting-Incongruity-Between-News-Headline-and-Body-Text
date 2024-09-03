# Deep-Hierarchical-Encoder-For-Detecting-Incongruity-Between-News-Headline-and-Body-Text
Google Collab link for the complete code: https://colab.research.google.com/github/amzad-786githumb/Deep-Hierarchical-Encoder-For-Detecting-Incongruity-Between-News-Headline-and-Body-Text/blob/main/major_project_final.ipynb 


Given a {headline} and {body text} pair, our goal is to determine whether a
news article contains an incongruent headline. So we calculate the output
probability as an incongruence score in our dissertation.
3.1 Baseline approaches
We present some baseline approaches that have been used to solve the
headline incongruence problem. Feature-based ensemble algorithms are
frequently utilized due to their simplicity and effectiveness. Among various
methods, the XGBoost algorithm has demonstrated superior performance
in a variety of prediction tasks (Chen and Guestrin 2016).
To begin, we implemented the XGBoost (XGB) classifier using the features
given in the winning model for the FNC-1 competition, such as cosine
similarity between the {headline} and {body text}.
3.1.1 XGBoost
XGBoost, which implements gradient boosted decision trees, is a
well-recognized and fast algorithm for classification tasks (Chen and
Guestrin 2016). We adopted XGBoost as a representative baseline
because it was used in the winning model for the stance detection
challenge in news headlines.
Extreme gradient boosting, often known as XGBoost, is a well-known
gradient boosting technique(ensemble) that improves the performance and
speed of tree-based (sequential decision trees) deep learning algorithms.
Using an incongruity label instead, we implemented the winning model from
this challenge by extracting a feature set consisting of TF-IDF vectors
based on word occurrences. Singular values decomposed from these
vectors indicate word-vector similarities between a {headline} and its
corresponding {body text}. We call this model XGB.
3.1.2 Recurrent Dual Encoder (RDE)
To compute the similarity between two text inputs, a recurrent dual encoder
composed of dual RNNs was used (Lowe et al. 2015). When an RNN
encodes a word sequence, each word is passed through a
word-embedding layer, which converts a word index to a 300-dimensional
vector. Using the final hidden state of each headline and body text RNN,
the probability of having an incongruent headline is calculated after the
encoding step. In the training objective, the incongruence score is as
follows:
p(label) = σ((h
H
th
)
T M h
B
tb + b),
L = − log𝜫N
n=1p(labeln
|h
H
n,th
, h
B
n,tb
) ….(3.1)
where h
H
th and h
B
tb are the last hidden state of each {headline} and {body
text} RNN with the dimensionality h ∈ R
d
. The M ∈ R
d*d and bias b are
learned model parameters. σ is the sigmoid function and N is the total
number of samples utilized in training.
3.1.3 Convolution Dual Encoder (CDE)
CDE is a CNN-based approach that employs a small set of words as
context. Local features are calculated with various-sized filters, and the
model determines the maximum value over time. CDE extracts one of the
most useful characteristics in this method to see if the article has
incongruent headlines. We apply Convolutional Dual Encoder to the
headline incongruence problem, following the CNN architecture for text
understanding (Kim 2014). We obtained a vector representation v = {vi |i =
1, · · ·, k} for each section of the article using max-over-time pooling after
computing convolution with k filters using the headline and body text word
sequences as input to the convolutional layer as follows:
vi = g(fi(W)) ….(3.2)
where g denotes the maximum-over-time pooling function, fi denotes the
CNN function with the i-th convolutional filter, and W ∈ R
t*d
is a word
sequence matrix. Using dual CNNs, we encode the {headline} and {body}
text as vector representations.
