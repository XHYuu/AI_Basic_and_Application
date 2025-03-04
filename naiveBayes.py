"""
Naive Bayes implementation.
code borrowed from: https://faculty.wcas.northwestern.edu/robvoigt/courses/2023_spring/ling334/assignments/a3.html

API inspired by SciKit-learn.
"""

import os, math
import re
import numpy as np
from collections import Counter


class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """

    def __init__(self, train_dir='data/haiti/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('data/english.stop')])
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        self.vocabulary = set()
        self.logprior = {}
        self.loglikelihood = {}  # keys should be tuples in the form (w, c)

    def train(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of SLP Chapter 4 (https://web.stanford.edu/~jurafsky/slp3/4.pdf).

        Note that self.train_data contains the paths to training data files.
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()

        The dataset's are pre-tokenized so you can get words with
        simply `words = doc.split()`

        Remember to account for whether the self.REMOVE_STOPWORDS flag is set or not;
        if it is True then the stopwords in self.stopwords should be removed whenever
        they appear.

        When converting from the pseudocode, consider how many loops over the data you
        will need to properly estimate the parameters of the model, and what intermediary
        variables you will need in this function to store the results of your computations.

        Follow the TrainNaiveBayes pseudocode to update the relevant class variables:
            - self.vocabulary, self.logprior, and self.loglikelihood.

        Note that the keys for self.loglikelihood should be tuples in the form of (w, c)
        where w is the string for the word and c is the string for the class.

        Parameters
        ----------
        None (reads training data from self.train_data)

        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # >>> YOUR ANSWER HERE
        p_class = []
        total_doc = 0
        counters = []
        total_word_in_class = []
        for c in self.classes:
            counter = Counter()
            total_word = 0
            c_docs = open(self.train_data[c]).readlines()
            p_class.append(len(c_docs))
            total_doc += len(c_docs)
            for doc in c_docs:
                words = re.split(r'[ \n.,!?":;()\[\]{}-]+', doc)
                if self.REMOVE_STOPWORDS:
                    words = [word.lower() for word in words
                             if word.isalnum() and word.lower() not in self.stopwords]
                else:
                    words = [word.lower() for word in words if word.isalnum()]
                for w in words:
                    if w == "":
                        continue
                    counter[w] += 1
                    total_word += 1
                    self.vocabulary.add(w)
            counters.append(counter)
            total_word_in_class.append(total_word)

        # Calculate class priors
        for i, c in enumerate(self.classes):
            self.logprior[c] = math.log(p_class[i] / total_doc)
        # Calculate likelihoods
        for i, counter in enumerate(counters):
            for word in self.vocabulary:
                self.loglikelihood[(word, self.classes[i])] = math.log(1. / len(self.vocabulary))
            for word, count in counter.items():
                self.loglikelihood[(word, self.classes[i])] = \
                    math.log((count + 1.) / (total_word_in_class[i] + 1. * len(self.vocabulary)))

        # >>> END YOUR ANSWER

    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier.

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4 (https://web.stanford.edu/~jurafsky/slp3/4.pdf).

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """
        # >>> YOUR ANSWER HERE
        words = re.split(r'[ \n.,!?":;()\[\]{}-]+', doc)
        if self.REMOVE_STOPWORDS:
            words = [word.lower() for word in words
                     if word.isalnum() and word.lower() not in self.stopwords]
        else:
            words = [word.lower() for word in words if word.isalnum()]
        log_p = self.logprior[c]
        for word in words:
            if word not in self.vocabulary or word == "" or (word, c) not in self.loglikelihood:
                continue
            log_p += self.loglikelihood[word, c]
        return log_p
        # >>> END YOUR ANSWER

    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.

        Consider using the `max` built-in function. There are a number of ways to do this:
           https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

        Parameters
        ----------
        doc : str
            A text representation of a document to score.

        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        # >>> YOUR ANSWER HERE
        max_score = -100000
        pred = ""
        for c in self.classes:
            score = self.score(doc, c)
            if max_score < score:
                pred = c
                max_score = score

        return pred
        # >>> END YOUR ANSWER

    def evaluate(self, test_dir='haiti/test', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Not the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to.

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """
        test_data = {c: os.path.join(test_dir, c) for c in self.classes}
        if not target in test_data:
            print('Error: target class does not exist in test data.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # >>> YOUR ANSWER HERE
        # Calculate outcomes
        for c in self.classes:
            if c == target:
                docs = open(test_data[c]).readlines()
                for doc in docs:
                    pred_c = self.predict(doc)
                    if pred_c == c:
                        outcomes['TP'] += 1
                    else:
                        outcomes['FN'] += 1
            else:
                docs = open(test_data[c]).readlines()
                for doc in docs:
                    pred_c = self.predict(doc)
                    if pred_c == c:
                        outcomes['TN'] += 1
                    else:
                        outcomes['FP'] += 1
        # calculate precision, recall, and F1 score
        precision = outcomes['TP'] / (outcomes['TP'] + outcomes['FP'])
        recall = outcomes['TP'] / (outcomes['TP'] + outcomes['FN'])
        f1_score = 2 * (precision * recall) / (recall + precision)
        # >>> END YOUR ANSWER
        return (precision, recall, f1_score)

    def print_top_features(self, k=10):
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp(self.loglikelihood[w, c] - min(
                    self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c))
                results[c][w] = ratio

        for c in self.classes:
            print(f'Top features for class <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key=lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')


target = 'relevant'

clf = NaiveBayesClassifier(train_dir='data/haiti/train')
clf.train()
print(f'Performance on class <{target.upper()}>, keeping stopwords')
precision, recall, f1_score = clf.evaluate(test_dir='data/haiti/dev', target=target)
print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

clf = NaiveBayesClassifier(train_dir='data/haiti/train', REMOVE_STOPWORDS=True)
clf.train()
print(f'Performance on class <{target.upper()}>, removing stopwords')
precision, recall, f1_score = clf.evaluate(test_dir='data/haiti/dev', target=target)
print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

clf.print_top_features()
