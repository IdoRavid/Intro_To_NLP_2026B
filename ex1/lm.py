from builtins import sum, zip, range, max, print, len

import math
import os
import pickle
from collections import Counter
import spacy
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")
CORPUS_CACHE = "corpus.pkl"


def get_corpus(dataset):
    if os.path.exists(CORPUS_CACHE):
        print("Loading corpus from cache...")
        with open(CORPUS_CACHE, "rb") as f:
            return pickle.load(f)
    texts = [row['text'] for row in dataset if row['text'].strip()]
    corpus = [
        [token.lemma_.lower() for token in doc if token.is_alpha]
        for doc in nlp.pipe(texts, batch_size=256)
    ]
    with open(CORPUS_CACHE, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


class LanguageModel:
    START = "<START>"

    def __init__(self, dataset):
        print("Building corpus...")
        self.corpus = get_corpus(dataset)
        print(f"Corpus built: {len(self.corpus)} documents")
        print("Counting n-grams...")
        self.unigram_counts = self._count_ngrams(1)
        self.bigram_counts = self._count_ngrams(2)
        self.total_tokens = sum(self.unigram_counts.values())
        self.start_total = sum(v for (k1, k2), v in self.bigram_counts.items() if k1 == self.START)
        print(f"Done. {len(self.unigram_counts)} unigrams, {len(self.bigram_counts)} bigrams")

    def _count_ngrams(self, n):
        counts = Counter()
        for doc in self.corpus:
            tokens = ([self.START] * (n - 1) + doc) if n > 1 else doc
            counts.update(zip(*[tokens[i:] for i in range(n)]))
        return counts

    def _predict_unigram(self, w1):
        return math.log(self.unigram_counts[(w1,)] / self.total_tokens)

    def _predict_bigram(self, w1, w2):
        count = self.bigram_counts.get((w1, w2), 0)
        if count == 0:
            return float('-inf')
        if w1 == self.START:
            denom = self.start_total
        else:
            denom = self.unigram_counts.get((w1,), 0)
        if denom == 0:
            return float('-inf')
        return math.log(count / denom)

    def _predict_interpolation(self, w1, w2, lambda_bigram):
        p_bigram = math.exp(self._predict_bigram(w1, w2))
        p_unigram = math.exp(self._predict_unigram(w2))
        return math.log(lambda_bigram * p_bigram + (1 - lambda_bigram) * p_unigram)

    def predict_next_bigram(self, w1):
        vocab = [w for (w,) in self.unigram_counts]
        return max(vocab, key=lambda w2: self._predict_bigram(w1, w2))

    def _tokenize(self, sentence):
        doc = nlp(sentence)
        return [token.lemma_.lower() for token in doc if token.is_alpha]

    def calculate_sentence_prob_bigram(self, sentence):
        words = [self.START] + self._tokenize(sentence)
        prob = 0.0
        for i in range(1, len(words)):
            prob += self._predict_bigram(words[i - 1], words[i])
        return prob

    def calculate_sentence_prob_inter(self, sentence, bigram_lambda):
        words = [self.START] + self._tokenize(sentence)
        prob = 0.0
        for i in range(1, len(words)):
            prob += self._predict_interpolation(words[i - 1], words[i], bigram_lambda)
        return prob

    def calculate_perplexity_bigram(self, sentences):
        total_log_prob = sum(self.calculate_sentence_prob_bigram(s) for s in sentences)
        N = sum(len(self._tokenize(s)) for s in sentences)
        return math.exp(-total_log_prob / N)

    def calculate_perplexity_inter(self, sentences, bigram_lambda):
        total_log_prob = sum(self.calculate_sentence_prob_inter(s, bigram_lambda) for s in sentences)
        N = sum(len(self._tokenize(s)) for s in sentences)
        return math.exp(-total_log_prob / N)


if __name__ == "__main__":
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    # constructor also `trains` the model
    lm = LanguageModel(dataset)
    # q2: "I have a house in ... ":
    print("I have a house in ... ")
    print(lm.predict_next_bigram("in"))
    # q3A
    sentences= ["Brad Pitt was born in Oklahoma", "The actor was born in USA "]
    for sentence in sentences:
        print("probability of (Bigram Model): " + sentence)
        print(lm.calculate_sentence_prob_bigram(sentence))
    # q3B
    print("Perplexity of  the test set(Bigram Model): ")
    print(lm.calculate_perplexity_bigram(sentences))
    # q4
    for sentence in sentences:
        print("probability of (Interpolation Model): " + sentence)
        print(lm.calculate_sentence_prob_inter(sentence, 2/3))
    print("Perplexity of  the test set(Bigram Model): ")
    print(lm.calculate_perplexity_inter(sentences, 2/3))
