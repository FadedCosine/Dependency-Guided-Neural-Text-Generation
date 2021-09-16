import os
from multiprocessing import Pool
import pdb
import numpy as np
import nltk
import math
from nltk.translate.bleu_score import SmoothingFunction, ngrams, sentence_bleu, brevity_penalty
from collections import Counter
import collections
from fractions import Fraction
try: 
    from multiprocessing import cpu_count
except: 
    from os import cpu_count

class Metrics(object):
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_score(self):
        pass


class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, num_real_sentences=500, num_fake_sentences=10000):
        super(Bleu, self).__init__()
        self.name = 'Bleu'
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = num_real_sentences
        self.reference = None
        self.is_first = True
        self.num_sentences = num_fake_sentences


    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    # fetch REAL DATA
    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        raise Exception('make sure you call BLEU paralell')
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        maxx = self.num_sentences
        with open(self.test_data) as test_data:
            for i, hypothesis in enumerate(test_data):
                #print('i : {}'.format(i))
                hypothesis = nltk.word_tokenize(hypothesis)
                result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis, weight)))
                if i > maxx : break
        score = 0.0
        cnt = 0
        for it, i in enumerate(result):
            #print('i : {}'.format(it))
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3, model_path='', num_sentences=500):
        super(SelfBleu, self).__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = num_sentences
        self.reference = None
        self.is_first = True


    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            #genious:
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    sentence_ngrams = list(ngrams(sentence, n))
    distinct_ngrams = set(sentence_ngrams)
    return len(distinct_ngrams) / len(list(sentence_ngrams))



def distinct_upto(sentences, n):
    sentences = [i for i in sentences if len(i) > 5]
    res = []
    for i in range(1,n+1):
        res.append(distinct_n_corpus_level(sentences, i))
    return res

def distinct_document(sentences, n):
    document = [item for sentence in sentences for item in sentence]
    res = []
    for i in range(1,n+1):
        res.append(distinct_n_sentence_level(document, i))
    return res


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

def distinct_n_corpus_level_single_sent(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return [distinct_n_sentence_level(sentence, n) for sentence in sentences]


def compute_probs(cnter,token_lists):
    tot = 0
    probs = []
    for i in cnter:
        tot+= cnter[i]
    for i in token_lists:
        if i in cnter:
            probs.append(cnter[i] / tot)
        else:
            probs.append(1e-10)
    return np.array(probs)

def count(x, n_gram):
    cnter = collections.Counter()
    for line in x:
        ngram_res = []
        temp = [-1] * (n_gram - 1) + line + [-1] * (n_gram - 1)
        for i in range(len(temp) + n_gram - 1):
            ngram_res.append(str(temp[i:i + n_gram]))
        cnter.update(ngram_res)
    return cnter

def kld(references, hypotheses, n_gram):

    r_cnter = count(references,n_gram)
    h_cnter = count(hypotheses,n_gram)

    s = set(r_cnter.keys())
    s.update(h_cnter.keys())
    s = list(s)
    r_probs = compute_probs(r_cnter, s)
    h_probs = compute_probs(h_cnter, s)
    kld = np.sum(r_probs * np.log(r_probs/h_probs))
    return kld

def ms_jaccard(ref,hyp,n_gram):
    res = []
    for i in range(1,1+n_gram):
        rc = count(ref,i)
        hc = count(hyp,i)
        n_gram_set = set(rc.keys())
        n_gram_set.update(hc.keys())
        rprob= compute_probs(rc,n_gram_set)
        hprob= compute_probs(hc,n_gram_set)
        numerator = np.sum(np.minimum(rprob,hprob))
        denominator = np.sum(np.maximum(rprob,hprob))
        res.append(numerator / denominator)
    score = []
    for i in range(1,1+n_gram):
        score.append(geo_mean(res[:i]))
    return score
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1,n_gram+1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res

def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)
    
    return score / len(reference)

def calc_bleu_ngram_singe_sent(reference, hypothesis, n_gram):
    score = []
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score.append(sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1))
    
    return score


def ref_cnts2(references,n):
    ref_mcnts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for i in reference_counts:
            if i not in ref_mcnts: ref_mcnts[i] = [reference_counts[i],0]
            elif ref_mcnts[i][-1] < reference_counts[i]:
                if ref_mcnts[i][0] < reference_counts[i]:
                    ref_mcnts[i] = [reference_counts[i],ref_mcnts[i][0]]
                else:
                    ref_mcnts[i][-1] = reference_counts[i]
    return ref_mcnts


def modified_precision(ref_mcnts, hypothesis,n, isself=False):
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    hyp_mcnts = {}
    for ngram in counts:
        if ngram in ref_mcnts: hyp_mcnts[ngram] = ref_mcnts[ngram]
        else : hyp_mcnts[ngram] = 0
    if isself:
        clipped_counts = {
            ngram: min(count, ref_mcnts[ngram][1]) if count == ref_mcnts[ngram][0] else min(count, ref_mcnts[ngram][0])
            for ngram, count in counts.items()
        }
    else:
        clipped_counts = {
            ngram: min(count, ref_mcnts.get(ngram,0)) for ngram, count in counts.items()
        }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)

def closest_ref_length(ref_lens, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
    )
    return closest_ref_len

def selfbleu(x, n):
    x_mcnts = {i: ref_cnts2(x, i) for i in range(1, n + 1)}
    x_lens = [len(i) for i in x]
    bleu_scores = {i:[] for i in range(1,n+1)}
    for idx, hyp in enumerate(x):
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
        for i in range(1, n + 1):
            p_i = modified_precision(x_mcnts[i], hyp, i,True)
            p_numerators[i] = p_i.numerator
            p_denominators[i] = p_i.denominator
        hyp_lengths = len(hyp)
        ref_lengths = closest_ref_length(iter(x_lens[:idx] + x_lens[idx+1:]), hyp_lengths)
        bp = brevity_penalty(ref_lengths, hyp_lengths)
        for i in range(1,n+1):
            if p_numerators[i] == 0: p_numerators[i] = 1e-100
            s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
            s = bp * math.exp(math.fsum(s))
            bleu_scores[i].append(s)
    return [np.mean(bleu_scores[i]) for i in range(1,n+1)]

def selfbleu_singe_sent(x, n):
    x_mcnts = {i: ref_cnts2(x, i) for i in range(1, n + 1)}
    x_lens = [len(i) for i in x]
    bleu_scores = {i:[] for i in range(1,n+1)}
    for idx, hyp in enumerate(x):
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
        for i in range(1, n + 1):
            p_i = modified_precision(x_mcnts[i], hyp, i,True)
            p_numerators[i] = p_i.numerator
            p_denominators[i] = p_i.denominator
        hyp_lengths = len(hyp)
        ref_lengths = closest_ref_length(iter(x_lens[:idx] + x_lens[idx+1:]), hyp_lengths)
        bp = brevity_penalty(ref_lengths, hyp_lengths)
        for i in range(1,n+1):
            if p_numerators[i] == 0: p_numerators[i] = 1e-100
            s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
            s = bp * math.exp(math.fsum(s))
            bleu_scores[i].append(s)
    return bleu_scores

def repetition(hyps):
    """
    计算生成文本，在结尾陷入循环的概率；
    并没有计算中间陷入循环的情况
    """
    max_n = 100
    n_repeated_examples = 0
    for obj in hyps:
        gen = obj
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n * n_repeat:n * (n_repeat + 1)]) == n and \
                    rev_gen[n * n_repeat:n * (n_repeat + 1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n + 1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            n_repeated_examples += 1
    return n_repeated_examples / len(hyps)
