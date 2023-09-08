import sys
from collections import Counter
import numpy as np
import math
import nltk


class NGramModel:
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    __slots__ = 'n_gram', 'is_laplace_smoothing', 'n_gram_counts', 'n_minus_1_gram_counts', 'threshold', 'vocab', 'probs'

    def __init__(self, n_gram=1, threshold=1, seed=1):
        """
        Args:
          n_gram (int): the context length
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
          threshold: words with frequency  below threshold will be converted to token
          n_gram_counts:
          n_minus_1_gram_counts:
          threshold:
          vocab:
          probs:
        """
        # Initializing different object attributes
        self.n_gram = n_gram
        self.is_laplace_smoothing = True
        self.vocab = []
        self.n_gram_counts = {}
        self.n_minus_1_gram_counts = {}
        self.threshold = threshold
        self.probs = {}
        np.random.seed(seed)

    def make_ngrams(self, tokens: list, n_gram) -> list:
        """Creates n-grams for the given token sequence.
        Args:
            tokens (list): a list of tokens as strings
            n_gram (int): the length of n-grams to create

        Returns:
            list: list of tuples of strings, each tuple being one of the individual n-grams
        """
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        n_grams = [" ".join(ngram) for ngram in ngrams]
        n_grams_token = []
        for word in n_grams:
            n_grams_token.append(tuple(word.split()))
        return n_grams_token

    def train(self, training_file_path):
        """Trains the language model on the given data. Input file that
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        N Gram Counts, Vocab, N Minus 1 Gram Counts
        """
        with open(training_file_path, 'r') as fh:
            content = fh.read().split()  # Read and split data to get list of words
        # Get the training data vocabulary
        self.vocab = list(set(content))

        # Get the count of each word
        # Replace the words with <UNK> if count is < threshold(=1) to reduce vocabulary
        counts = {}
        for word in content:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        for i in range(len(content)):
            if counts[content[i]] <= self.threshold:
                content[i] = 'UNK'

        # make use of make_n_grams function
        n_grams = self.make_ngrams(content, self.n_gram)
        n_minus_1_grams = self.make_ngrams(content, self.n_gram - 1)

        # populate dictionary of n_gram_counts
        for word in n_grams:
            if word in self.n_gram_counts.keys():
                self.n_gram_counts[word] += 1
            else:
                self.n_gram_counts[word] = 1

        # populate dictionary of n_minus_1_gram_counts (only for n_gram > 1)
        if self.n_gram > 1:
            for word in n_minus_1_grams:
                if word in self.n_minus_1_gram_counts.keys():
                    self.n_minus_1_gram_counts[word] += 1
                else:
                    self.n_minus_1_gram_counts[word] = 1
        print('n_gram_counts {}, n_minus_1_gram_counts {}, vocab {}'.format(len(self.n_gram_counts),
                                                                            len(self.n_minus_1_gram_counts),
                                                                            len(self.vocab)))

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        # Split the input sentence and replace out of vocabulary tokens with <UNK>
        # Calculate probability for each word and multiply(or take log and sum) them to get the sentence probability
        content = sentence.split()

        # Get the index of actual first token other than <s>
        index = 0
        if self.n_gram != 1:
            for i in range(len(content)):
                if content[i] != '<s>':
                    index = i
                    break
        # replace tokens unseen in training data with 'UNK' tag
        for i in range(len(content)):
            if content[i] not in self.vocab:
                content[i] = 'UNK'

        logProbability = 0
        y = len(self.vocab)
        for i in range(index, len(content)):
            num = tuple(content[i - self.n_gram + 1:i + 1])
            den = tuple(content[i - self.n_gram + 1:i])
            x = 0
            y = 0
            if self.is_laplace_smoothing:
                x = 1
                y = len(self.vocab)
            if num in self.n_gram_counts.keys():
                x += self.n_gram_counts[num]
            if self.n_gram != 1:
                if den in self.n_minus_1_gram_counts.keys():
                    y += self.n_minus_1_gram_counts[den]
            else:
                y += len(self.vocab)
            logProbability += math.log(x) - math.log(y)

        return math.exp(logProbability)

    def calculateProbs(self):
        counts = {}
        if self.n_gram > 1:
            for given_word in self.n_minus_1_gram_counts:
                counts[given_word] = {}
                for next_word in self.vocab:
                    given_word_list = []
                    given_word_list = list(given_word)
                    given_word_list.append(next_word)
                    next_word = tuple(given_word_list)
                    if next_word in self.n_gram_counts:
                        counts[given_word][next_word] = self.n_gram_counts[next_word]

            for given_word in self.n_minus_1_gram_counts:
                self.probs[given_word] = {}
                for next_word in self.vocab:
                    given_word_list = []
                    given_word_list = list(given_word)
                    given_word_list.append(next_word)
                    next_word = tuple(given_word_list)
                    if next_word in self.n_gram_counts:
                        self.probs[given_word][next_word] = 0

            for key, value in counts.items():
                denominator = 0
                for key2, value2 in counts[key].items():
                    denominator += value2
                for key2, value2 in counts[key].items():
                    self.probs[key][key2] = value2 / denominator

        else:
            for given_word in self.n_gram_counts:
                self.probs[given_word] = self.n_gram_counts[given_word] / len(self.vocab)
        print("Probability calulated")

    def generate_sentence(self, n_gram=1):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        # Start with <s> and randomly generate words until we encounter sentence end
        # Append sentence begin markers for n>2
        # Keep track of previous word for stop condition

        sentence = "<s>"
        prev_word = "<s>"
        if self.n_gram > 2:
            for i in range(0, self.n_gram - 2):
                sentence = sentence + " " + "<s>"

        if self.n_gram > 1:
            while prev_word != "</s>":
                # Construct the (n-1) gram so far
                # Get the counts of all available choices based on n-1 gram
                # Convert the counts into probability for random.choice() function
                # If <s> is generated, ignore and generate another word
                split_sent = tuple(sentence.split())
                sent_len = len(split_sent)
                variable = self.probs[split_sent[sent_len - n_gram + 1:sent_len]]
                val1 = []
                for val in list(variable.keys()):
                    val1.append(val[len(val) - 1])

                random_word = np.random.choice(val1, p=list(variable.values()))
                if random_word != "<s>":
                    sentence = sentence + " " + random_word
                    prev_word = random_word
            if n_gram > 2:
                for i in range(0, n_gram - 2):
                    sentence = sentence + " </s>"
            print(sentence)

        else:
            # In case of unigram model, n-1 gram is just the previous word and possible choice is whole vocabulary
            while prev_word != "</s>":
                # Convert the counts into probability for random.choice() function
                # If <s> is generated, ignore and generate another word
                # Append sentence end markers for n>2
                val1 = []
                for val in list(self.probs.keys()):
                    val1.append(val[0])
                val2 = []
                for val in list(self.probs.values()):
                    val2.append(val)
                # print(probs.values())
                prob_dist = list(self.probs.values())
                np.random.seed(1)
                random_number = np.random.choice(val1, p=prob_dist)
                if random_number != "<s>":
                    sentence = sentence + random_number + " "
                    prev_word = random_number
            print(sentence)
        return sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        # Generate sentences one by one and store
        sentences = []
        for i in range(1, n):
            sentences.append(self.generate_sentence(n_gram=2))
        return sentences
    
    def calculate_bigram_perplexity(self, test_file_path):
        """
        Calculate the perplexity of the bigram model on a test set.

        Args:
            test_file_path (str): Path to the test set file

        Returns:
            float: The calculated perplexity
        """
        # read the test data
        with open(test_file_path, 'r') as fh:
            content = fh.read().split()

        # get the bigrams in the test data
        test_data_bigrams = list(nltk.bigrams(content))

        # calculate the perplexity
        N = len(test_data_bigrams)  # total number of bigrams
        log_perplexity = 0.0

        for bigram in test_data_bigrams:
            n_minus_1_gram = bigram[0]  # the (n-1)-gram is just the first word of the bigram
            # get the probability of the bigram
            if n_minus_1_gram in self.n_minus_1_gram_counts:
                count_bigram = self.n_gram_counts.get(bigram, 0)
                if self.is_laplace_smoothing:
                    prob_bigram = (count_bigram + 1) / (self.n_minus_1_gram_counts[n_minus_1_gram] + len(self.vocab))
                else:
                    prob_bigram = count_bigram / self.n_minus_1_gram_counts[n_minus_1_gram]
            else:
                # if the (n-1)-gram is not in the training data, assign it a small probability
                if self.is_laplace_smoothing:
                    prob_bigram = 1 / len(self.vocab)
                else:
                    prob_bigram = 0.0
            # update the perplexity (add the log probability)
            if prob_bigram > 0:
                log_perplexity -= math.log(prob_bigram)

        # take the Nth root of the perplexity
        perplexity = math.exp(log_perplexity / float(N))

        return perplexity





# if __name__ == '__main__':
#     file = ''
#     data_files = {
#         1: ('train_data/berp-training_uni.txt', 'test_data/hw2-test_uni.txt'),
#         2: ('train_data/berp-training_bi.txt', 'test_data/hw2-test_bi.txt'),
#         3: ('train_data/berp-training-tri.txt', 'test_data/hw2-test_tri.txt'),
#         4: ('train_data/berp-training-four.txt', 'test_data/hw2-test_four.txt'),
#         5: ('train_data/berp-training-five.txt', 'test_data/hw2-test_five.txt'),
#         6: ('train_data/berp-training_six.txt', 'test_data/hw2-test_six.txt'),
#         7: ('train_data/berp-training_seven.txt' 'test_data/hw2-test_seven.txt')
#     }
    
if __name__ == '__main__':
    file = ''
    data_files = {
        2: ('train_data/berp-training_bi.txt', 'test_data/hw2-test_bi.txt')
    }    
    
    ngm = NGramModel()

    n_g = 2

    ngm = NGramModel(n_gram=n_g)
    ngm.train(training_file_path=data_files[n_g][0])

    # load and read the test file content.
    with open(data_files[n_g][1], 'r') as fh:
        test_content = fh.read().split("\n")

    print(data_files[n_g][1], ' loaded... {} sentences.'.format(len(test_content)))
    # Take first 10 sentences
    ten_sentences_1 = test_content[:10]
    # calculate score of each sentence
    probabilities = []
    for test_sentence in ten_sentences_1:
        prob = ngm.score(test_sentence)
        probabilities.append(prob)
        print(f"Score for '{test_sentence}' is {prob}")
    probabilities = np.array(probabilities)
    mean = np.mean(probabilities)
    std_dev = np.std(probabilities)
    print(f"Mean: {float(mean)}")
    print(f"Std_dev: {float(std_dev)}")

    #
    ngm.calculateProbs()

    sentences = ngm.generate(10)
    # print("Sentences:")
    # for sentence in sentences:
    #     print(sentence)
  
    perplexity = ngm.calculate_bigram_perplexity(test_file_path=data_files[2][1])
    print('Perplexity:', perplexity)    