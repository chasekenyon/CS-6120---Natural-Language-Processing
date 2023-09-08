import nltk
from sklearn.model_selection import train_test_split

n = 2  # for bigrams, adjust as necessary for other n-gram models

f = open("WarrenBuffet.txt", "r")
text = f.read()
f.close()

sentences = nltk.sent_tokenize(text)

# Process each sentence
for i, sentence in enumerate(sentences):
    # Tokenize the sentence into words
    tokens = nltk.word_tokenize(sentence)

    # Pad sentence with <s> and </s> tokens
    tokens = ['<s>'] * (n-1) + tokens + ['</s>'] * (n-1)
    sentences[i] = ' '.join(tokens)

# Split the sentences into a training and testing set (90% training, 10% testing)
train_sentences, test_sentences = train_test_split(sentences, test_size=0.1)

# Write the training sentences to a file, one sentence per line
with open(f'train_data/berp-training_bi.txt', 'w') as f:
    for sentence in train_sentences:
        f.write(sentence + '\n')

# Write the testing sentences to a file, one sentence per line
with open(f'test_data/hw2-test_bi.txt', 'w') as f:
    for sentence in test_sentences:
        f.write(sentence + '\n')
