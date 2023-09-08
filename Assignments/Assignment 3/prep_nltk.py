# Importing libraries
import nltk
from nltk import pos_tag
from sklearn.model_selection import train_test_split
import pprint, time

# download the treebank corpus from nltk
# nltk.download('treebank')
# nltk.download('brown')
# download the universal tagset from nltk
# nltk.download('universal_tagset')

# reading the Treebank tagged sentences
# https://www.nltk.org/_modules/nltk/tag/mapping.html

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
for i in range(len(nltk_data)):
    nltk_data[i].insert(0, ('<@>', 'SoS'))
    nltk_data[i].insert(len(nltk_data[i]), ('</@>', 'EoS'))
    print()

train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 0)
pos_tagger = pos_tag(sentences=train_set, load=None)
start = time.time()

tagged_seq = pos_tagger.test(sentence='<@> they can fish well </@>')
print(tagged_seq)
end = time.time()
difference = end - start

print("Time taken in seconds: ", difference)
