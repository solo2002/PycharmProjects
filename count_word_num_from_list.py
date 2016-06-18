from nltk.probability import FreqDist
import nltk
#from nltk.book import *
all_str = 'This is str 1  this is str 2  and here is 3'
all_tokens = nltk.word_tokenize(all_str.lower()) #list

text = nltk.Text(all_tokens)

fdist = FreqDist(text)

word_counter = {}
for word in all_tokens:
    if word in word_counter:
        word_counter[word] += 1
    else:
        word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse=True)
top_word_set = popular_words[:3]
print popular_words

new_str = 'this is new str and test it'
new_tokens = new_str.split()#nltk.word_tokenize(new_str)


word_dict = {}
for w in new_tokens:
    if w in top_word_set:
        if w in word_dict.keys():
            word_dict[w] = word_dict[w] + 1
        else:
            word_dict[w] = 1
print word_dict

#print fdist['this']



'''from nltk.corpus import movie_reviews
import nltk

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
          for fileid in movie_reviews.fileids(category)]
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
print type(documents)
print documents'''