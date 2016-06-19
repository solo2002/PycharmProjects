from nltk.probability import FreqDist
import nltk

#from nltk.book import *
all_str = 'This is str 1  this is str 2  and here is 3'
all_tokens = nltk.word_tokenize(all_str.lower()) #list

text = nltk.Text(all_tokens)

fdist = FreqDist(text)
#fdist.plot()
word_counter = {}
for word in all_tokens:
    if word in word_counter:
        word_counter[word] += 1
    else:
        word_counter[word] = 1
popular_words = sorted(word_counter, key = word_counter.get, reverse=True)
top_word_set = popular_words[:4]
print 'top_word_set', top_word_set

new_str = 'this is new str and test it this this this'
new_tokens = new_str.split()#nltk.word_tokenize(new_str)


word_dict = dict.fromkeys(top_word_set, 0)
print word_dict
for w in new_tokens:
    if w in top_word_set:
        word_dict[w] += 1

print word_dict

#print fdist['this']


'''1. create a Repo at github
2. at local directory:
git init
git add .
git commit -m "Initialize Project"
git remote add origin "... "//copy from github repo
git push origin master //enter username and password

//Add a Readme for this repo and "commit new file"
...
git remote -v//show two link
git pull //the link showing above
git push
...
//Add a tag
git tag -a v1.0 -m "Version 1.0"
git push origin v1.0 // could find at "release" panel in the repo at github

...
//clone repo from other's repo
1. fork the repo
2. mkdir newDirectoryName
3. cd newDirectoryName
4. git clone URL
from nltk.corpus import movie_reviews
import nltk

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
          for fileid in movie_reviews.fileids(category)]
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
print type(documents)
print documents'''