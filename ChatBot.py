import nltk
import io #deal with opening/reading a file
import random
import string #process python strings
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open(r'C:\Users\keeee\Desktop\Codealpha\datafile.txt', 'r', errors='ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
sent_tokens = nltk.sent_tokenize(raw)# converts to List of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words


