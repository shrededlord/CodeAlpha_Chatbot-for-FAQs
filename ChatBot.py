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


lemmer = nltk.stem.WordNetLemmatizer()
#wordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens (tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



GREETING_INPUTS=("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES=["hi there", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting (sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
