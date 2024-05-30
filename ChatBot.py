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
        
def response (user_response):
    chatbot_response=""
    sent_tokens.append(user_response)
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat= vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        chatbot_response = chatbot_response+"I am sorry! I don't understand you."
        return chatbot_response
    else:
        chatbot_response = chatbot_response+sent_tokens[idx]
        return chatbot_response

if __name__=="__main__":
    flag=True
    print("Chatbot: My name is Chatbot. I will answer your queries about Chatbots. If you want to exit, type Bye!")
    while(flag==True):
        user_response=input("Input: ")
        user_response=user_response.lower()
        if (user_response!='bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                flag=False
                print("Chatbot: You are welcome..")
            else:
                if (greeting(user_response)!=None):
                    print("Chatbot: "+greeting(user_response))
                else:
                    print("Chatbot:",end="")
                    print (response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag=False
            print("Chatbot: Bye! take care..")