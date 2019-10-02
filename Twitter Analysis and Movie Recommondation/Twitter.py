import tweepy 
from tweepy import OAuthHandler
# auth hadler is to authentication and authorization to twitter
from tweepy import Stream
# stream will fetch live data and it will take a object to write data
from tweepy import StreamListener
# it is stream listener class which will get data from twitter
# it will call on_data method whenever it gets tweets
# it will call on_error method whenever error comes
consumer_key = "SrLW2zfM0OyLGvYNNcFrYzxDr"
consumer_secret = "XvtEXoytn3yt8JBFOjw6SME7CuzK7OrvMpDPi1b5bstGa6SP0u"

access_token  = "1148898544343908353-CYnIwZoqadfRyUB1JsaJpegJ0HHatj"
access_secret = "LI8KseED7Je5TqbdrXgMpW6dX2F2p0Klk6KcUZ91ZuUYy"
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

class Fetch_Data(StreamListener):
    def on_data(self,tweet):
        #print(tweet)
        fp = open("Prog_l.json","a")
        fp.write(tweet)
        fp.close()
    def on_error(self,error):
        print(error)
      
def Start_Fetching():
    fetch = Fetch_Data()
    stream = Stream(auth,fetch)
    # Stream will authentication and fetch data and feed into fetch.on_data
    # or fetch.on_error methods accordingly
    stream.filter(track=["python","java","c","c++","javascript","ruby"])
    # list of topicks to fetch relvent tweets about those topics

fp = open("Prog_lang.json")
lang = []
for line in fp : 
    try : 
        obj = json.loads(line)
        lang.append(obj)
    except : 
        continue
# Start_Fetching()  # start fetching data 



fp = open("Prog_lang.json")
lang = []
for line in fp : 
    try : 
        obj = json.loads(line)
        lang.append(obj)
    except : 
        continue

l = []
for tweet in lang : 
    try :
        l.append(tweet['lang'])
    except : 
        continue 
s  = pd.Series(l)
s.value_counts()[:10].plot(kind='bar')


# konsi country se kiya huaa h post
c = []
for tweet in lang : 
    try :
        c.append(tweet['place']['country'])
    except : 
        continue 
s  = pd.Series(c)
s.value_counts()[:10].plot(kind='bar')


# print all tweets which are in English language
for tweet in lang :
    try : 
        if tweet['lang'] == 'en' :
            print(tweet['text'],sep='\n\n')
    except : 
        continue

# hash_tag 
h = []
for tweet in lang : 
    try :
        if tweet['lang'] == 'en' :
            h.extend([ tag for tag in tweet['text'].split() if "#" in tag ])
    except : 
        continue 
s  = pd.Series(h)
s.value_counts()[:20].plot(kind='bar')

