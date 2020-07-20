import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt





consumer_key = 'xxxxxx'
consumer_secret = 'xxxxx'
access_token = 'xxxx'
access_token_secret = 'xxxxx'

# Doğrulama nesnesi oluşturma
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    
#access token ve access token secret atanması
authenticate.set_access_token(access_token, access_token_secret) 
    
# Kimlik doğrulama bilgilerini iletirken API nesnesi oluşturma
api = tweepy.API(authenticate, wait_on_rate_limit = True)        
        
# screen name ;aranan kısının Twitter user name i belirtir.
posts = api.user_timeline(screen_name="Fenerbahce", count = 2000, lang ="tr", tweet_mode="extended")
print("Yakında atılan 5 tweet:\n")
i=1
for tweet in posts[:5]:
    print(str(i) +') '+ tweet.full_text + '\n')
    i= i+1


#Tweet adlı bir sütuna sahip bir veri DataFrame oluşturma
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
# ilk 5 tweetin görüntülenmesi
df.head()



# Öznellik için oluşturulan fonksiyon
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Polarity için oluşturulan fonksiyon
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


# Öznellik ve polarite için 2 adet sutun oluşturma
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

# Öznellik ve polaritenin gösterilmesi
df


# Sık sık kullanılan kelimeleri bir görsele dokme kısmı
allWords = ' '.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)


plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Hesaplama fonksiyonunun; negative (-1), neutral (0) ve positive (+1) analiz şeklinde oluşturulması
def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)
# bu analizin gösterimi.
df



# Pozitif tweet gosterimi.
print('Pozitif tweet gosterimi:\n')
j=1
sortedDF = df.sort_values(by=['Polarity']) #tweet sıralama
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analysis'][i] == 'Positive'):
    print(str(j) + ') '+ sortedDF['Tweets'][i])
    print()
    j= j+1
    


# Negatif tweet gosterimi 
print('Negatif tweet gosterimi :\n')
j=1
sortedDF = df.sort_values(by=['Polarity'],ascending=False) #tweet sıralama
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analysis'][i] == 'Negative'):
    print(str(j) + ') '+sortedDF['Tweets'][i])
    print()
    j=j+1
    
    




# Plot işlemi
plt.figure(figsize=(8,6)) 
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue') 

plt.title('Sentiment Analysis') 
plt.xlabel('Polarity') 
plt.ylabel('Subjectivity') 
plt.show()



# Pozitif tweetin oraanı
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']
ptweets

round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)




# Negatif tweetin oraanı
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']
ntweets

round( (ntweets.shape[0] / df.shape[0]) * 100, 1)


df['Analysis'].value_counts()



# Görselleştirme ve sayısal ifadelerin son hali
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()