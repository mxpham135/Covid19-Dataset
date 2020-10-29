!pip install spacy
!pip install newsapi-python

!python -m spacy download en_core_web_lg

#import libraries
import spacy
import en_core_web_lg
from newsapi import NewsApiClient
import pickle
import pandas as pd
from collections import Counter 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key='7c9b7e90bdad4ca89565175bb4240918')

def getArticles(pagina):
    temp = newsapi.get_everything(q='coronavirus', language='en', 
                                  from_param='2020-10-01', to='2020-10-29', 
                                  sort_by='relevancy', page=pagina)
    return temp
articles = list(map(getArticles, range(1,6)))

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = '/content/articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

dados = []
titles = []
dates = []
descriptions = []
counter = 0
for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        titles.append(title)
        date = x['publishedAt']
        dates.append(date)
        description = x['description']
        descriptions.append(description)
        content = x['content']
        dados.append({'title':titles[counter], 'date':dates[counter], 'desc':descriptions[counter], 'content':content})
        counter += 1
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

def get_keywords_eng(content):
    result = []
    punctuation = string.punctuation
    pos_tag = ['NOUN','VERB','PROPN']
    for token in nlp_eng(content):
        if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
          continue
        if (token.pos_ in pos_tag):
          result.append(token.text)
    return result

results = []
for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

from google.colab import files
df.to_csv('covid19-dataset.csv') 
files.download('covid19-dataset.csv')
