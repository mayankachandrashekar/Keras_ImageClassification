import re

import requests


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  digi=re.compile(r'\d')
  cleantext1=re.sub(digi,"",cleantext)
  return cleantext1

rootFolder = "C:\\Users\\Vivaswan Chandru\\Box Sync\\Mayanka-Research\\2018-CrisisCNN-WordEmbedding\\Code\\Keras_ImageClassification\\"
f1 = open(rootFolder + "data\\wikipediaLinks.txt", "r")
f3 = open(rootFolder + "data\\wikipediaData.txt", "w", encoding="utf-8")
if f1.mode == 'r':
    contents = f1.readlines()
    for line in contents:
        annotation=line.replace("\n","").split("/")
        f2 = open(rootFolder + "data\\WikipediaData\\"+annotation[len(annotation)-1]+".txt", "w",encoding="utf-8")
        url2 = "https://en.wikipedia.org/w/api.php?action=parse&page="+annotation[len(annotation)-1]+"&contentmodel=wikitext&format=json"
        resp2 = requests.get(url2)
        print(url2)
        respjson=resp2.json()
        respjson1=respjson['parse']['text']
        print("WIKIPEDIA TEXT \n ____________")
        #print(respjson1['*'])
        cleantext=cleanhtml(respjson1['*']).replace("&#","")
        cleantextList=cleantext.split("\n")

        cleantext1=list(filter(lambda x: len(x)> 3, cleantextList))
        cleantext2 = " ".join(cleantext1)#.replace("\\u", "")
        #print(cleantext1)
        #print(" ".join(cleantext1))
        f2.write(cleantext2)
        f3.write(cleantext2)
        f2.close()
