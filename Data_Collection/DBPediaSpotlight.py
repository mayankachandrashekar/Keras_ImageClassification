import requests
import xml.etree.ElementTree as ET

rootFolder = "C:\\Users\\Vivaswan Chandru\\Box Sync\\Mayanka-Research\\2018-CrisisCNN-WordEmbedding\\Code\\Keras_ImageClassification\\"
f = open(rootFolder + "data\\IAPR_text", "r")
f1 = open(rootFolder + "data\\wikipediaLinks.txt", "w+")
if f.mode == 'r':
    contents = " ".join(f.readlines())
    contents = contents.replace("\n", " ")
    contents1 = list(set(contents.split(" ")))
    contents1.remove("")
    for line in contents1:
        line = line.replace("-", " ")
        # text = "Narendra Modi is the Prime Minister of India"
        url = "http://model.dbpedia-spotlight.org/en/annotate?text=" + line
        resp = requests.get(url)
        if resp.status_code != 200:
            # This means something went wrong.
            print("Error")
        print(resp.text)

        root = ET.fromstring(resp.text)
        print(root.tag)
        print(root.attrib)

        for child in root.iter('Resource'):
            annotationURI = child.get('URI')
            print(annotationURI)
            annotation = annotationURI.split('/')
            url2 = "https://en.wikipedia.org/wiki/" + annotation[len(annotation) - 1]
            resp2 = requests.get(url2)
            print(url2)
            print(resp2)
            if resp2.status_code == 200:
                f1.write(url2 + "\n")
