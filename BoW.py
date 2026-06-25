import nltk
import pandas as pd
import string
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

corpus = [
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "El perro perezoso no se movió.",
    "El zorro marrón es rápido y ágil.",
    "El perro corre al zorro marrón.",
    "El perro persigue al zorro marrón.",
    "El perro atrapa al zorro marrón.",
    "El perro atrapa al zorro marrón de la cola.",
    "El perro atrapa al zorro marrón de la cola y no lo suelta.",
    "El perro atrapa al zorro marrón de la cola y no lo suelta porque lo quiere.",
    "El perro atrapa al zorro marrón de la cola y no lo suelta porque lo quiere mucho."
]

# Preprocesamiento: tokenizar eliminar puntuacion y stopwords 
stopwords = set(stopwords.words('spanish'))
translator = str.maketrans('', '', string.punctuation)

def preprocess(text):
    text = text.lower()
    text = text.translate(translator)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

preprocessed_corpus = [preprocess(doc) for doc in corpus]

# Crear el diccionario de terminos
dictionary = corpora.Dictionary(preprocessed_corpus)

# Crear la bolsa de palabras (BoW) para cada documento
corpus_bow = [dictionary.doc2bow(doc) for doc in preprocessed_corpus] 

# Imprimir el BoW en un dataframe con pandas
df_bow = pd.DataFrame([
    {dictionary[id] : fred for id, fred in doc}
    for doc in corpus_bow
]).fillna(0)

print(df_bow)