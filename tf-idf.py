from gensim import corpora, models
import pandas as pd 

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

tokenized_corpus = [doc.lower().split() for doc in corpus]

# Crear el diccionario y el corpus de BoW
dictionary = corpora.Dictionary(tokenized_corpus)
bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

# Modelo TF - IDF
tfidf = models.TfidfModel(bow_corpus)

# Convertir el corpus a TF - IDF
tfidf_corpus = tfidf[bow_corpus]

# Imprimir el TF - IDF en un dataframe con pandas
df_tfidf = pd.DataFrame([
    {dictionary[id] : float(value) for id, value in doc}
    for doc in tfidf_corpus
]).fillna(0)

print(df_tfidf)