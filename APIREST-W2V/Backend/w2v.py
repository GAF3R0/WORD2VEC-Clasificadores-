from gensim.models import Word2Vec
import json
import os
from gensim.utils import simple_preprocess 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
"""
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
"""

# Cargar dataset

corpus = [
    "El rapido zorro marron salta sobre el perro perezoso.",
    "El perro perezoso no se movio.",
    "El zorro marron es rapido y agil.",
    "El perro corre al zorro marron.",
    "El perro persigue al zorro marron.",
    "El perro atrapa al zorro marron.",
    "El perro atrapa al zorro marron de la cola.",
    "El perro atrapa al zorro marron de la cola y no lo suelta.",
    "El perro atrapa al zorro marron de la cola y no lo suelta porque lo quiere.",
    "El perro atrapa al zorro marron de la cola y no lo suelta porque lo quiere mucho."
]

# Leer archivo y separar en documentos
lemmatizer = WordNetLemmatizer()
stopwords_es = set(stopwords.words('spanish'))

# Tokenización y limpieza
def preprocess_text(text):
    tokens = simple_preprocess(text)
    lemmatized__tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_es]
    return lemmatized__tokens

tokenized_corpus = [preprocess_text(sentence) for sentence in corpus]
print(tokenized_corpus)

# Creación del modelo Word2Vec
model = Word2Vec(
    sentences = tokenized_corpus,
    vector_size = 100,
    window = 10,
    min_count = 1,
    workers = 1,
    sg = 1
)

# Pasar corpus a api
script_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(script_dir, 'APIREST-W2V', 'corpus.json')
with open(corpus_path, 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)
print(f"Corpus guardado exitosamente en: {corpus_path}")

# Guardar el modelo en la carpeta APIREST-W2V para que la API pueda cargarlo
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'APIREST-W2V', 'word2vec_model.bin')
model.save(model_path)
print(f"\nModelo guardado exitosamente en: {model_path}")


