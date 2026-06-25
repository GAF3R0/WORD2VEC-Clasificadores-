import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)
stemmer = SnowballStemmer('spanish')
stopwords_es = set(stopwords.words('spanish'))

Corpus = [
    "Me encanta este producto, es excelente",
    "Muy mala calidad, no lo recomiendo",
    "El servicio fue rápido y amable",
    "Horrible experiencia, nunca volveré",
    "Es un artículo fantástico, muy útil",
    
    "No me gustó, llegó roto",
    "La atención al cliente fue perfecta",
    "Pésimo, no funciona como esperaba",
    "La mejor pelicula que he visto",
    "El video esta aburrido",
    "La pelicula fue mala y aburrida",
    "No me gustó la película",
    "La película estuvo increíble",
    "El video fue espectacular",
    "Me gustó mucho la película"
]
etiquetas = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]

def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r'\d+', '', texto)
    
    # Tokenizar, eliminar stopwords y aplicar Stemming
    palabras = texto.split()
    palabras_procesadas = [stemmer.stem(p) for p in palabras if p not in stopwords_es]
    return " ".join(palabras_procesadas)

textos_limpios = [limpiar_texto(t) for t in Corpus]

# Usamos TfidfVectorizer en lugar de CountVectorizer
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(textos_limpios)

modelo = MultinomialNB()
modelo.fit(X, etiquetas)

print("¡Modelo entrenado con TF-IDF usando todo el corpus!")

while True:
    nuevo_texto = input("\nIngrese un texto (o escribe 'salir' para terminar): ")
    if nuevo_texto.lower().strip() in ['salir', 'exit', 'quit']:
        break
    if not nuevo_texto.strip():
        continue
        
    texto_limpio = [limpiar_texto(nuevo_texto)]
    X_nuevo = vectorizador.transform(texto_limpio)
    prediccion = modelo.predict(X_nuevo)
    
    print("Predicción para nuevo texto:")
    print("Positivo" if prediccion[0] == 1 else "Negativo")
