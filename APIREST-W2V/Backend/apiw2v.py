import flask
import os
import json
from flask import request, jsonify
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# General del Configuración del modelo Word2Vec
stop_words = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = simple_preprocess(text, deacc=True)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

# Cargar el modelo pre-entrenado de w2v desde el archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'word2vec_model.bin')
model = Word2Vec.load(model_path)

# Cargar el corpus desde el archivo JSON si existe
corpus_json_path = os.path.join(script_dir, 'corpus.json')
if os.path.exists(corpus_json_path):
    with open(corpus_json_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
else:
    corpus = []

# Creación de la App Flask
app = flask.Flask(__name__)

# Configuración de CORS para permitir peticiones desde el Frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Ruta principal
@app.route('/')
def index():
    return "API CINVESTAV de Word2Vec"

# Metodos GET

# Obtener las Frases del Corpus (sin procesar) con sus respectivos índices
@app.route('/corpus/all', methods=['GET'])
def get_all_sentences():
    corpus_with_indices = [{"index": i, "phrase": sentence} for i, sentence in enumerate(corpus)]
    return jsonify({'corpus': corpus_with_indices})

# Obtener todas las palabras
@app.route('/words/all', methods=['GET'])
def get_all_words():
    words = model.wv.index_to_key
    return jsonify({'words': words})

# Obtener el número de palabras 
@app.route('/word/count', methods=['GET'])
def get_word_count():
    words = model.wv.index_to_key
    return jsonify({'words': len(words)})

# Obtener palabras similares
@app.route('/similary', methods=['GET'])
def get_similar():
    word = request.args.get('word', '')
    topn = int(request.args.get('topn', 5))
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=topn)
        return jsonify({'word': word, 'similar': similar})
    else:
        return jsonify({'error': 'Palabra no encontrada en el vocabulario'}), 404
 
# Obtener vectores
@app.route('/vector', methods=['GET'])
def get_vector():
    word = request.args.get('word', '')
    if word in model.wv:
        vector = model.wv[word]
        return jsonify({'word': word, 'vector': vector.tolist()})
    else:
        return jsonify({'error': 'Palabra no encontrada en el vocabulario'}), 404

# Obtener todos los embeddings (vocabulario + vectores)
@app.route('/embeddings/all', methods=['GET'])
def get_all_embeddings():
    embeddings = []
    for word in model.wv.index_to_key:
        embeddings.append({
            'word': word,
            'vector': model.wv[word].tolist()
        })
    return jsonify({'embeddings': embeddings})


# Metodos POST

# Agregar una nueva frase al corpus
@app.route('/corpus/add', methods=['POST'])
def add_sentence():
    data = request.get_json()
    if not data or 'phrase' not in data:
        return jsonify({'error': 'Falta el parámetro "phrase" en el cuerpo de la solicitud.'}), 400
    
    phrase = data['phrase']
    
    # Registrar la frase en el corpus
    corpus.append(phrase)
    with open(corpus_json_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
        
    # Procesar (tokenizar/limpiar) la frase
    tokens = preprocess_text(phrase)
    
    return jsonify({
        'message': 'Frase registrada exitosamente en el corpus.',
        'phrase': phrase,
        'processed_tokens': tokens
    }), 201

# Procesar/Re-entrenar el modelo con el corpus actualizado
@app.route('/corpus/train', methods=['POST'])
def train_corpus():
    global model
    if not corpus:
        return jsonify({'error': 'El corpus está vacío.'}), 400
        
    # Preprocesar todo el corpus actualizado
    tokenized = [preprocess_text(sentence) for sentence in corpus]
    
    # Re-entrenar el modelo Word2Vec
    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=10,
        min_count=1,
        workers=1,
        sg=1
    )
    
    model.save(model_path)
    
    return jsonify({
        'message': 'Modelo Word2Vec re-entrenado exitosamente con el corpus actualizado.',
        'vocabulary_size': len(model.wv.index_to_key)
    }), 200

# Guardar los embeddings en el servidor
@app.route('/embeddings/save', methods=['POST'])
def save_embeddings():
    try:
        embeddings_txt_path = os.path.join(script_dir, 'embeddings.txt')
        model.wv.save_word2vec_format(embeddings_txt_path)
        return jsonify({
            'message': 'Embeddings guardados exitosamente en el servidor.',
            'path': embeddings_txt_path,
            'filename': 'embeddings.txt'
        }), 200
    except Exception as e:
        return jsonify({'error': f'Error al guardar embeddings: {str(e)}'}), 500


# Metodos PUT

# Actualizar una frase en el corpus
@app.route('/corpus/update', methods=['PUT'])
def update_sentence():
    data = request.get_json()
    if not data or 'phrase' not in data or 'index' not in data:
        return jsonify({'error': 'Falta el parámetro "phrase" o "index" en el cuerpo de la solicitud.'}), 400
    
    phrase = data['phrase']
    
    # Validar y convertir index a entero
    try:
        index = int(data['index'])
    except (ValueError, TypeError):
        return jsonify({'error': 'El parámetro "index" debe ser un número entero válido.'}), 400
    
    # Verificar límites del índice
    if index < 0 or index >= len(corpus):
        return jsonify({'error': f'Índice fuera de rango. El corpus tiene actualmente {len(corpus)} frases.'}), 400
    
    # Actualizar la frase en el corpus
    corpus[index] = phrase
    with open(corpus_json_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
        
    # Procesar (tokenizar/limpiar) la frase
    tokens = preprocess_text(phrase)
    
    return jsonify({
        'message': 'Frase actualizada exitosamente en el corpus.',
        'phrase': phrase,
        'processed_tokens': tokens
    }), 200

# Metodos DELETE

# Eliminar una frase del corpus
@app.route('/corpus/delete', methods=['DELETE'])
def delete_sentence():
    data = request.get_json()
    if not data or 'index' not in data:
        return jsonify({'error': 'Falta el parámetro "index" en el cuerpo de la solicitud.'}), 400
    
    # Validar y convertir index a entero
    try:
        index = int(data['index'])
    except (ValueError, TypeError):
        return jsonify({'error': 'El parámetro "index" debe ser un número entero válido.'}), 400
        
    # Verificar límites del índice
    if index < 0 or index >= len(corpus):
        return jsonify({'error': f'Índice fuera de rango. El corpus tiene actualmente {len(corpus)} frases.'}), 400
    
    # Eliminar la frase del corpus
    del corpus[index]
    with open(corpus_json_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
        
    return jsonify({
        'message': 'Frase eliminada exitosamente del corpus.'
    }), 200


if __name__ == '__main__':
    app.run(debug=True)