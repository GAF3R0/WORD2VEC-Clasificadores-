import flask
import os
import json
import io
import pandas as pd
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

# Cargar la configuración del modelo si existe, de lo contrario usar valores por defecto
config_path = os.path.join(script_dir, 'config.json')
default_config = {
    'vector_size': 100,
    'window': 10,
    'min_count': 1,
    'workers': 1,
    'sg': 1
}

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            model_config = json.load(f)
        except Exception:
            model_config = default_config.copy()
else:
    model_config = default_config.copy()

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
        vector_size=model_config.get('vector_size', 100),
        window=model_config.get('window', 10),
        min_count=model_config.get('min_count', 1),
        workers=model_config.get('workers', 1),
        sg=model_config.get('sg', 1)
    )
    
    model.save(model_path)
    
    return jsonify({
        'message': 'Modelo Word2Vec re-entrenado exitosamente con el corpus actualizado.',
        'vocabulary_size': len(model.wv.index_to_key)
    }), 200

# Analizar un archivo CSV para obtener sus columnas y una vista previa
@app.route('/corpus/analyze_csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo CSV en la solicitud.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío.'}), 400
    
    try:
        file_bytes = file.read()
        
        # Intentar leer CSV con detección automática de separador
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='latin-1')
        
        columns = list(df.columns)
        row_count = len(df)
        
        # Rellenar NaN con cadena vacía para evitar problemas de JSON
        preview_df = df.head(5).fillna('')
        preview = preview_df.to_dict(orient='records')
        
        return jsonify({
            'filename': file.filename,
            'columns': columns,
            'row_count': row_count,
            'preview': preview
        }), 200
    except Exception as e:
        return jsonify({'error': f'Error al analizar el archivo CSV: {str(e)}'}), 400

# Entrenar el modelo con los datos del CSV (primera columna por defecto)
@app.route('/corpus/train_csv', methods=['POST'])
def train_csv():
    global model, corpus
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo CSV.'}), 400
        
    file = request.files['file']
    column = request.form.get('column', '').strip()
        
    try:
        file_bytes = file.read()
        
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='latin-1')
            
        # Si no se especifica columna, buscar la mejor columna que contenga texto
        if not column:
            if df.columns.empty:
                return jsonify({'error': 'El archivo CSV no contiene columnas válidas.'}), 400
            
            best_column = df.columns[0]
            max_tokens = 0
            for col in df.columns:
                sample = df[col].dropna().head(10).astype(str).tolist()
                tokens_count = sum(len(preprocess_text(s)) for s in sample)
                if tokens_count > max_tokens:
                    max_tokens = tokens_count
                    best_column = col
            column = best_column
            
        if column not in df.columns:
            return jsonify({'error': f'La columna "{column}" no existe en el archivo CSV.'}), 400
            
        # Extraer filas, eliminar nulos y cadenas vacías
        sentences = df[column].dropna().astype(str).str.strip().tolist()
        sentences = [s for s in sentences if s != '']
        
        if not sentences:
            return jsonify({'error': f'La columna "{column}" no contiene datos de texto válidos.'}), 400
            
        # Reemplazar el corpus global y guardarlo
        corpus = sentences
        with open(corpus_json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
            
        # Preprocesar e iniciar entrenamiento
        tokenized = [preprocess_text(sentence) for sentence in corpus]
        
        model = Word2Vec(
            sentences=tokenized,
            vector_size=model_config.get('vector_size', 100),
            window=model_config.get('window', 10),
            min_count=model_config.get('min_count', 1),
            workers=model_config.get('workers', 1),
            sg=model_config.get('sg', 1)
        )
        
        model.save(model_path)
        
        return jsonify({
            'message': 'Modelo Word2Vec entrenado exitosamente con los datos del CSV.',
            'vocabulary_size': len(model.wv.index_to_key),
            'row_count': len(corpus)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error durante el entrenamiento con CSV: {str(e)}'}), 500

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


# Obtener o actualizar la configuración de entrenamiento del modelo
@app.route('/config/model', methods=['GET', 'POST'])
def handle_config():
    global model_config
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos de configuración.'}), 400
        
        try:
            vector_size = int(data.get('vector_size', model_config['vector_size']))
            window = int(data.get('window', model_config['window']))
            min_count = int(data.get('min_count', model_config['min_count']))
            sg = int(data.get('sg', model_config['sg']))
            
            if sg not in [0, 1]:
                return jsonify({'error': 'El parámetro de algoritmo (sg) debe ser 0 o 1.'}), 400
                
            model_config['vector_size'] = vector_size
            model_config['window'] = window
            model_config['min_count'] = min_count
            model_config['sg'] = sg
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=4)
                
            return jsonify({
                'message': 'Configuración del modelo guardada con éxito.',
                'config': model_config
            }), 200
        except ValueError:
            return jsonify({'error': 'Los parámetros deben ser valores numéricos enteros.'}), 400
            
    return jsonify(model_config), 200


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