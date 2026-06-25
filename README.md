# WORD2VEC-Clasificadores-

Repositorio de proyectos de Procesamiento del Lenguaje Natural (PLN).

---

## 📂 Estructura del Repositorio

```
PROYECTO/
├── APIREST-W2V/
│   ├── Frontend/
│   │   ├── Front.html
│   │   └── script.js
│   ├── Backend/
│   │   ├── apiw2v.py
│   │   └── corpus.json
│   ├── w2v.py
│   └── word2vec_model.bin
├── BoW.py
├── tf-idf.py
├── clasificadorTFIDF.py
├── clasificadroBoW.py
├── allmet.py
└── README.md
```

---

## 🚀 Proyectos Disponibles

### 1. Word2Vec con API REST

Interfaz web para gestionar corpus y consultar vectores de palabras.

- **Interfaz:** `APIREST-W2V/Frontend/Front.html`
- **API:** `APIREST-W2V/Backend/apiw2v.py`
- **Modelo:** `word2vec_model.bin`
- **Corpus:** `APIREST-W2V/Backend/corpus.json`

### 2. Word2Vec (Script)

Script principal de Word2Vec.

- **Archivo:** `w2v.py`

### 3. TF-IDF

Implementación de TF-IDF.

- **Archivo:** `tf-idf.py`

### 4. Clasificadores

Clasificadores basados en BoW y TF-IDF.

- **Clasificador BoW:** `clasificadroBoW.py`
- **Clasificador TF-IDF:** `clasificadorTFIDF.py`

### 5. Todos los Métodos

Script que combina todos los métodos.

- **Archivo:** `allmet.py`

---

## 🛠️ Requisitos Previos

- **Python 3.6+**
- **gensim**
- **scikit-learn**
- **numpy**
- **flask**
