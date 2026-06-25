import nltk
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext
from gensim import corpora, models
from gensim.utils import simple_preprocess 
from nltk.corpus import stopwords

# Solo descargamos stopwords de NLTK ya que es lo único que usamos
nltk.download('stopwords', quiet=True)

# Corpus
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
    "El perro atrapa al zorro marrón de la cola y no lo suelta porque lo quiere mucho.",
    "EL zorro vive en el bosque",
    "El perro vive en la ciudad",
    "La casa del perro es grande y bonita",
    "La casa del zorro es pequeña y acogedora",
    "El gato negro duerme en el techo"

]
# Cargar listado de stopwords en español
stopwords_es = set(stopwords.words('spanish'))

# Preprocesar corpus: convertir a minúsculas, tokenizar, quitar puntuación y filtrar stopwords
def preprocess(text):
    tokens = simple_preprocess(text)
    return [token for token in tokens if token not in stopwords_es]

tokenized_corpus = [preprocess(doc) for doc in corpus]

# Creacion de diccionario para TF-IDF y BoW
dictionary = corpora.Dictionary(tokenized_corpus)

def w2v(corpus):
    model = models.Word2Vec(
        sentences = corpus,
        vector_size = 30,
        window = 10,
        min_count = 1,
        workers = 1,
        sg = 1
    )
    return model

def tfidf(corpus):
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return tfidf_corpus

def bow(corpus):
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    return bow_corpus

# Calcular matrices TF-IDF y BoW
df_tfidf = pd.DataFrame([
    {dictionary[id] : float(value) for id, value in doc}
    for doc in tfidf(tokenized_corpus)
]).fillna(0)

df_bow = pd.DataFrame([
    {dictionary[id] : fred for id, fred in doc}
    for doc in bow(tokenized_corpus)
]).fillna(0)

# Entrenar modelo Word2Vec
model = w2v(tokenized_corpus)

# ----------------- Ventana Gráfica (Tkinter) -----------------
root = tk.Tk()
root.title("Visualizador de Modelos NLP (TF-IDF, BoW y Word2Vec)")
root.geometry("1400x650")

# Paleta de colores oscuros (Estilo moderno)
BG_COLOR = "#21272d"       
TEXT_COLOR = "#d8dee9"    
ACCENT_COLOR = "#88c0d0"  
CARD_BG = "#2e3440"        

root.configure(bg=BG_COLOR)

# Estilos ttk
style = ttk.Style()
style.theme_use('clam')
style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
style.configure("TNotebook.Tab", background=CARD_BG, foreground=TEXT_COLOR, padding=[20, 6], font=("Segoe UI", 10, "bold"))
style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", "#2e3440")])
style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 11))
style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=ACCENT_COLOR)

# Contenedor de pestañas
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=15, pady=15)

# --- Pestaña 1: TF-IDF ---
tab_tfidf = ttk.Frame(notebook)
notebook.add(tab_tfidf, text="  TF-IDF Matrix  ")

lbl_tfidf = ttk.Label(tab_tfidf, text="Matriz de Frecuencia Inversa (TF-IDF)", style="Header.TLabel")
lbl_tfidf.pack(anchor="w", padx=15, pady=10)

txt_tfidf = scrolledtext.ScrolledText(tab_tfidf, font=("Consolas", 10), bg=CARD_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR, borderwidth=0)
txt_tfidf.pack(fill="both", expand=True, padx=15, pady=10)
txt_tfidf.insert(tk.END, df_tfidf.to_string())
txt_tfidf.configure(state="disabled")

# --- Pestaña 2: Bag of Words (BoW) ---
tab_bow = ttk.Frame(notebook)
notebook.add(tab_bow, text="  Bag of Words (BoW)  ")

lbl_bow = ttk.Label(tab_bow, text="Matriz de Bolsa de Palabras (BoW)", style="Header.TLabel")
lbl_bow.pack(anchor="w", padx=15, pady=10)

txt_bow = scrolledtext.ScrolledText(tab_bow, font=("Consolas", 10), bg=CARD_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR, borderwidth=0)
txt_bow.pack(fill="both", expand=True, padx=15, pady=10)
txt_bow.insert(tk.END, df_bow.to_string())
txt_bow.configure(state="disabled")

# --- Pestaña 3: Word2Vec ---
tab_w2v = ttk.Frame(notebook)
notebook.add(tab_w2v, text="  Word2Vec Model  ")

lbl_w2v = ttk.Label(tab_w2v, text="Representación Vectorial y Similitud Word2Vec", style="Header.TLabel")
lbl_w2v.pack(anchor="w", padx=15, pady=10)

# Vector de la palabra zorro
lbl_vec_title = ttk.Label(tab_w2v, text="Vector de la palabra 'zorro' (30 dimensiones):", font=("Segoe UI", 11, "bold"))
lbl_vec_title.pack(anchor="w", padx=15, pady=5)

vector_text = str(model.wv['zorro'])
txt_vector = scrolledtext.ScrolledText(tab_w2v, height=5, font=("Consolas", 10), bg=CARD_BG, fg=TEXT_COLOR, borderwidth=0)
txt_vector.pack(fill="x", padx=15, pady=5)
txt_vector.insert(tk.END, vector_text)
txt_vector.configure(state="disabled")

# Palabras similares
lbl_sim_title = ttk.Label(tab_w2v, text="Palabras más similares a 'zorro':", font=("Segoe UI", 11, "bold"))
lbl_sim_title.pack(anchor="w", padx=15, pady=10)

frame_sim = tk.Frame(tab_w2v, bg=BG_COLOR)
frame_sim.pack(fill="both", expand=True, padx=15, pady=5)

for word, sim in model.wv.most_similar('zorro', topn=5):
    row_frame = tk.Frame(frame_sim, bg=CARD_BG, pady=8, padx=15)
    row_frame.pack(fill="x", pady=3)
    
    lbl_w = tk.Label(row_frame, text=word, font=("Segoe UI", 11, "bold"), bg=CARD_BG, fg=ACCENT_COLOR, width=15, anchor="w")
    lbl_w.pack(side="left")
    
    lbl_s = tk.Label(row_frame, text=f"{sim:.4f} (Similitud: {sim*100:.2f}%)", font=("Segoe UI", 11), bg=CARD_BG, fg=TEXT_COLOR)
    lbl_s.pack(side="right")

root.mainloop()