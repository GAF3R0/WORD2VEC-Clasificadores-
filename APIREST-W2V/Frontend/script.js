const API_BASE_URL = 'http://localhost:5000';

let allVocabWords = [];
let allEmbeddings = [];
let currentVector = null;
let selectedCSVFile = null;
let currentModelConfig = null;

// Cargar corpus y renderizar la tabla
async function loadcorpus() {
    const url = `${API_BASE_URL}/corpus/all`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error('Error HTTP: ' + response.status);
        }

        const tableBody = document.getElementById('corpus-table-body');
        if (!tableBody) {
            throw new Error('No se encontró el elemento con id="corpus-table-body"');
        }

        tableBody.innerHTML = '';

        if (data.corpus && Array.isArray(data.corpus)) {
            // Actualizar contador del corpus en el header
            const corpusCount = document.getElementById('corpus-count-value');
            if (corpusCount) {
                corpusCount.textContent = data.corpus.length + (data.corpus.length === 1 ? ' frase' : ' frases');
            }

            data.corpus.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><span class="phrase-index-badge">${item.index}</span></td>
                    <td>${escapeHtml(item.phrase)}</td>
                    <td>
                        <div class="actions-cell">
                            <button class="btn-text btn-edit" title="Editar">Editar</button>
                            <button class="btn-text btn-delete" title="Eliminar">Eliminar</button>
                        </div>
                    </td>
                `;

                // Event listener para editar
                tr.querySelector('.btn-edit').addEventListener('click', () => {
                    startEditing(item.index, item.phrase);
                });

                // Event listener para eliminar
                tr.querySelector('.btn-delete').addEventListener('click', () => {
                    deletePhrase(item.index);
                });

                tableBody.appendChild(tr);
            });
        }
    } catch (error) {
        console.error("Error al cargar el corpus: " + error.message);
    }
}

// Escapar HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Iniciar modo edición
function startEditing(index, phrase) {
    const indexInput = document.getElementById('phrase-index-input');
    const phraseInput = document.getElementById('phrase-input');
    const saveBtnText = document.getElementById('save-btn-text');
    const saveBtn = document.getElementById('btn-save-phrase');
    const cancelBtn = document.getElementById('btn-cancel-edit');

    indexInput.value = index;
    phraseInput.value = phrase;
    saveBtnText.textContent = 'Guardar';
    saveBtn.className = 'btn btn-primary'; // Cambia a azul para denotar guardar
    cancelBtn.classList.remove('hidden');

    phraseInput.focus();
}

// Cancelar modo edición
function cancelEditing() {
    const indexInput = document.getElementById('phrase-index-input');
    const phraseInput = document.getElementById('phrase-input');
    const saveBtnText = document.getElementById('save-btn-text');
    const saveBtn = document.getElementById('btn-save-phrase');
    const cancelBtn = document.getElementById('btn-cancel-edit');

    indexInput.value = '';
    phraseInput.value = '';
    saveBtnText.textContent = 'Agregar';
    saveBtn.className = 'btn btn-success'; // Vuelve a verde
    cancelBtn.classList.add('hidden');
}

// Guardar o Agregar frase y activar re-entrenamiento automático
async function savePhrase(event) {
    event.preventDefault();
    const indexInput = document.getElementById('phrase-index-input');
    const phraseInput = document.getElementById('phrase-input');
    const phrase = phraseInput.value.trim();
    const index = indexInput.value;

    if (!phrase) return;

    try {
        let response;
        if (index !== '') {
            // Edición
            response = await fetch(`${API_BASE_URL}/corpus/update`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: parseInt(index), phrase })
            });
        } else {
            // Creación
            response = await fetch(`${API_BASE_URL}/corpus/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phrase })
            });
        }

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Error al guardar la frase');
        }

        cancelEditing();
        
        // Auto-entrenamiento en segundo plano y actualización silenciosa
        await autoTrainModel();
    } catch (error) {
        alert("Error: " + error.message);
    }
}

// Eliminar frase y activar re-entrenamiento automático
async function deletePhrase(index) {
    if (!confirm('¿Estás seguro de que deseas eliminar esta frase del corpus?')) return;

    try {
        const response = await fetch(`${API_BASE_URL}/corpus/delete`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: parseInt(index) })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Error al eliminar la frase');
        }

        // Auto-entrenamiento en segundo plano y actualización silenciosa
        await autoTrainModel();
    } catch (error) {
        alert("Error: " + error.message);
    }
}

// Re-entrenamiento automático y actualización silenciosa del modelo
async function autoTrainModel() {
    const cardHeader = document.querySelector('.vocab-card .card-header');
    let indicator = document.getElementById('vocab-loading-indicator');
    
    if (cardHeader && !indicator) {
        indicator = document.createElement('span');
        indicator.id = 'vocab-loading-indicator';
        indicator.className = 'loading-indicator';
        indicator.textContent = 'Actualizando modelo...';
        
        const h2 = cardHeader.querySelector('h2');
        if (h2) {
            h2.appendChild(indicator);
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}/corpus/train`, {
            method: 'POST'
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error en el entrenamiento del modelo');
        }

        // Refrescar todos los datos
        await countwords();
        await allwords();
        await loadcorpus();
        await loadEmbeddings();
    } catch (error) {
        console.error("Error al auto-entrenar modelo: " + error.message);
    } finally {
        if (indicator) {
            indicator.remove();
        }
    }
}

// Obtener el número de palabras en el vocabulario
async function countwords() {
    const wordCount = document.getElementById('vocab-count-value');
    if (!wordCount) return;
    const url = `${API_BASE_URL}/word/count`;

    try {
        const response = await fetch(url);
        const data = await response.json();
        if (!response.ok) {
            throw new Error('Error HTTP: ' + response.status);
        }
        if (data.words !== undefined) {
            wordCount.textContent = data.words + (data.words === 1 ? ' palabra' : ' palabras');
        }
    } catch (error) {
        console.error("Error al contar palabras: " + error.message);
    }
}

// Obtener todas las palabras y renderizar el vocabulario
async function allwords() {
    const url = `${API_BASE_URL}/words/all`;
    const container = document.getElementById('vocab-words-container');

    if (!container) return;

    try {
        const response = await fetch(url);
        const data = await response.json();
        if (!response.ok) {
            throw new Error('Error HTTP: ' + response.status);
        }

        if (data.words && Array.isArray(data.words)) {
            allVocabWords = data.words;
            renderVocabulary(allVocabWords);
        }
    } catch (error) {
        console.error("Error al cargar el vocabulario: " + error.message);
        container.innerHTML = `<p style="color: var(--danger); font-size: 13px; text-align: center; width: 100%;">Error al cargar vocabulario.</p>`;
    }
}

// Renderizar palabras del vocabulario
function renderVocabulary(words) {
    const container = document.getElementById('vocab-words-container');
    const countBadge = document.getElementById('vocab-visible-count');
    if (!container) return;

    container.innerHTML = '';

    if (words.length === 0) {
        container.innerHTML = `<p style="color: var(--text-muted); font-size: 13px; padding: 10px; width: 100%; text-align: center;">No hay palabras que coincidan.</p>`;
        if (countBadge) countBadge.textContent = `0 de ${allVocabWords.length}`;
        return;
    }

    words.forEach(word => {
        const pill = document.createElement('span');
        pill.className = 'vocab-word-pill';
        pill.textContent = word;
        pill.addEventListener('click', () => {
            selectWord(word);
        });
        container.appendChild(pill);
    });

    if (countBadge) {
        countBadge.textContent = `${words.length} de ${allVocabWords.length}`;
    }
}

// Filtrar vocabulario
function filterVocabulary() {
    const query = document.getElementById('vocab-search-input').value.trim().toLowerCase();
    if (!query) {
        renderVocabulary(allVocabWords);
        return;
    }

    const filtered = allVocabWords.filter(word => word.toLowerCase().includes(query));
    renderVocabulary(filtered);
}

// Seleccionar palabra del vocabulario
function selectWord(word) {
    const searchInput = document.getElementById('word-search-input');
    if (searchInput) {
        searchInput.value = word;
    }
    queryWord(word);
}

// Ejecutar consultas para una palabra
async function queryWord(word) {
    if (!word) return;

    const placeholder = document.getElementById('results-placeholder');
    const termDisplay = document.getElementById('search-term-display');
    
    if (placeholder) placeholder.classList.add('hidden');
    if (termDisplay) {
        termDisplay.textContent = word;
        termDisplay.classList.remove('hidden');
    }

    // Ejecutar consultas secuencialmente
    await getWordVector(word);
    await getSimilarWords(word);
}

// Consultar vector de palabra
async function getWordVector(word) {
    const vectorContainer = document.getElementById('vector-container');
    const vectorResultContainer = document.getElementById('vector-result-container');
    
    if (!vectorContainer || !vectorResultContainer) return;

    try {
        const response = await fetch(`${API_BASE_URL}/vector?word=${encodeURIComponent(word)}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Palabra no encontrada');
        }

        vectorContainer.innerHTML = '';
        currentVector = data.vector;

        if (Array.isArray(data.vector)) {
            data.vector.forEach((val, idx) => {
                const cell = document.createElement('div');
                cell.className = 'vector-cell';
                cell.title = `Dimensión ${idx + 1}`;
                cell.textContent = val.toFixed(4);
                vectorContainer.appendChild(cell);
            });
            vectorResultContainer.classList.remove('hidden');
        }
    } catch (error) {
        vectorContainer.innerHTML = `<p style="grid-column: span 5; color: var(--danger); font-size: 13px; text-align: center;">${error.message}</p>`;
        vectorResultContainer.classList.remove('hidden');
        currentVector = null;
    }
}

// Consultar palabras similares
async function getSimilarWords(word) {
    const similarContainer = document.getElementById('similar-words-container');
    const similarResultContainer = document.getElementById('similar-result-container');

    if (!similarContainer || !similarResultContainer) return;

    try {
        const response = await fetch(`${API_BASE_URL}/similary?word=${encodeURIComponent(word)}&topn=5`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Palabra no encontrada');
        }

        similarContainer.innerHTML = '';

        if (data.similar && Array.isArray(data.similar)) {
            data.similar.forEach(item => {
                const simWord = item[0];
                const score = item[1];
                const percentage = (score * 100).toFixed(1);

                const itemDiv = document.createElement('div');
                itemDiv.className = 'similar-item';
                itemDiv.innerHTML = `
                    <div class="similar-item-header">
                        <span class="similar-word-name" style="cursor: pointer; font-weight: 500; color: var(--primary);">${escapeHtml(simWord)}</span>
                        <span class="similar-score">${percentage}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: ${percentage}%;"></div>
                    </div>
                `;

                // Clic en palabra similar realiza nueva consulta
                itemDiv.querySelector('.similar-word-name').addEventListener('click', () => {
                    selectWord(simWord);
                });

                similarContainer.appendChild(itemDiv);
            });
            similarResultContainer.classList.remove('hidden');
        }
    } catch (error) {
        similarContainer.innerHTML = `<p style="color: var(--danger); font-size: 13px; padding: 10px;">${error.message}</p>`;
        similarResultContainer.classList.remove('hidden');
    }
}

// Copiar vector al portapapeles
async function copyVectorToClipboard() {
    if (!currentVector) return;

    const btn = document.getElementById('btn-copy-vector');

    try {
        const textToCopy = currentVector.join(', ');
        await navigator.clipboard.writeText(textToCopy);

        btn.textContent = '¡Copiado!';
        btn.classList.add('copied');

        setTimeout(() => {
            btn.textContent = 'Copiar';
            btn.classList.remove('copied');
        }, 1500);
    } catch (err) {
        console.error('No se pudo copiar el vector: ', err);
    }
}

// Carga inicial y Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Carga inicial
    loadcorpus();
    countwords();
    allwords();
    loadEmbeddings();
    initCSVUpload();
    loadModelConfig();
    initModelConfigPanel();

    // Formulario de corpus
    const phraseForm = document.getElementById('phrase-form');
    if (phraseForm) {
        phraseForm.addEventListener('submit', savePhrase);
    }

    const cancelEditBtn = document.getElementById('btn-cancel-edit');
    if (cancelEditBtn) {
        cancelEditBtn.addEventListener('click', cancelEditing);
    }

    // Buscador de vocabulario (filtrado)
    const vocabSearch = document.getElementById('vocab-search-input');
    if (vocabSearch) {
        vocabSearch.addEventListener('input', filterVocabulary);
    }

    // Búsqueda de palabra en el explorador
    const btnGetVector = document.getElementById('btn-get-vector');
    const btnGetSimilar = document.getElementById('btn-get-similar');
    const wordSearchInput = document.getElementById('word-search-input');

    if (btnGetVector && wordSearchInput) {
        btnGetVector.addEventListener('click', () => {
            const word = wordSearchInput.value.trim();
            if (word) {
                const placeholder = document.getElementById('results-placeholder');
                if (placeholder) placeholder.classList.add('hidden');
                
                const termDisplay = document.getElementById('search-term-display');
                if (termDisplay) {
                    termDisplay.textContent = word;
                    termDisplay.classList.remove('hidden');
                }
                
                getWordVector(word);
                
                const similarResultContainer = document.getElementById('similar-result-container');
                if (similarResultContainer) similarResultContainer.classList.add('hidden');
            } else {
                alert('Por favor ingrese una palabra para consultar.');
            }
        });
    }

    if (btnGetSimilar && wordSearchInput) {
        btnGetSimilar.addEventListener('click', () => {
            const word = wordSearchInput.value.trim();
            if (word) {
                const placeholder = document.getElementById('results-placeholder');
                if (placeholder) placeholder.classList.add('hidden');
                
                const termDisplay = document.getElementById('search-term-display');
                if (termDisplay) {
                    termDisplay.textContent = word;
                    termDisplay.classList.remove('hidden');
                }

                getSimilarWords(word);

                const vectorResultContainer = document.getElementById('vector-result-container');
                if (vectorResultContainer) vectorResultContainer.classList.add('hidden');
            } else {
                alert('Por favor ingrese una palabra para consultar.');
            }
        });
    }

    // Copiar vector
    const copyVectorBtn = document.getElementById('btn-copy-vector');
    if (copyVectorBtn) {
        copyVectorBtn.addEventListener('click', copyVectorToClipboard);
    }

    // Buscador de embeddings
    const embeddingsSearch = document.getElementById('embeddings-search-input');
    if (embeddingsSearch) {
        embeddingsSearch.addEventListener('input', filterEmbeddings);
    }

});

// ==========================================
// NUEVAS FUNCIONES PARA EMBEDDINGS
// ==========================================

// Cargar todos los embeddings del backend
async function loadEmbeddings() {
    const url = `${API_BASE_URL}/embeddings/all`;
    try {
        const response = await fetch(url);
        const data = await response.json();
        if (!response.ok) {
            throw new Error('Error HTTP: ' + response.status);
        }
        if (data.embeddings && Array.isArray(data.embeddings)) {
            allEmbeddings = data.embeddings;
            renderEmbeddingsTable(allEmbeddings);
        }
    } catch (error) {
        console.error("Error al cargar embeddings: " + error.message);
        const tableBody = document.getElementById('embeddings-table-body');
        if (tableBody) {
            tableBody.innerHTML = `<tr><td colspan="2" style="color: var(--danger); text-align: center;">Error al cargar embeddings.</td></tr>`;
        }
    }
}

// Renderizar tabla de embeddings
function renderEmbeddingsTable(embeddings) {
    const tableBody = document.getElementById('embeddings-table-body');
    const countBadge = document.getElementById('embeddings-visible-count');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    if (embeddings.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="2" style="color: var(--text-muted); text-align: center; padding: 20px;">No hay embeddings disponibles.</td></tr>`;
        if (countBadge) countBadge.textContent = `0 de ${allEmbeddings.length}`;
        return;
    }

    embeddings.forEach(item => {
        const tr = document.createElement('tr');
        
        // Crear vista previa del vector (100 dimensiones)
        const vectorContainer = document.createElement('div');
        vectorContainer.className = 'vector-preview-container';
        
        item.vector.forEach((val, idx) => {
            const cell = document.createElement('span');
            cell.className = 'vector-preview-cell';
            cell.title = `Dimensión ${idx + 1}`;
            cell.textContent = val.toFixed(4);
            vectorContainer.appendChild(cell);
        });

        // Fila
        tr.innerHTML = `
            <td style="cursor: pointer; color: var(--primary);" title="Haga clic para consultar esta palabra">${escapeHtml(item.word)}</td>
            <td></td>
        `;

        // Insertar el contenedor del vector en el segundo td
        tr.cells[1].appendChild(vectorContainer);

        // Event listener para que al hacer clic en la palabra, se consulte
        tr.cells[0].addEventListener('click', () => {
            selectWord(item.word);
        });

        tableBody.appendChild(tr);
    });

    if (countBadge) {
        countBadge.textContent = `${embeddings.length} de ${allEmbeddings.length}`;
    }
}

// Filtrar la tabla de embeddings
function filterEmbeddings() {
    const query = document.getElementById('embeddings-search-input').value.trim().toLowerCase();
    if (!query) {
        renderEmbeddingsTable(allEmbeddings);
        return;
    }

    const filtered = allEmbeddings.filter(item => item.word.toLowerCase().includes(query));
    renderEmbeddingsTable(filtered);
}

// ==========================================
// NUEVAS FUNCIONES PARA CARGA DE CSV
// ==========================================

function initCSVUpload() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const cancelBtn = document.getElementById('btn-cancel-csv');
    const trainBtn = document.getElementById('btn-train-csv');

    if (!dropzone || !fileInput || !cancelBtn || !trainBtn) return;

    // Abrir selector al hacer clic en dropzone
    dropzone.addEventListener('click', () => fileInput.click());

    // Eventos drag & drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleCSVFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleCSVFile(fileInput.files[0]);
        }
    });

    cancelBtn.addEventListener('click', resetCSVUpload);
    trainBtn.addEventListener('click', trainWithCSV);
}

// Procesar archivo seleccionado
async function handleCSVFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Por favor, seleccione un archivo con formato .csv');
        return;
    }

    selectedCSVFile = file;

    // Cambiar mensaje a cargando
    const messageSpan = document.querySelector('#csv-dropzone .dropzone-message');
    const originalMessage = messageSpan.textContent;
    messageSpan.textContent = 'Analizando archivo CSV...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/corpus/analyze_csv`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Error al analizar el CSV');
        }

        // Mostrar opciones y ocultar dropzone
        document.getElementById('csv-dropzone').classList.add('hidden');
        document.getElementById('csv-options-container').classList.remove('hidden');

        // Detalles del archivo
        document.getElementById('selected-file-name').textContent = data.filename;
        document.getElementById('selected-file-size').textContent = (file.size / 1024).toFixed(1) + ' KB';
        document.getElementById('selected-file-rows').textContent = data.row_count + (data.row_count === 1 ? ' fila' : ' filas');

        // Habilitar botón de entrenamiento
        document.getElementById('btn-train-csv').disabled = false;

        // Tabla de vista previa
        const thead = document.getElementById('csv-preview-thead');
        const tbody = document.getElementById('csv-preview-tbody');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (data.preview && data.preview.length > 0) {
            // Headers
            const headerRow = document.createElement('tr');
            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            // Filas
            data.preview.forEach(row => {
                const tr = document.createElement('tr');
                data.columns.forEach(col => {
                    const td = document.createElement('td');
                    td.textContent = row[col] !== undefined ? row[col] : '';
                    td.title = row[col] !== undefined ? row[col] : '';
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }
    } catch (error) {
        alert('Error: ' + error.message);
        resetCSVUpload();
    } finally {
        messageSpan.textContent = originalMessage;
    }
}

// Cancelar y limpiar formulario CSV
function resetCSVUpload() {
    selectedCSVFile = null;
    document.getElementById('csv-file-input').value = '';
    document.getElementById('csv-dropzone').classList.remove('hidden');
    document.getElementById('csv-options-container').classList.add('hidden');
    document.getElementById('csv-preview-thead').innerHTML = '';
    document.getElementById('csv-preview-tbody').innerHTML = '';
    document.getElementById('btn-train-csv').disabled = true;
}

// Enviar archivo y columna seleccionada para entrenar modelo
async function trainWithCSV() {
    if (!selectedCSVFile) return;

    const trainBtn = document.getElementById('btn-train-csv');
    const cancelBtn = document.getElementById('btn-cancel-csv');
    const originalBtnText = trainBtn.textContent;

    trainBtn.disabled = true;
    cancelBtn.disabled = true;
    trainBtn.textContent = 'Entrenando modelo...';

    const formData = new FormData();
    formData.append('file', selectedCSVFile);

    try {
        const response = await fetch(`${API_BASE_URL}/corpus/train_csv`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Error en el entrenamiento');
        }

        alert('¡Entrenamiento Completado exitosamente!\n' + data.message + '\nVocabulario: ' + data.vocabulary_size + ' palabras.');
        resetCSVUpload();

        // Refrescar todos los datos en el dashboard
        await countwords();
        await allwords();
        await loadcorpus();
        await loadEmbeddings();

    } catch (error) {
        alert('Error al entrenar con CSV: ' + error.message);
    } finally {
        trainBtn.disabled = false;
        cancelBtn.disabled = false;
        trainBtn.textContent = originalBtnText;
    }
}

// ==========================================
// NUEVAS FUNCIONES PARA CONFIGURACIÓN
// ==========================================

// Cargar configuración de entrenamiento de Word2Vec desde el backend
async function loadModelConfig() {
    try {
        const response = await fetch(`${API_BASE_URL}/config/model`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error al obtener la configuración.');
        }
        
        currentModelConfig = data;
        populateConfigForm(currentModelConfig);
    } catch (error) {
        console.error("Error al cargar la configuración: " + error.message);
    }
}

// Llenar el formulario con los parámetros cargados
function populateConfigForm(config) {
    if (!config) return;
    
    const vectorSize = config.vector_size || 100;
    document.getElementById('config-vector-size').value = vectorSize;
    document.getElementById('config-window').value = config.window || 10;
    document.getElementById('config-min-count').value = config.min_count || 1;
    document.getElementById('config-sg').value = config.sg !== undefined ? config.sg : 1;
    
    // Actualizar dinámicamente los textos de dimensiones en la UI
    const embeddingsHeader = document.getElementById('lbl-embeddings-dimensions-header');
    if (embeddingsHeader) {
        embeddingsHeader.textContent = `Vector (Vista Previa de ${vectorSize} Dimensiones)`;
    }
    const resultHeader = document.getElementById('lbl-vector-dimensions-header');
    if (resultHeader) {
        resultHeader.textContent = `Vector de Palabra (${vectorSize} dimensiones)`;
    }
}

// Inicializar panel de configuración
function initModelConfigPanel() {
    const toggleBtn = document.getElementById('btn-toggle-config');
    const configCard = document.getElementById('config-card');
    const configForm = document.getElementById('config-form');
    const resetBtn = document.getElementById('btn-reset-config');
    
    if (!toggleBtn || !configCard || !configForm || !resetBtn) return;
    
    // Toggle visibilidad
    toggleBtn.addEventListener('click', () => {
        configCard.classList.toggle('hidden');
        if (!configCard.classList.contains('hidden') && currentModelConfig) {
            populateConfigForm(currentModelConfig);
        }
    });
    
    // Resetear formulario
    resetBtn.addEventListener('click', () => {
        if (currentModelConfig) {
            populateConfigForm(currentModelConfig);
        } else {
            loadModelConfig();
        }
    });
    
    // Guardar cambios
    configForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const vector_size = parseInt(document.getElementById('config-vector-size').value);
        const window = parseInt(document.getElementById('config-window').value);
        const min_count = parseInt(document.getElementById('config-min-count').value);
        const sg = parseInt(document.getElementById('config-sg').value);
        
        const saveBtn = document.getElementById('btn-save-config');
        const originalText = saveBtn.textContent;
        saveBtn.disabled = true;
        saveBtn.textContent = 'Guardando...';
        
        try {
            const response = await fetch(`${API_BASE_URL}/config/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ vector_size, window, min_count, sg })
            });
            
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Error al guardar configuración');
            }
            
            currentModelConfig = data.config;
            alert(data.message);
            configCard.classList.add('hidden'); // Ocultar al guardar con éxito
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = originalText;
        }
    });
}
