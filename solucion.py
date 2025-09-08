# %% [markdown]
# # 🧠 Solución Tarea Semana 3: Visualización de Mapas de Activación en CNN con MNIST
# 
# ## 🎯 Objetivo
# Entrenar una red neuronal convolucional (CNN) para clasificar imágenes de dígitos escritos a mano (MNIST), y visualizar qué características aprende cada capa mediante mapas de activación.
# 
# ---

# %% [markdown]
# ## Setup inicial (importe de módulos)

# %%
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# %%
# Cargar y preparar los datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Agregar dimensión de canal (escala de grises = 1 canal)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"Forma de x_train: {x_train.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de x_test: {x_test.shape}")
print(f"Forma de y_test: {y_test.shape}")

# Visualizar algunas imágenes de ejemplo
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f'Digit: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Ejemplos del dataset MNIST')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 📦 Parte 1: Definición y entrenamiento del modelo
# 
# ### Definición de variables para la arquitectura

# %%
# parámetros de la arquitectura
alto = 28  # altura de la imagen
ancho = 28  # ancho de la imagen
clases = 10  # número de clases (dígitos 0-9)

print(f"Dimensiones de entrada: {alto} x {ancho}")
print(f"Número de clases: {clases}")

# %%
# Definir la entrada de la red 
inputs = Input(shape=(alto, ancho, 1))

# Primer bloque convolucional:
x = layers.Conv2D(8, (3, 3), activation='relu', name='conv1')(inputs)
x = layers.MaxPooling2D((2, 2))(x)  

# Segundo bloque convolucional: 
x = layers.Conv2D(16, (3, 3), activation='relu', name='conv2')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Capa densa para clasificación final
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(clases, activation='softmax')(x)

# Compilacion
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
print(model.summary())

# %%
# Entrenar el modelo por 5 épocas
print("Iniciando entrenamiento...")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
print("\nEntrenamiento completado!")

# %% [markdown]
# ### Preguntas Pt. 1 
# 
# **1. ¿Por qué usamos una entrada de tamaño `(28, 28, 1)`? ¿Qué representa cada dimensión?**
# 
# **Respuesta:** Dimensiones:`(28, 28, 1)`:
# - 28: Altura de la imagen en píxeles
# - 28: Ancho de la imagen en píxeles  
# - 1: Número de canales (escala de grises, solo un canal vs RGB que tendría 3)
# 
# **2. En la primera capa convolucional usamos 8 filtros de tamaño 3×3. ¿Qué significan estos filtros?**
# 
# **Respuesta:** Los filtros son matrices de 3×3 que se deslizan por la imagen para detectar las características básicas (por ejemplo, bordes horizontales, verticales, diagonales, esquinas, etc.) Cada filtro aprende a detectar un patrón específico.
# 
# **3. ¿Qué efecto tiene `MaxPooling2D` sobre la salida de la convolución?**
# 
# **Respuesta:** El maxpooling2d va a reducir las dimensiones espaciales a la mitad tomando el valor máximo de cada ventana 2×2.
# - Reduce el tamaño de los datos
# - Conserva las características más importantes
# - Agrega invariancia a pequeñas traslaciones
# - Reduce el costo computacional
# 
# **4. ¿Por qué la última capa tiene 10 neuronas y qué significa la función `softmax` en este contexto?**
# 
# **Respuesta:** 
# - Una neurona por cada clase. (dígitos 0-9)
# - Softmax convierte las salidas en probabilidades que suman 1, representando la confianza del modelo para cada clase
# 
# **5. Observa el `model.summary()` y explica:**

# %% [markdown]
# Entrada: (28, 28, 1) - Imagen original
# Conv1: (26, 26, 8) - Tras convolución 3x3, se pierden 2 píxeles por lado
# MaxPool1: (13, 13, 8) - Tras pooling 2x2, dimensiones se reducen a la mitad
# Conv2: (11, 11, 16) - Tras segunda convolución 3x3
# MaxPool2: (5, 5, 16) - Tras segundo pooling 2x2
# Flatten: (400,) - 5×5×16 = 400 características lineales
# Dense1: (64,) - Capa densa intermedia
# Dense2: (10,) - Salida final con probabilidades para cada clase
# 
# ### • ¿Cómo cambia el tamaño del tensor?
#   Las dimensiones espaciales (altura×ancho) disminuyen progresivamente
#   mientras que la profundidad (canales) aumenta.
# 
# ### • ¿Por qué disminuyen las dimensiones espaciales?
#   - Convoluciones sin padding reducen el tamaño
#   - MaxPooling reduce las dimensiones a la mitad
#   - Esto permite capturar patrones en diferentes escalas
# 
# ### • ¿Por qué aumenta el número de filtros?
#   - Primeras capas: detectan características simples (pocos filtros)
#   - Capas más profundas: combinan características para patrones complejos (más filtros)
#   - Permite representar jerarquías de características

# %% [markdown]
# ## 🔍 Parte 2: Visualización de mapas de activación

# %%
# Se selecciona una imagen del conjunto de prueba
img_index = 0
img = x_test[img_index:img_index+1]
true_label = y_test[img_index]

# Se muestra la imagen
plt.figure(figsize=(4, 4))
plt.imshow(img[0].reshape(28, 28), cmap='gray')
plt.title(f'Imagen seleccionada - Dígito: {true_label}')
plt.axis('off')
plt.show()

print(f"Forma de la imagen: {img.shape}")

# %%
# Se crea un modelo que devuelva las salidas intermedias de las capas convolucionales
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
feature_model = models.Model(inputs=model.input, outputs=layer_outputs)

print("Capas convolucionales encontradas:")
for i, layer in enumerate(model.layers):
    if 'conv' in layer.name:
        print(f"  {i}: {layer.name} - Salida: {layer.output.shape}")

# Obtiene los mapas de características
feature_maps = feature_model.predict(img)
print(f"\nNúmero de capas convolucionales: {len(feature_maps)}")
for i, fmap in enumerate(feature_maps):
    print(f"Capa {i+1}: {fmap.shape}")

# %%
# Visualiza los mapas de cada capa
for i, fmap in enumerate(feature_maps):
    num_filters = fmap.shape[-1]
    
    # Calcula el número de filas necesarias
    cols = min(8, num_filters)  # Máximo 8 columnas
    rows = (num_filters + cols - 1) // cols 
    
    plt.figure(figsize=(16, rows * 2))
    
    for j in range(num_filters):
        plt.subplot(rows, cols, j+1)
        plt.imshow(fmap[0, :, :, j], cmap='viridis')
        plt.title(f'Filtro {j+1}')
        plt.axis('off')
    
    plt.suptitle(f"Mapas de Activación - Capa Convolucional {i+1} ({num_filters} filtros)", fontsize=16)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 🧠 Pregunta (Parte 2)
# 
# **Describe lo que observas en los mapas de características:**

# %% [markdown]
# # ANÁLISIS DE MAPAS DE ACTIVACIÓN
# 
# ## PRIMERA CAPA CONVOLUCIONAL (Conv1):
# • ¿La primera capa detecta bordes o contornos?
#   SÍ - La primera capa detecta características básicas como:
#   - Bordes horizontales y verticales
#   - Contornos y líneas
#   - Cambios de intensidad en diferentes orientaciones
#   - Cada filtro se especializa en detectar un tipo específico de borde
# 
# ## SEGUNDA CAPA CONVOLUCIONAL (Conv2):
# • ¿La segunda capa comienza a detectar formas más complejas?
#   SÍ - La segunda capa combina los bordes de la primera capa para detectar:
#   - Esquinas y ángulos
#   - Curvas y bucles
#   - Patrones más complejos que forman partes de dígitos
#   - Combinaciones de líneas que forman formas características
# 
# ## REDUCCIÓN DE TAMAÑO CON MaxPooling:
# • ¿Qué tanto se reduce la imagen con las capas de MaxPooling?
#   - Imagen original: 28×28 píxeles
#   - Después del primer MaxPooling: 13×13 píxeles (~75% de reducción en área)
#   - Después del segundo MaxPooling: 5×5 píxeles (~97% de reducción total)
#   - Cada MaxPooling reduce las dimensiones espaciales a la mitad
#   - Se conservan las características más importantes (valores máximos)
# 
# ## INTERPRETACIÓN:
# - Jerarquía de características: simple → compleja
# - Reducción progresiva del tamaño espacial
# - Aumento en la abstracción de las características detectadas
# - Los filtros aprenden automáticamente qué patrones son útiles para clasificar dígitos

# %% [markdown]
# ## 📊 Parte 3: Evaluación del modelo

# %%
# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}")
print(f"Pérdida en el conjunto de prueba: {test_loss:.4f}")

# %%
# Predecir clases
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convertir probabilidades en etiquetas

print(f"Forma de predicciones: {y_pred_probs.shape}")
print(f"Primeras 10 predicciones: {y_pred[:10]}")
print(f"Primeras 10 etiquetas reales: {y_test[:10]}")

# %%
# Matriz de confusión
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap='Blues',
    colorbar=True,
    display_labels=np.arange(10)
)
plt.title("Matriz de Confusión - Clasificación MNIST", fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.show()

# %%
# Reporte de métricas por clase
print("=== REPORTE DE CLASIFICACIÓN ===")
print()
print(classification_report(y_test, y_pred, digits=4))

# %% [markdown]
# ### 🧠 Preguntas (Parte 3)

# %%
# Análisis detallado de los resultados
from sklearn.metrics import confusion_matrix

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

print("=== ANÁLISIS DE RESULTADOS ===")
print()

# 1. Dígitos más fáciles de clasificar
diagonal = np.diag(cm)
totals = np.sum(cm, axis=1)
accuracies = diagonal / totals

print("1. ¿Qué dígitos fueron más fáciles de clasificar?")
best_digits = np.argsort(accuracies)[::-1][:3]
for digit in best_digits:
    acc = accuracies[digit]
    print(f"   Dígito {digit}: {acc:.4f} ({acc*100:.2f}%)")
print()

# 2. Dígitos con más errores
print("2. ¿Dónde se cometieron más errores? ¿En qué clases?")
worst_digits = np.argsort(accuracies)[:3]
for digit in worst_digits:
    acc = accuracies[digit]
    errors = totals[digit] - diagonal[digit]
    print(f"   Dígito {digit}: {errors} errores, precisión {acc:.4f} ({acc*100:.2f}%)")
print()

# Analizar confusiones más comunes
print("   Confusiones más comunes (errores > 5):")
for i in range(10):
    for j in range(10):
        if i != j and cm[i][j] > 5:
            print(f"   {cm[i][j]} veces: {i} clasificado como {j}")
print()

# 3. Métricas importantes
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

print("3. ¿Qué métrica destacarías y por qué?")
print(f"   • F1-Score promedio: {np.mean(f1):.4f}")
print(f"   • Precisión promedio: {np.mean(precision):.4f}")
print(f"   • Recall promedio: {np.mean(recall):.4f}")
print()
print("   DESTACARÍA el F1-Score porque:")
print("   - Combina precisión y recall en una sola métrica")
print("   - Es útil cuando las clases están balanceadas (como en MNIST)")
print("   - Nos da una medida general del rendimiento del modelo")
print()


# %% [markdown]
# ## 🎯 Reflexión Final y Conclusiones

# %% [markdown]
# ## ¿Consideras que este modelo convolucional es adecuado para el problema?
#    Si
#    - Logra una precisión muy alta (>98%)
#    - Las CNN son ideales para reconocimiento de imágenes
#    - Detecta automáticamente jerarquías de características
#    - Es computacionalmente eficiente para este tamaño de problema
# 
# ## ¿Qué modificaciones propondrías para mejorar el rendimiento?
#    - Data Augmentation: rotaciones, traslaciones, ruido
#    - Más capas convolucionales para mayor profundidad
#    - Batch Normalization para estabilizar el entrenamiento
#    - Dropout para reducir overfitting
#    - Learning rate scheduling
#    - Más épocas de entrenamiento
# 
# ## ¿Cómo se compara con una red neuronal multicapa (MLP) tradicional?
#    VENTAJAS de CNN sobre MLP:
#    - Invariancia a traslaciones
#    - Menos parámetros (pesos compartidos)
#    - Detecta patrones locales automáticamente
#    - Preserva información espacial
#    - Mejor generalización para imágenes
# 
#    3. La primera capa detecta bordes, las siguientes formas más complejas
#    4. MaxPooling reduce dimensionalidad conservando información importante
#    5. La visualización ayuda a interpretar y debuggear modelos
#    6. MNIST es un problema bien resuelto por CNN básicas

# %%
# Guardar algunos ejemplos de predicciones incorrectas para análisis
incorrect_indices = np.where(y_pred != y_test)[0]
print(f"Total de predicciones incorrectas: {len(incorrect_indices)}")

# Mostrar algunos ejemplos de errores
if len(incorrect_indices) > 0:
    plt.figure(figsize=(15, 6))
    num_examples = min(8, len(incorrect_indices))
    
    for i in range(num_examples):
        idx = incorrect_indices[i]
        plt.subplot(2, 4, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Real: {y_test[idx]}, Pred: {y_pred[idx]}\nConf: {y_pred_probs[idx][y_pred[idx]]:.3f}')
        plt.axis('off')
    
    plt.suptitle('Ejemplos de Predicciones Incorrectas', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nObservaciones sobre los errores:")
    print("• Algunos dígitos pueden ser ambiguos incluso para humanos")
    print("• La calidad de escritura afecta la clasificación")
    print("• Dígitos muy similares (como 4 y 9, o 3 y 8) se confunden más")
else:
    print("¡Perfecto! No hay errores de clasificación.")


