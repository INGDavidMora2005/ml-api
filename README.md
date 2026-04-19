# 🤖 Agente de Telegram con n8n — Clasificación de Imágenes por Colores

> **Taller Práctico — Inteligencia Artificial | UCEVA**  
> Clasificación de imágenes con Regresión Logística sobre histogramas de color RGB

---

## 📋 Descripción

Este proyecto implementa un agente de inteligencia artificial integrado con **Telegram**, orquestado mediante **n8n**, capaz de:

- **Entrenar** modelos de regresión logística para clasificación de imágenes basada en histogramas de color RGB.
- **Clasificar** imágenes enviadas por el usuario y responder con la clase predicha y la confianza del modelo.

El servicio de Machine Learning se expone como **API REST con FastAPI**, se conteneriza con **Docker** y se hace accesible públicamente mediante un túnel con **ngrok**.

---

## 🧰 Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Interfaz de usuario | Telegram Bot API |
| Orquestación | n8n + AI Agent node |
| API REST | FastAPI + Uvicorn |
| Machine Learning | scikit-learn (Regresión Logística) |
| Extracción de features | OpenCV (histograma RGB) |
| Persistencia de modelos | joblib / pickle |
| Contenerización | Docker + Docker Compose |
| Túnel público | ngrok |

---

## 🏗️ Arquitectura del Sistema

```
Usuario Telegram
      │
      │  .zip / imagen
      ▼
Telegram Bot API (Webhook HTTPS)
      │
      ▼
n8n (Telegram Trigger → AI Agent → HTTP Tools)
      │
      │  POST /train  /  POST /classify
      ▼
ngrok (HTTPS Tunnel)
      │
      ▼
FastAPI + Uvicorn  [Docker Container]
      │
      ├── /train   → Histograma RGB → LogisticRegression → .pkl
      └── /classify → Histograma RGB → modelo.pkl → predicción
```

---

## 📁 Estructura del Proyecto

```
ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI (/health, /train, /classify, /models)
│   ├── model_service.py     # Lógica de entrenamiento y clasificación
│   └── feature_extractor.py # Extracción de histograma de color RGB
├── models/                  # Modelos .pkl entrenados (persistencia)
├── Dockerfile
├── docker-compose.yml       # Servicios: ml-api + n8n
└── requirements.txt
```

---

## ⚙️ Requisitos Previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [ngrok](https://ngrok.com/download) (cuenta gratuita)
- Cuenta en [Telegram](https://telegram.org/) y bot creado con [@BotFather](https://t.me/BotFather)
- Cuenta en [n8n Cloud](https://n8n.io) o n8n local vía Docker

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd ml-api
```

### 2. Levantar los servicios con Docker

```bash
docker compose up --build -d
```

Esto levanta dos contenedores:
- **ml-api** en `http://localhost:8000`
- **n8n** en `http://localhost:5678`

### 3. Verificar que la API está activa

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

O abre en el navegador: `http://localhost:8000/docs` para ver la documentación interactiva (Swagger UI).

### 4. Crear el túnel con ngrok

```bash
ngrok config add-authtoken TU_TOKEN
ngrok http 8000
```

Copia la URL pública generada (ej: `https://abc123.ngrok-free.app`).

---

## 📡 Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Verifica que la API esté activa |
| GET | `/models` | Lista los modelos entrenados disponibles |
| POST | `/train` | Entrena un nuevo clasificador con un `.zip` de imágenes |
| POST | `/classify` | Clasifica una imagen con un modelo entrenado |

### POST `/train`

**Body (form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | File | Archivo `.zip` con imágenes organizadas por carpetas/clases |
| `classifier_name` | Text | Nombre del clasificador a entrenar |

**Respuesta:**
```json
{
  "classifier_name": "colores",
  "accuracy": 0.9583,
  "classes": ["amarillo", "azul", "rojo", "verde"],
  "total_images": 120
}
```

### POST `/classify`

**Body (form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | File | Imagen a clasificar (`.jpg`, `.png`, `.bmp`) |
| `classifier_name` | Text | Nombre del clasificador a usar |

**Respuesta:**
```json
{
  "prediction": "azul",
  "confidence": 0.9841,
  "classifier_name": "colores"
}
```

---

## 📦 Formato del Dataset (.zip)

El archivo `.zip` debe contener subcarpetas donde **cada carpeta representa una clase**:

```
dataset.zip
└── dataset/
    ├── rojo/
    │   ├── img001.jpg
    │   └── img002.jpg
    ├── verde/
    │   └── ...
    └── azul/
        └── ...
```

> Se recomienda un mínimo de **20 imágenes por clase** para obtener resultados aceptables.  
> Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## 🧠 Modelo de Machine Learning

### Regresión Logística

Se utiliza `LogisticRegression` de scikit-learn en su modo multiclase. El modelo no trabaja con píxeles en crudo sino con un **vector de características compacto** basado en histogramas de color.

### Extracción de Características — Histograma RGB

1. La imagen se redimensiona a **128×128 píxeles**
2. Se calcula el histograma de cada canal de color (B, G, R) con **32 bins** por canal
3. Cada histograma se normaliza y se concatenan → vector de **96 dimensiones**

```python
# Ejemplo de extracción
features = extract_color_histogram(image_bytes)
# shape: (96,) → 32 bins × 3 canales
```

**¿Por qué histogramas?** Capturan la distribución de colores de la imagen sin importar la posición de los objetos, siendo una representación eficiente para distinguir imágenes con paletas de color diferentes (ej: cielo azul vs bosque verde).

### Persistencia

Los modelos entrenados se serializan con `joblib` y se almacenan en `/models/` dentro del contenedor, montado como volumen persistente para sobrevivir reinicios.

---

## 🤖 Configuración del Agente en n8n

El workflow de n8n conecta los siguientes nodos:

```
Telegram Trigger → AI Agent (LLM) → HTTP Tool: /train
                                  → HTTP Tool: /classify
                ↓
         Telegram (Send Message)
```

### System Prompt del AI Agent

```
Eres un asistente de clasificación de imágenes con dos herramientas:

1. ENTRENAR MODELO: úsala cuando el usuario envíe un .zip con imágenes 
   y un nombre de clasificador. Responde con el score de accuracy obtenido.

2. CLASIFICAR IMAGEN: úsala cuando el usuario envíe una imagen y mencione 
   un clasificador. Responde con la clase predicha y la confianza.
```

---

## 🧪 Dataset de Prueba

Se incluye un script para generar un dataset sintético de colores sólidos:

```bash
python generar_dataset.py
# Genera dataset_colores.zip con 4 clases × 30 imágenes
# Clases: rojo, verde, azul, amarillo
```

---

## 📊 Criterios de Evaluación

| Criterio | Estado |
|---|---|
| API FastAPI funcional (train + classify) | ✅ |
| Workflow de n8n con AI Agent y tools | ✅ |
| Bot de Telegram operativo end-to-end | ✅ |
| Contenerización con Docker + docker-compose | ✅ |
| Persistencia de modelos con joblib | ✅ |
| Documentación | ✅ |

---

## 📚 Referencias

- [Documentación n8n](https://docs.n8n.io)
- [n8n AI Agent node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/)
- [FastAPI](https://fastapi.tiangolo.com)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [OpenCV Histogramas](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
- [ngrok Docs](https://ngrok.com/docs)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 👤 Integrantes

**David Mora**  
**Juan Diego Rodriguez**
**Juan Pablo Devia Masso**
Ingeniería de Sistemas — UCEVA  
Unidad Central del Valle del Cauca
