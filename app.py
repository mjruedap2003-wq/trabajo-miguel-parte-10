from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf

# --- CONFIGURACIÓN DE PÁGINA FUTURISTA ---
st.set_page_config(
    page_title="NEURAL EXAM MONITOR // PROCTOR AI",
    page_icon="🛡️",
    layout="wide",
)

# Estilos CSS Cyberpunk / Futurista
st.markdown(
    """
    <style>
    /* Fondo oscuro y tipografía futurista estilo interfaz neón */
    .stApp {
        background-color: #030712;
        color: #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3, h4, span, label {
        color: #00ffcc !important;
        font-family: 'Courier New', Courier, monospace !important;
        text-shadow: 0 0 10px #00ffcc;
    }
    .stSidebar {
        background-color: #0a0f1d !important;
        border-right: 2px solid #00ffcc;
    }
    .status-card {
        border: 2px solid #00ffcc;
        padding: 20px;
        background-color: rgba(0, 255, 204, 0.05);
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.2);
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .alert-card {
        border: 2px solid #ff0055;
        padding: 20px;
        background-color: rgba(255, 0, 85, 0.1);
        box-shadow: 0 0 25px rgba(255, 0, 85, 0.4);
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_keras_model():
    try:
        # Carga del modelo usando la API oficial de tf.keras para evitar el TypeError
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(
            f"❌ CRITICAL ERROR: No se pudo cargar el archivo 'keras_model.h5':"
            f" {e}"
        )
        return None


# Carga del modelo de Teachable Machine
model = load_keras_model()

# Mapeo de labels de Teachable Machine
LABELS = {
    0: "Con teléfono 📱",
    1: "Con nada 🧘",
    2: "Con audífonos 🎧",
    3: "Con libro 📖",
    4: "Tomando agua / Termo 🥤",
}

# --- ENCABEZADO ---
st.title("🛡️ NEURAL PROCTOR AI // MONITOR DE EXÁMENES")
st.caption(
    "SISTEMA IA DE ESCANEO DE DESCONCENTRACIÓN Y MONITOREO EN TIEMPO REAL"
)

# Imagen decorativa previa (opcional)
try:
    banner = Image.open("OIG5.jpg")
    st.image(banner, width=320, caption="Scanner Biométrico Neuronal v4.0")
except FileNotFoundError:
    pass

st.divider()

with st.sidebar:
    st.title("⚙️ PANEL DE MONITOREO")
    st.subheader("Reglas del Examen")
    st.markdown("""
    - ❌ **Prohibido:** Teléfono
    - ❌ **Prohibido:** Libro / Apuntes
    - ❌ **Prohibido:** Termo / Tomar agua
    - ❌ **Prohibido:** Audífonos
    - ✅ **Permitido:** Estar despejado ("Con nada")
    """)
    st.divider()
    st.caption("Cyberdyne Security Systems © 2026")

# --- ESCANEO VÍA CÁMARA ---
st.subheader("📸 ESCANEO BIOMÉTRICO DEL POSTULANTE")
img_file_buffer = st.camera_input("INICIAR RECONOCIMIENTO FACIAL Y ENTORNO")

if img_file_buffer is not None and model is not None:
    # Procesamiento de la imagen para Teachable Machine (224x224 RGB)
    img = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    img_array = np.asarray(img_resized)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1.0

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción
    with st.spinner("ANALIZANDO ENTIDAD Y OBJETOS CERCANOS..."):
        prediction = model.predict(data)
        index_max = np.argmax(prediction[0])
        confidence = prediction[0][index_max]

    clase_detectada = LABELS.get(index_max, "Desconocido")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 RESULTADO DEL DIAGNÓSTICO")
        st.write(f"**Estado Detectado:** {clase_detectada}")
        st.write(f"**Nivel de Certeza:** {confidence * 100:.2f}%")

        # Barra de progreso para visualización
        st.progress(float(confidence))

    with col2:
        st.subheader("⚡ EVALUACIÓN DE ACCESO AL EXAMEN")

        # Lógica de bloqueo de examen:
        # Si detecta teléfono (0), audífonos (2), libro (3) o agua (4) -> Bloquea
        if index_max in [0, 2, 3, 4]:
            st.markdown(
                f"""
                <div class="alert-card">
                    <h2 style="color: #ff0055; margin:0;">🚫 EXAMEN BLOQUEADO</h2>
                    <h3 style="color: #ffffff !important; margin-top:10px;">DISTRACCIÓN DETECTADA: {clase_detectada.upper()}</h3>
                    <p style="color: #ff80a0; font-size:16px;">
                        <strong>ACCESO DENEGADO:</strong> No puedes realizar la prueba. Debes retirar todos los elementos de distracción (teléfono, libro, termo o audífonos) de tu campo visual antes de continuar.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.error("🚨 ALERTA: Retira las distracciones para desbloquear la prueba.")

        # Si detecta "Con nada" (1) -> Permite realizar el examen
        elif index_max == 1:
            st.markdown(
                """
                <div class="status-card">
                    <h2 style="color: #00ffcc; margin:0;">✅ ACCESO CONCEDIDO</h2>
                    <h3 style="color: #ffffff !important; margin-top:10px;">ENTORNO LIMPIO DE DISTRACCIONES</h3>
                    <p style="color: #80ffe5; font-size:16px;">
                        <strong>ESTADO ÓPTIMO:</strong> Postulante enfocado y sin elementos no autorizados. Puedes iniciar tu examen de inmediato.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.balloons()
            st.button("🚀 INICIAR EXAMEN AHORA", type="primary")

elif model is None:
    st.error("No se pudo cargar el modelo. Revisa la consola y tus requerimientos.")
