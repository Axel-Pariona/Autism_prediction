import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de TEA",
    page_icon="🧠",
    layout="wide"
)

# Título y descripción
st.title("🧠 Predictor de Trastorno del Espectro Autista")
st.markdown("### Aplicación de diagnóstico clínico usando IA")
st.markdown("**Precisión del modelo: ~70%** | Basado en variables clínicas y conductuales")

# Definir las opciones para cada variable categórica (basadas en el dataset de entrenamiento)
OPCIONES_VARIABLES = {
    'Sexo': ['Masculino', 'Femenino'],
    'Lenguaje': ['No verbal', 'Ecolalia', 'Frases simples', 'Lenguaje funcional'],
    'Comunicación no verbal': ['Ausente', 'Muy limitada', 'Limitada', 'Adecuada'],
    'Contacto visual': ['Evitativo', 'Intermitente', 'Sostenido', 'Natural'],
    'Interacción social': ['Ausente', 'Pasiva', 'Inapropiada', 'Adecuada'],
    'Respuesta al nombre': ['Nunca', 'A veces', 'Siempre'],
    'Estereotipias': ['Muy frecuentes', 'Frecuentes', 'Ocasionales', 'Ausentes'],
    'Intereses restringidos': ['Muy intensos', 'Persistentes', 'Leves', 'Ausentes'],
    'Regulación emocional': ['Autolesiva', 'Crisis frecuentes', 'Ocasionales', 'Adecuada'],
    'TDAH': ['Sí', 'No'],
    'Discapacidad intelectual': ['Sí', 'No'],
    'Hipersensibilidad sensorial': ['Alta', 'Moderada', 'Leve', 'Ninguna'],
    'Trastornos del sueño': ['Severo', 'Moderado', 'Leve', 'Normal'],
    'Alimentación selectiva': ['Alta', 'Moderada', 'Leve', 'Ninguna'],
    'Antecedentes familiares': ['TEA', 'TDAH', 'Discapacidad intelectual', 'Ninguno']
}

# Cargar modelo
# Función para crear el preprocessor (debe ser igual al del entrenamiento)
@st.cache_resource
def create_preprocessor():
    # Columnas categóricas y numéricas (igual que en el entrenamiento)
    categorical_cols = [
        'Sexo', 'Lenguaje', 'Comunicación no verbal', 'Contacto visual',
        'Interacción social', 'Respuesta al nombre', 'Estereotipias',
        'Intereses restringidos', 'Regulación emocional', 'TDAH',
        'Discapacidad intelectual', 'Hipersensibilidad sensorial',
        'Trastornos del sueño', 'Alimentación selectiva', 'Antecedentes familiares'
    ]
    numeric_cols = ['Edad (meses)', 'Puntaje riesgo']
    
    # Crear dataset de entrenamiento que incluya TODAS las categorías posibles
    print("🔧 Creando dataset de entrenamiento para preprocessor...")
    
    data = []
    # Crear al menos una fila por cada combinación de categorías para asegurar que 
    # el OneHotEncoder vea todas las opciones posibles
    for i in range(500):  # Dataset suficiente para entrenar el preprocessor
        edad = np.random.randint(3, 36)
        
        fila = {
            'Edad (meses)': edad,
            'Sexo': np.random.choice(OPCIONES_VARIABLES['Sexo']),
            'Lenguaje': np.random.choice(OPCIONES_VARIABLES['Lenguaje']),
            'Comunicación no verbal': np.random.choice(OPCIONES_VARIABLES['Comunicación no verbal']),
            'Contacto visual': np.random.choice(OPCIONES_VARIABLES['Contacto visual']),
            'Interacción social': np.random.choice(OPCIONES_VARIABLES['Interacción social']),
            'Respuesta al nombre': np.random.choice(OPCIONES_VARIABLES['Respuesta al nombre']),
            'Estereotipias': np.random.choice(OPCIONES_VARIABLES['Estereotipias']),
            'Intereses restringidos': np.random.choice(OPCIONES_VARIABLES['Intereses restringidos']),
            'Regulación emocional': np.random.choice(OPCIONES_VARIABLES['Regulación emocional']),
            'TDAH': np.random.choice(OPCIONES_VARIABLES['TDAH']),
            'Discapacidad intelectual': np.random.choice(OPCIONES_VARIABLES['Discapacidad intelectual']),
            'Hipersensibilidad sensorial': np.random.choice(OPCIONES_VARIABLES['Hipersensibilidad sensorial']),
            'Trastornos del sueño': np.random.choice(OPCIONES_VARIABLES['Trastornos del sueño']),
            'Alimentación selectiva': np.random.choice(OPCIONES_VARIABLES['Alimentación selectiva']),
            'Antecedentes familiares': np.random.choice(OPCIONES_VARIABLES['Antecedentes familiares']),
            'Puntaje riesgo': np.random.randint(0, 25)
        }
        data.append(fila)
    
    # Asegurar que TODAS las categorías estén representadas
    for var, opciones in OPCIONES_VARIABLES.items():
        for opcion in opciones:
            fila_completa = {
                'Edad (meses)': 24,
                'Puntaje riesgo': 10
            }
            # Llenar con valores por defecto
            for v, ops in OPCIONES_VARIABLES.items():
                fila_completa[v] = ops[0]  # Primer valor por defecto
            
            # Establecer la categoría específica
            fila_completa[var] = opcion
            data.append(fila_completa)
    
    df_entrenamiento = pd.DataFrame(data)
    
    # Crear el preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])
    
    # Entrenar el preprocessor
    preprocessor.fit(df_entrenamiento)
    
    print(f"✅ Preprocessor entrenado con {len(df_entrenamiento)} muestras")
    
    # Verificar dimensiones
    test_sample = df_entrenamiento.iloc[0:1]
    processed_shape = preprocessor.transform(test_sample).shape
    print(f"✅ Dimensiones generadas: {processed_shape[1]}")
    
    return preprocessor, categorical_cols, numeric_cols

def calcular_puntaje_riesgo(datos):
    """Calcula el puntaje de riesgo basado en las respuestas clínicas"""
    score = 0
    
    # Lenguaje
    if datos['Lenguaje'] == 'No verbal': score += 3
    elif datos['Lenguaje'] == 'Ecolalia': score += 2
    elif datos['Lenguaje'] == 'Frases simples': score += 1
    
    # Comunicación no verbal
    if datos['Comunicación no verbal'] == 'Ausente': score += 3
    elif datos['Comunicación no verbal'] == 'Muy limitada': score += 2
    elif datos['Comunicación no verbal'] == 'Limitada': score += 1
    
    # Contacto visual
    if datos['Contacto visual'] == 'Evitativo': score += 2
    elif datos['Contacto visual'] == 'Intermitente': score += 1
    
    # Interacción social
    if datos['Interacción social'] == 'Ausente': score += 3
    elif datos['Interacción social'] == 'Pasiva': score += 2
    elif datos['Interacción social'] == 'Inapropiada': score += 1
    
    # Respuesta al nombre
    if datos['Respuesta al nombre'] == 'Nunca': score += 2
    elif datos['Respuesta al nombre'] == 'A veces': score += 1
    
    # Estereotipias
    if datos['Estereotipias'] == 'Muy frecuentes': score += 3
    elif datos['Estereotipias'] == 'Frecuentes': score += 2
    elif datos['Estereotipias'] == 'Ocasionales': score += 1
    
    # Intereses restringidos
    if datos['Intereses restringidos'] == 'Muy intensos': score += 2
    elif datos['Intereses restringidos'] == 'Persistentes': score += 1
    
    # Regulación emocional
    if datos['Regulación emocional'] == 'Autolesiva': score += 3
    elif datos['Regulación emocional'] == 'Crisis frecuentes': score += 2
    elif datos['Regulación emocional'] == 'Ocasionales': score += 1
    
    # Comorbilidades
    if datos['Discapacidad intelectual'] == 'Sí': score += 2
    if datos['TDAH'] == 'Sí': score += 1
    
    return score
@st.cache_resource
def load_model():
    model_path = 'modelo_autismo.tflite'
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        st.error("❌ No se encontró el archivo modelo_autismo.tflite")
        st.info("Asegúrate de que el archivo esté en la misma carpeta que esta aplicación")
        return None

# Función de predicción
def predecir_tea(interpreter, preprocessor, datos_usuario):
    """
    Realiza predicción usando el modelo TFLite y los datos preprocesados
    """
    try:
        # Crear DataFrame con los datos del usuario
        df_usuario = pd.DataFrame([datos_usuario])
        
        # Preprocesar los datos (aplicar OneHotEncoder y StandardScaler)
        X_processed = preprocessor.transform(df_usuario)
        
        # Convertir a float32 para TFLite
        input_data = X_processed.astype(np.float32)
        
        # Realizar predicción
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        pred_idx = int(np.argmax(output_data))
        confianza = float(np.max(output_data))
        
        # Etiquetas (deben coincidir con las del entrenamiento)
        ETIQUETAS = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
        resultado = ETIQUETAS[pred_idx] if pred_idx < len(ETIQUETAS) else "Resultado desconocido"
        
        return resultado, confianza, output_data[0]
        
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return "Error", 0.0, []

# Interfaz principal
def main():
    # Cargar modelo y preprocessor
    interpreter = load_model()
    if interpreter is None:
        st.stop()
    
    preprocessor, categorical_cols, numeric_cols = create_preprocessor()
    
    # Sidebar para información
    st.sidebar.header("ℹ️ Información del Modelo")
    st.sidebar.markdown("""
    **Modelo:** Red Neuronal  
    **Precisión:** ~70%  
    **Variables:** 17 características clínicas  
    **Diagnósticos:** 5 categorías
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Modo de Predicción")
    
    modo = st.sidebar.radio(
        "Selecciona el modo:",
        ["📋 Evaluación Clínica Completa", "🔮 Datos Simulados", "📊 Información del Modelo"]
    )
    
    if modo == "📋 Evaluación Clínica Completa":
        mostrar_evaluacion_clinica(interpreter, preprocessor)
    elif modo == "🔮 Datos Simulados":
        mostrar_datos_simulados(interpreter, preprocessor)
    else:
        mostrar_informacion_modelo()

def mostrar_evaluacion_clinica(interpreter, preprocessor):
    st.header("📋 Evaluación Clínica Completa")
    st.markdown("**Completa todos los campos para obtener un diagnóstico orientativo**")
    
    # Crear dos columnas principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👶 Información Básica")
        edad = st.slider("Edad (en meses)", 3, 36, 36, help="Edad del paciente en meses")
        sexo = st.selectbox("Sexo", OPCIONES_VARIABLES['Sexo'])
        
        st.subheader("🗣️ Comunicación y Lenguaje")
        lenguaje = st.selectbox("Nivel de Lenguaje", OPCIONES_VARIABLES['Lenguaje'])
        comunicacion_nv = st.selectbox("Comunicación No Verbal", OPCIONES_VARIABLES['Comunicación no verbal'])
        contacto_visual = st.selectbox("Contacto Visual", OPCIONES_VARIABLES['Contacto visual'])
        respuesta_nombre = st.selectbox("Respuesta al Nombre", OPCIONES_VARIABLES['Respuesta al nombre'])
        
        st.subheader("🤝 Interacción Social")
        interaccion_social = st.selectbox("Interacción Social", OPCIONES_VARIABLES['Interacción social'])
        
        st.subheader("🔄 Comportamientos Repetitivos")
        estereotipias = st.selectbox("Estereotipias", OPCIONES_VARIABLES['Estereotipias'])
        intereses_restringidos = st.selectbox("Intereses Restringidos", OPCIONES_VARIABLES['Intereses restringidos'])
    
    with col2:
        st.subheader("😌 Regulación Emocional")
        regulacion = st.selectbox("Regulación Emocional", OPCIONES_VARIABLES['Regulación emocional'])
        
        st.subheader("🏥 Comorbilidades")
        tdah = st.selectbox("TDAH", OPCIONES_VARIABLES['TDAH'])
        discapacidad_int = st.selectbox("Discapacidad Intelectual", OPCIONES_VARIABLES['Discapacidad intelectual'])
        
        st.subheader("👂 Aspectos Sensoriales")
        hipersensibilidad = st.selectbox("Hipersensibilidad Sensorial", OPCIONES_VARIABLES['Hipersensibilidad sensorial'])
        
        st.subheader("💤 Hábitos")
        sueno = st.selectbox("Trastornos del Sueño", OPCIONES_VARIABLES['Trastornos del sueño'])
        alimentacion = st.selectbox("Alimentación Selectiva", OPCIONES_VARIABLES['Alimentación selectiva'])
        
        st.subheader("👨‍👩‍👧‍👦 Antecedentes")
        antecedentes = st.selectbox("Antecedentes Familiares", OPCIONES_VARIABLES['Antecedentes familiares'])
    
    # Botón para predecir
    st.markdown("---")
    
    if st.button("🧠 Realizar Diagnóstico", type="primary", use_container_width=True):
        # Preparar datos del usuario
        datos_usuario = {
            'Edad (meses)': edad,
            'Sexo': sexo,
            'Lenguaje': lenguaje,
            'Comunicación no verbal': comunicacion_nv,
            'Contacto visual': contacto_visual,
            'Interacción social': interaccion_social,
            'Respuesta al nombre': respuesta_nombre,
            'Estereotipias': estereotipias,
            'Intereses restringidos': intereses_restringidos,
            'Regulación emocional': regulacion,
            'TDAH': tdah,
            'Discapacidad intelectual': discapacidad_int,
            'Hipersensibilidad sensorial': hipersensibilidad,
            'Trastornos del sueño': sueno,
            'Alimentación selectiva': alimentacion,
            'Antecedentes familiares': antecedentes
        }
        
        # Calcular puntaje de riesgo
        puntaje_riesgo = calcular_puntaje_riesgo(datos_usuario)
        datos_usuario['Puntaje riesgo'] = puntaje_riesgo
        
        # Realizar predicción
        try:
            with st.spinner("🔄 Procesando diagnóstico..."):
                resultado, confianza, probabilidades = predecir_tea(interpreter, preprocessor, datos_usuario)
            
            # Mostrar resultados
            mostrar_resultados(resultado, confianza, probabilidades, puntaje_riesgo)
            
        except Exception as e:
            st.error(f"❌ Error en el procesamiento: {str(e)}")
            st.error("Por favor, revisa que todos los campos estén completos.")

def mostrar_resultados(resultado, confianza, probabilidades, puntaje_riesgo):
    st.markdown("---")
    st.header("📊 Resultados del Diagnóstico")
    
    # Columnas para resultados principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Color según resultado
        if resultado == 'Desarrollo típico':
            st.success(f"**Diagnóstico:** {resultado}")
        elif 'TEA' in resultado:
            st.warning(f"**Diagnóstico:** {resultado}")
        else:
            st.info(f"**Diagnóstico:** {resultado}")
    
    with col2:
        st.metric("Confianza del Modelo", f"{confianza*100:.1f}%")
    
    with col3:
        # Interpretar puntaje de riesgo (actualizado para escala /24)
        if puntaje_riesgo >= 18:  # 75% del máximo
            riesgo_nivel = "Muy Alto"
            riesgo_color = "🔴"
        elif puntaje_riesgo >= 15:  # ~62% del máximo
            riesgo_nivel = "Alto"
            riesgo_color = "�"
        elif puntaje_riesgo >= 10:  # ~42% del máximo
            riesgo_nivel = "Moderado"
            riesgo_color = "🟡"
        elif puntaje_riesgo >= 6:   # 25% del máximo
            riesgo_nivel = "Leve"
            riesgo_color = "�"
        else:
            riesgo_nivel = "Muy Bajo"
            riesgo_color = "🟢"
        
        st.metric("Puntaje de Riesgo", f"{puntaje_riesgo}/24", f"{riesgo_color} {riesgo_nivel}")
    
    # Gráfico de probabilidades
    if len(probabilidades) > 0:
        st.subheader("📈 Distribución de Probabilidades")
        
        ETIQUETAS = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
        chart_data = {}
        
        for i, etiqueta in enumerate(ETIQUETAS[:len(probabilidades)]):
            chart_data[etiqueta] = float(probabilidades[i])
        
        st.bar_chart(chart_data)
        
        # Tabla con probabilidades
        df_prob = pd.DataFrame(list(chart_data.items()), columns=['Diagnóstico', 'Probabilidad'])
        df_prob['Probabilidad'] = df_prob['Probabilidad'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(df_prob, use_container_width=True)
    
    # Advertencia médica
    st.warning("""
    ⚠️ **IMPORTANTE:** Este es un sistema de apoyo al diagnóstico basado en IA. 
    Los resultados NO sustituyen la evaluación clínica profesional. 
    Consulte siempre con un especialista en neurología o psiquiatría infantil.
    """)

def mostrar_datos_simulados(interpreter, preprocessor):
    st.header("🔮 Predicción con Datos Simulados")
    st.info("Esta opción utiliza valores predeterminados para probar el modelo")
    
    if st.button("🎲 Generar Predicción Simulada", type="primary"):
        # Datos simulados representativos
        datos_simulados = {
            'Edad (meses)': 30,
            'Sexo': 'Masculino',
            'Lenguaje': 'Frases simples',
            'Comunicación no verbal': 'Limitada',
            'Contacto visual': 'Intermitente',
            'Interacción social': 'Pasiva',
            'Respuesta al nombre': 'A veces',
            'Estereotipias': 'Frecuentes',
            'Intereses restringidos': 'Persistentes',
            'Regulación emocional': 'Ocasionales',
            'TDAH': 'No',
            'Discapacidad intelectual': 'No',
            'Hipersensibilidad sensorial': 'Moderada',
            'Trastornos del sueño': 'Leve',
            'Alimentación selectiva': 'Moderada',
            'Antecedentes familiares': 'TEA'
        }
        
        puntaje_riesgo = calcular_puntaje_riesgo(datos_simulados)
        datos_simulados['Puntaje riesgo'] = puntaje_riesgo
        
        # Mostrar datos simulados
        st.subheader("📋 Datos Utilizados")
        col1, col2 = st.columns(2)
        
        items = list(datos_simulados.items())
        mid = len(items) // 2
        
        with col1:
            for key, value in items[:mid]:
                st.write(f"**{key}:** {value}")
        
        with col2:
            for key, value in items[mid:]:
                st.write(f"**{key}:** {value}")
        
        # Entrenar preprocessor y predecir
        try:
            with st.spinner("🔄 Procesando..."):
                resultado, confianza, probabilidades = predecir_tea(interpreter, preprocessor, datos_simulados)
            
            mostrar_resultados(resultado, confianza, probabilidades, puntaje_riesgo)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Hay un problema con el modelo o los datos.")

def mostrar_informacion_modelo():
    st.header("📊 Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Características del Modelo")
        st.markdown("""
        - **Tipo:** Red Neuronal Profunda
        - **Precisión:** ~70%
        - **Variables de entrada:** 17 características
        - **Categorías de diagnóstico:** 5
        - **Tamaño del dataset:** 500,000 casos simulados
        - **Validación:** División 80/20
        """)
        
        st.subheader("📈 Rendimiento")
        st.markdown("""
        - **Desarrollo típico:** Buena precisión
        - **TEA Nivel 1:** Detección moderada
        - **TEA Nivel 2:** Detección buena
        - **TEA Nivel 3:** Detección excelente
        - **Indeterminado:** Casos ambiguos
        """)
    
    with col2:
        st.subheader("📋 Variables Utilizadas")
        st.markdown("""
        **Información Básica:**
        - Edad (meses)
        - Sexo
        
        **Comunicación:**
        - Nivel de lenguaje
        - Comunicación no verbal
        - Contacto visual
        - Respuesta al nombre
        
        **Social:**
        - Interacción social
        
        **Comportamental:**
        - Estereotipias
        - Intereses restringidos
        - Regulación emocional
        
        **Comorbilidades:**
        - TDAH
        - Discapacidad intelectual
        
        **Sensorial/Hábitos:**
        - Hipersensibilidad sensorial
        - Trastornos del sueño
        - Alimentación selectiva
        
        **Antecedentes:**
        - Historia familiar
        """)
    
    st.info("""
    💡 **Nota:** Este modelo fue entrenado con datos sintéticos basados en criterios clínicos establecidos. 
    Está diseñado como herramienta de apoyo y NO debe usarse como único método diagnóstico.
    """)

if __name__ == "__main__":
    main()
