# Predictor de TEA - Aplicación Mejorada

## 🎯 Descripción
Aplicación para predicción de Trastorno del Espectro Autista basada en un modelo de Red Neuronal con **70% de precisión**. Utiliza 17 variables clínicas y conductuales para generar diagnósticos orientativos.

## 🚀 Versión Mejorada - Streamlit

### **Archivo Principal:** `app_streamlit.py`

**Nuevas Características:**
- ✅ **17 variables clínicas reales** (basadas en el modelo entrenado)
- ✅ **Interfaz intuitiva** con formularios clínicos
- ✅ **Cálculo automático del puntaje de riesgo**
- ✅ **3 modos de uso:** Evaluación clínica, datos simulados, información del modelo
- ✅ **Visualizaciones avanzadas** con gráficos de probabilidades
- ✅ **Preprocesamiento automático** (OneHotEncoder + StandardScaler)
- ✅ **Advertencias médicas** apropiadas

### **Variables Clínicas Incluidas:**
1. **Básicas:** Edad, Sexo
2. **Comunicación:** Lenguaje, Comunicación no verbal, Contacto visual, Respuesta al nombre
3. **Social:** Interacción social
4. **Comportamental:** Estereotipias, Intereses restringidos, Regulación emocional
5. **Comorbilidades:** TDAH, Discapacidad intelectual
6. **Sensorial:** Hipersensibilidad sensorial, Trastornos del sueño, Alimentación selectiva
7. **Antecedentes:** Historia familiar

### **Instalación y Uso:**

```bash
# Instalar dependencias
pip install streamlit tensorflow pandas numpy scikit-learn

# Ejecutar aplicación
streamlit run app_streamlit.py
```

### **Diagnósticos Disponibles:**
- 🟢 **Desarrollo típico**
- 🟡 **TEA - Nivel 1** (Leve)
- 🟠 **TEA - Nivel 2** (Moderado)
- 🔴 **TEA - Nivel 3** (Severo)
- ⚪ **Indeterminado**

---

## 📱 Otras Opciones Disponibles

### 1. **Tkinter (Sin instalaciones extra)**
**Archivo:** `app_tkinter.py`
```bash
python app_tkinter.py
```

### 2. **PyQt5 (Más profesional)**
**Archivo:** `app_pyqt.py`
```bash
pip install PyQt5
python app_pyqt.py
```

---

## 📋 Requisitos del Sistema

- **Python 3.8+**
- **Archivo:** `modelo_autismo.tflite` (generado del notebook)
- **RAM:** Mínimo 2GB
- **Sistema:** Windows, macOS, Linux

---

## � Solución de Problemas

### Error: "No se encontró modelo_autismo.tflite"
1. Ejecuta el notebook `mark3.ipynb` completamente
2. Asegúrate de que se genere el archivo `modelo_autismo.tflite`
3. Coloca el archivo en la misma carpeta que `app_streamlit.py`

### Error de dependencias
```bash
# Instalar versión específica de TensorFlow
pip install tensorflow==2.13.0

# O usar versión ligera
pip install tflite-runtime
```

### Problemas de memoria
- Usar `tflite-runtime` en lugar de `tensorflow` completo
- Cerrar otras aplicaciones durante la ejecución

---

## � Información del Modelo

- **Arquitectura:** Red Neuronal Densa (32→16→5 neuronas)
- **Precisión:** ~70% en conjunto de prueba
- **Dataset:** 500,000 casos sintéticos basados en criterios clínicos
- **Validación:** División 80/20
- **Optimizador:** Adam
- **Función de pérdida:** Sparse Categorical Crossentropy

---

## ⚠️ Advertencia Médica

Este sistema es una **herramienta de apoyo al diagnóstico** y NO sustituye la evaluación clínica profesional. Los resultados deben ser siempre validados por especialistas en neurología o psiquiatría infantil.

---

## 💡 Recomendaciones de Uso

1. **Para desarrollo rápido:** Streamlit (recomendado)
2. **Para aplicación de escritorio:** Tkinter o PyQt5  
3. **Para móvil:** Considera Flutter o React Native
4. **Para web:** Deploy de Streamlit en Heroku/Streamlit Cloud

¡La versión de Streamlit ahora refleja exactamente las variables y el preprocesamiento usado en el entrenamiento del modelo!
