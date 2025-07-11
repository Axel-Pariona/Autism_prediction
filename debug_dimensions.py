#!/usr/bin/env python3
"""
Script para debuggear las dimensiones del modelo y preprocessing
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def debug_modelo_dimensiones():
    """Debuggea las dimensiones del modelo y preprocessing"""
    print("🔍 Debugging dimensiones del modelo...")
    
    # Cargar modelo
    interpreter = tf.lite.Interpreter(model_path='modelo_autismo.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Dimensiones del modelo:")
    print(f"   Entrada esperada: {input_details[0]['shape']}")
    print(f"   Salida: {output_details[0]['shape']}")
    
    expected_features = input_details[0]['shape'][1]
    print(f"   Características esperadas: {expected_features}")
    
    # Recrear el preprocessing exacto del entrenamiento
    print("\n🔧 Recreando preprocessing del entrenamiento...")
    
    # Variables exactas del entrenamiento
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
    
    # Crear dataset de entrenamiento simulado (igual que en el notebook)
    print("📋 Creando dataset simulado para entrenamiento del preprocessor...")
    
    data = []
    for i in range(1000):  # Dataset pequeño para prueba
        edad = np.random.randint(6, 36)
        
        fila = {
            'ID': i,
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
            'Antecedentes familiares': np.random.choice(OPCIONES_VARIABLES['Antecedentes familiares']),                'Puntaje riesgo': np.random.randint(0, 25),
            'Diagnóstico orientativo': np.random.choice(['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado'])
        }
        data.append(fila)
    
    df = pd.DataFrame(data)
    print(f"Dataset creado: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Separar X e y (exactamente como en el notebook)
    X = df.drop(columns=["ID", "Diagnóstico orientativo"])
    y = df["Diagnóstico orientativo"]
    
    print(f"\nX shape: {X.shape}")
    print(f"Columnas de X: {list(X.columns)}")
    
    # Columnas categóricas y numéricas (exactamente como en el notebook)
    categorical_cols = [
        'Sexo', 'Lenguaje', 'Comunicación no verbal', 'Contacto visual',
        'Interacción social', 'Respuesta al nombre', 'Estereotipias',
        'Intereses restringidos', 'Regulación emocional', 'TDAH',
        'Discapacidad intelectual', 'Hipersensibilidad sensorial',
        'Trastornos del sueño', 'Alimentación selectiva', 'Antecedentes familiares'
    ]
    numeric_cols = ['Edad (meses)', 'Puntaje riesgo']
    
    print(f"\nColumnas categóricas ({len(categorical_cols)}): {categorical_cols}")
    print(f"Columnas numéricas ({len(numeric_cols)}): {numeric_cols}")
    
    # Crear preprocessor (exactamente como en el notebook)
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])
    
    # Entrenar el preprocessor
    print("\n🔧 Entrenando preprocessor...")
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Shape después del preprocessing: {X_processed.shape}")
    print(f"Características generadas: {X_processed.shape[1]}")
    
    # Comparar con lo que espera el modelo
    print(f"\n📊 Comparación:")
    print(f"   Modelo espera: {expected_features} características")
    print(f"   Preprocessor genera: {X_processed.shape[1]} características")
    print(f"   Diferencia: {expected_features - X_processed.shape[1]}")
    
    if X_processed.shape[1] == expected_features:
        print("✅ ¡Las dimensiones coinciden!")
    else:
        print("❌ Las dimensiones NO coinciden")
        
        # Analizar la diferencia
        print(f"\n🔍 Análisis detallado:")
        
        # Contar características por categoría
        cat_encoder = preprocessor.named_transformers_['cat']
        
        total_cat_features = 0
        for i, col in enumerate(categorical_cols):
            unique_values = len(OPCIONES_VARIABLES[col])
            encoded_features = len(cat_encoder.categories_[i])
            total_cat_features += encoded_features
            print(f"   {col}: {unique_values} valores únicos → {encoded_features} características")
        
        print(f"   Total categóricas: {total_cat_features}")
        print(f"   Total numéricas: {len(numeric_cols)}")
        print(f"   Total calculado: {total_cat_features + len(numeric_cols)}")
    
    # Crear ejemplo de predicción
    print(f"\n🧪 Probando predicción con ejemplo:")
    
    ejemplo = pd.DataFrame([{
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
        'Antecedentes familiares': 'TEA',
        'Puntaje riesgo': 10
    }])
    
    ejemplo_processed = preprocessor.transform(ejemplo)
    print(f"Ejemplo procesado shape: {ejemplo_processed.shape}")
    
    if ejemplo_processed.shape[1] == expected_features:
        print("✅ El ejemplo tiene las dimensiones correctas")
        
        # Probar predicción
        try:
            input_data = ejemplo_processed.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            pred_idx = int(np.argmax(output_data))
            confianza = float(np.max(output_data))
            
            etiquetas = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
            resultado = etiquetas[pred_idx]
            
            print(f"✅ Predicción exitosa: {resultado} (confianza: {confianza:.3f})")
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
    else:
        print("❌ El ejemplo NO tiene las dimensiones correctas")

if __name__ == "__main__":
    debug_modelo_dimensiones()
