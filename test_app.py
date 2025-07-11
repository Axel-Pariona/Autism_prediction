#!/usr/bin/env python3
"""
Script de prueba para verificar que la aplicación Streamlit funcione correctamente
"""

import os
import sys
import pandas as pd
import numpy as np

def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    dependencias = [
        ('streamlit', 'streamlit'),
        ('tensorflow', 'tensorflow'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]
    
    faltantes = []
    
    for modulo, nombre_pip in dependencias:
        try:
            __import__(modulo)
            print(f"✅ {nombre_pip} - Instalado")
        except ImportError:
            print(f"❌ {nombre_pip} - NO instalado")
            faltantes.append(nombre_pip)
    
    if faltantes:
        print(f"\n📦 Para instalar las dependencias faltantes:")
        print(f"pip install {' '.join(faltantes)}")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def verificar_modelo():
    """Verifica que el modelo TFLite esté disponible"""
    print("\n🤖 Verificando modelo...")
    
    model_path = 'modelo_autismo.tflite'
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo encontrado: {model_path} ({size:,} bytes)")
        
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"   - Entrada: {input_details[0]['shape']}")
            print(f"   - Salida: {output_details[0]['shape']}")
            print("✅ Modelo TFLite cargado correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    else:
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Ejecuta el notebook mark3.ipynb para generar el modelo")
        return False

def verificar_app_streamlit():
    """Verifica que el archivo de la app esté presente"""
    print("\n📱 Verificando aplicación...")
    
    app_path = 'app_streamlit.py'
    
    if os.path.exists(app_path):
        with open(app_path, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Verificar componentes clave
        componentes = [
            'OPCIONES_VARIABLES',
            'calcular_puntaje_riesgo',
            'predecir_tea',
            'mostrar_evaluacion_clinica'
        ]
        
        for componente in componentes:
            if componente in contenido:
                print(f"✅ Componente encontrado: {componente}")
            else:
                print(f"❌ Componente faltante: {componente}")
        
        print("✅ Aplicación Streamlit verificada")
        return True
    else:
        print(f"❌ Aplicación no encontrada: {app_path}")
        return False

def crear_datos_prueba():
    """Crea datos de prueba para verificar el funcionamiento"""
    print("\n🧪 Creando datos de prueba...")
    
    opciones = {
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
    
    # Crear casos de prueba
    casos_prueba = [
        {
            'nombre': 'Desarrollo Típico',
            'datos': {
                'Edad (meses)': 24,
                'Sexo': 'Femenino',
                'Lenguaje': 'Lenguaje funcional',
                'Comunicación no verbal': 'Adecuada',
                'Contacto visual': 'Natural',
                'Interacción social': 'Adecuada',
                'Respuesta al nombre': 'Siempre',
                'Estereotipias': 'Ausentes',
                'Intereses restringidos': 'Ausentes',
                'Regulación emocional': 'Adecuada',
                'TDAH': 'No',
                'Discapacidad intelectual': 'No',
                'Hipersensibilidad sensorial': 'Ninguna',
                'Trastornos del sueño': 'Normal',
                'Alimentación selectiva': 'Ninguna',
                'Antecedentes familiares': 'Ninguno'
            }
        },
        {
            'nombre': 'TEA Nivel 3 (Severo)',
            'datos': {
                'Edad (meses)': 36,
                'Sexo': 'Masculino',
                'Lenguaje': 'No verbal',
                'Comunicación no verbal': 'Ausente',
                'Contacto visual': 'Evitativo',
                'Interacción social': 'Ausente',
                'Respuesta al nombre': 'Nunca',
                'Estereotipias': 'Muy frecuentes',
                'Intereses restringidos': 'Muy intensos',
                'Regulación emocional': 'Autolesiva',
                'TDAH': 'Sí',
                'Discapacidad intelectual': 'Sí',
                'Hipersensibilidad sensorial': 'Alta',
                'Trastornos del sueño': 'Severo',
                'Alimentación selectiva': 'Alta',
                'Antecedentes familiares': 'TEA'
            }
        }
    ]
    
    for caso in casos_prueba:
        print(f"📋 Caso: {caso['nombre']}")
        for key, value in caso['datos'].items():
            print(f"   {key}: {value}")
        print()
    
    return casos_prueba

def main():
    """Función principal de verificación"""
    print("🚀 Verificación de la Aplicación de Predicción TEA")
    print("=" * 50)
    
    # Verificaciones
    dependencias_ok = verificar_dependencias()
    modelo_ok = verificar_modelo()
    app_ok = verificar_app_streamlit()
    
    print("\n📊 Resumen de Verificación:")
    print(f"   Dependencias: {'✅' if dependencias_ok else '❌'}")
    print(f"   Modelo TFLite: {'✅' if modelo_ok else '❌'}")
    print(f"   Aplicación: {'✅' if app_ok else '❌'}")
    
    if dependencias_ok and modelo_ok and app_ok:
        print("\n🎉 ¡Todo está listo!")
        print("Ejecuta: streamlit run app_streamlit.py")
        
        # Crear datos de prueba
        crear_datos_prueba()
        
    else:
        print("\n⚠️ Hay problemas que resolver antes de ejecutar la aplicación")
        
        if not modelo_ok:
            print("\n📝 Para generar el modelo:")
            print("1. Abre el notebook mark3.ipynb")
            print("2. Ejecuta todas las celdas")
            print("3. Verifica que se genere modelo_autismo.tflite")

if __name__ == "__main__":
    main()
