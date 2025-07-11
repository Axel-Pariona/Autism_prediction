import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tflite_runtime.interpreter as tflite
import os

class TEAPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de TEA")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Cargar modelo
        self.load_model()
        
        # Crear interfaz
        self.create_widgets()
    
    def load_model(self):
        model_path = 'modelo_autismo.tflite'
        if os.path.exists(model_path):
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            messagebox.showerror("Error", "No se encontró el archivo modelo_autismo.tflite")
            self.interpreter = None
    
    def create_widgets(self):
        # Título
        title_label = tk.Label(
            self.root, 
            text="🧠 Predictor de Trastorno del Espectro Autista",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Notebook para pestañas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Pestaña 1: Datos simulados
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Datos Simulados")
        
        # Contenido pestaña 1
        info_label1 = tk.Label(
            tab1,
            text="Haz clic para realizar una predicción con datos de prueba",
            font=("Arial", 12),
            wraplength=400
        )
        info_label1.pack(pady=20)
        
        predict_btn = tk.Button(
            tab1,
            text="🔮 Predecir con Datos Simulados",
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            command=self.predecir_simulado,
            height=2,
            width=25
        )
        predict_btn.pack(pady=10)
        
        # Resultado
        self.result_label = tk.Label(
            tab1,
            text="Resultado aparecerá aquí",
            font=("Arial", 14, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50',
            wraplength=400,
            justify='center',
            height=3
        )
        self.result_label.pack(pady=20, padx=20, fill='x')
        
        # Pestaña 2: Datos manuales
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Datos Personalizados")
        
        # Contenido pestaña 2
        info_label2 = tk.Label(
            tab2,
            text="Ajusta los valores y haz clic en predecir",
            font=("Arial", 12)
        )
        info_label2.pack(pady=10)
        
        # Frame para sliders
        self.sliders_frame = ttk.Frame(tab2)
        self.sliders_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        self.sliders = []
        if self.interpreter:
            input_shape = self.input_details[0]['shape'][1]
            
            # Crear canvas con scrollbar para muchos sliders
            canvas = tk.Canvas(self.sliders_frame, height=250)
            scrollbar = ttk.Scrollbar(self.sliders_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Crear sliders
            for i in range(min(input_shape, 20)):  # Limitar a 20 sliders
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill='x', padx=5, pady=2)
                
                label = tk.Label(frame, text=f"Característica {i+1}:", width=15)
                label.pack(side='left')
                
                slider = tk.Scale(
                    frame,
                    from_=0.0,
                    to=1.0,
                    resolution=0.1,
                    orient='horizontal',
                    length=200
                )
                slider.set(0.5)
                slider.pack(side='left', padx=10)
                
                self.sliders.append(slider)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
        
        # Botón predecir personalizado
        predict_custom_btn = tk.Button(
            tab2,
            text="🔮 Predecir con Datos Personalizados",
            font=("Arial", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            command=self.predecir_personalizado,
            height=2,
            width=30
        )
        predict_custom_btn.pack(pady=10)
        
        # Resultado personalizado
        self.result_custom_label = tk.Label(
            tab2,
            text="Resultado aparecerá aquí",
            font=("Arial", 14, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50',
            wraplength=400,
            justify='center',
            height=3
        )
        self.result_custom_label.pack(pady=10, padx=20, fill='x')
    
    def predecir_simulado(self):
        if not self.interpreter:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        try:
            # Datos simulados
            dummy_input = np.array([[0.5] * self.input_details[0]['shape'][1]], dtype=np.float32)
            
            # Predicción
            resultado, confianza = self.realizar_prediccion(dummy_input)
            
            # Mostrar resultado
            self.result_label.config(
                text=f"Diagnóstico: {resultado}\nConfianza: {confianza:.1f}%",
                bg='#d5edf5' if resultado == 'Desarrollo típico' else '#fdeaa7'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
    
    def predecir_personalizado(self):
        if not self.interpreter:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        try:
            # Obtener valores de sliders
            valores = [slider.get() for slider in self.sliders]
            
            # Completar con valores por defecto si faltan
            input_shape = self.input_details[0]['shape'][1]
            while len(valores) < input_shape:
                valores.append(0.5)
            
            input_data = np.array([valores[:input_shape]], dtype=np.float32)
            
            # Predicción
            resultado, confianza = self.realizar_prediccion(input_data)
            
            # Mostrar resultado
            self.result_custom_label.config(
                text=f"Diagnóstico: {resultado}\nConfianza: {confianza:.1f}%",
                bg='#d5edf5' if resultado == 'Desarrollo típico' else '#fdeaa7'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
    
    def realizar_prediccion(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        pred_idx = int(np.argmax(output_data))
        confianza = float(np.max(output_data)) * 100
        
        etiquetas = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
        resultado = etiquetas[pred_idx] if pred_idx < len(etiquetas) else "Resultado desconocido"
        
        return resultado, confianza

def main():
    root = tk.Tk()
    app = TEAPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
