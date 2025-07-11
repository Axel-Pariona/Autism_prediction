import sys
import os
import numpy as np
import tflite_runtime.interpreter as tflite
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QSlider, QTabWidget, 
                             QScrollArea, QFrame, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

class PredictionThread(QThread):
    result_ready = pyqtSignal(str, float)
    
    def __init__(self, interpreter, input_data):
        super().__init__()
        self.interpreter = interpreter
        self.input_data = input_data
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
    
    def run(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            pred_idx = int(np.argmax(output_data))
            confianza = float(np.max(output_data)) * 100
            
            etiquetas = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
            resultado = etiquetas[pred_idx] if pred_idx < len(etiquetas) else "Resultado desconocido"
            
            self.result_ready.emit(resultado, confianza)
        except Exception as e:
            self.result_ready.emit(f"Error: {str(e)}", 0.0)

class TEAPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpreter = None
        self.sliders = []
        self.init_model()
        self.init_ui()
    
    def init_model(self):
        model_path = 'modelo_autismo.tflite'
        if os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando modelo: {str(e)}")
        else:
            QMessageBox.critical(self, "Error", "No se encontró el archivo modelo_autismo.tflite")
    
    def init_ui(self):
        self.setWindowTitle('🧠 Predictor de TEA')
        self.setGeometry(100, 100, 700, 600)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Título
        title = QLabel('🧠 Predictor de Trastorno del Espectro Autista')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin: 20px;")
        layout.addWidget(title)
        
        # Pestañas
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Pestaña 1: Datos simulados
        tab1 = QWidget()
        tabs.addTab(tab1, "Datos Simulados")
        
        tab1_layout = QVBoxLayout()
        tab1.setLayout(tab1_layout)
        
        # Descripción
        desc1 = QLabel("Haz clic para realizar una predicción con datos de prueba")
        desc1.setAlignment(Qt.AlignCenter)
        desc1.setFont(QFont('Arial', 12))
        desc1.setWordWrap(True)
        desc1.setStyleSheet("margin: 20px; color: #34495e;")
        tab1_layout.addWidget(desc1)
        
        # Botón predecir simulado
        self.btn_simulado = QPushButton('🔮 Predecir con Datos Simulados')
        self.btn_simulado.setFont(QFont('Arial', 12, QFont.Bold))
        self.btn_simulado.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.btn_simulado.clicked.connect(self.predecir_simulado)
        tab1_layout.addWidget(self.btn_simulado)
        
        # Barra de progreso
        self.progress_bar1 = QProgressBar()
        self.progress_bar1.setVisible(False)
        tab1_layout.addWidget(self.progress_bar1)
        
        # Resultado simulado
        self.result_label1 = QLabel("Resultado aparecerá aquí")
        self.result_label1.setAlignment(Qt.AlignCenter)
        self.result_label1.setFont(QFont('Arial', 14, QFont.Bold))
        self.result_label1.setWordWrap(True)
        self.result_label1.setStyleSheet("""
            background-color: #ecf0f1;
            color: #2c3e50;
            padding: 20px;
            border-radius: 8px;
            margin: 20px;
            min-height: 80px;
        """)
        tab1_layout.addWidget(self.result_label1)
        
        tab1_layout.addStretch()
        
        # Pestaña 2: Datos personalizados
        tab2 = QWidget()
        tabs.addTab(tab2, "Datos Personalizados")
        
        tab2_layout = QVBoxLayout()
        tab2.setLayout(tab2_layout)
        
        # Descripción
        desc2 = QLabel("Ajusta los valores usando los deslizadores y haz clic en predecir")
        desc2.setAlignment(Qt.AlignCenter)
        desc2.setFont(QFont('Arial', 12))
        desc2.setWordWrap(True)
        desc2.setStyleSheet("margin: 10px; color: #34495e;")
        tab2_layout.addWidget(desc2)
        
        # Área de scroll para sliders
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        
        # Crear sliders
        if self.interpreter:
            input_shape = self.input_details[0]['shape'][1]
            
            for i in range(min(input_shape, 20)):  # Limitar a 20 sliders
                frame = QFrame()
                frame_layout = QHBoxLayout()
                frame.setLayout(frame_layout)
                
                label = QLabel(f"Característica {i+1}:")
                label.setMinimumWidth(120)
                frame_layout.addWidget(label)
                
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(100)
                slider.setValue(50)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(20)
                frame_layout.addWidget(slider)
                
                value_label = QLabel("0.5")
                value_label.setMinimumWidth(40)
                slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v/100:.1f}"))
                frame_layout.addWidget(value_label)
                
                self.sliders.append(slider)
                scroll_layout.addWidget(frame)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(300)
        tab2_layout.addWidget(scroll_area)
        
        # Botón predecir personalizado
        self.btn_personalizado = QPushButton('🔮 Predecir con Datos Personalizados')
        self.btn_personalizado.setFont(QFont('Arial', 12, QFont.Bold))
        self.btn_personalizado.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.btn_personalizado.clicked.connect(self.predecir_personalizado)
        tab2_layout.addWidget(self.btn_personalizado)
        
        # Barra de progreso
        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setVisible(False)
        tab2_layout.addWidget(self.progress_bar2)
        
        # Resultado personalizado
        self.result_label2 = QLabel("Resultado aparecerá aquí")
        self.result_label2.setAlignment(Qt.AlignCenter)
        self.result_label2.setFont(QFont('Arial', 14, QFont.Bold))
        self.result_label2.setWordWrap(True)
        self.result_label2.setStyleSheet("""
            background-color: #ecf0f1;
            color: #2c3e50;
            padding: 20px;
            border-radius: 8px;
            margin: 20px;
            min-height: 80px;
        """)
        tab2_layout.addWidget(self.result_label2)
    
    def predecir_simulado(self):
        if not self.interpreter:
            QMessageBox.critical(self, "Error", "Modelo no cargado")
            return
        
        self.btn_simulado.setEnabled(False)
        self.progress_bar1.setVisible(True)
        self.progress_bar1.setRange(0, 0)  # Progreso indeterminado
        
        # Datos simulados
        dummy_input = np.array([[0.5] * self.input_details[0]['shape'][1]], dtype=np.float32)
        
        # Ejecutar predicción en hilo separado
        self.prediction_thread = PredictionThread(self.interpreter, dummy_input)
        self.prediction_thread.result_ready.connect(lambda result, conf: self.on_result_ready(result, conf, 1))
        self.prediction_thread.start()
    
    def predecir_personalizado(self):
        if not self.interpreter:
            QMessageBox.critical(self, "Error", "Modelo no cargado")
            return
        
        self.btn_personalizado.setEnabled(False)
        self.progress_bar2.setVisible(True)
        self.progress_bar2.setRange(0, 0)
        
        # Obtener valores de sliders
        valores = [slider.value() / 100.0 for slider in self.sliders]
        
        # Completar con valores por defecto si faltan
        input_shape = self.input_details[0]['shape'][1]
        while len(valores) < input_shape:
            valores.append(0.5)
        
        input_data = np.array([valores[:input_shape]], dtype=np.float32)
        
        # Ejecutar predicción en hilo separado
        self.prediction_thread = PredictionThread(self.interpreter, input_data)
        self.prediction_thread.result_ready.connect(lambda result, conf: self.on_result_ready(result, conf, 2))
        self.prediction_thread.start()
    
    def on_result_ready(self, resultado, confianza, tab_num):
        if tab_num == 1:
            self.btn_simulado.setEnabled(True)
            self.progress_bar1.setVisible(False)
            label = self.result_label1
        else:
            self.btn_personalizado.setEnabled(True)
            self.progress_bar2.setVisible(False)
            label = self.result_label2
        
        # Actualizar resultado
        text = f"Diagnóstico: {resultado}\nConfianza: {confianza:.1f}%"
        label.setText(text)
        
        # Cambiar color según resultado
        if "Error" in resultado:
            bg_color = "#e74c3c"
            text_color = "white"
        elif resultado == 'Desarrollo típico':
            bg_color = "#2ecc71"
            text_color = "white"
        else:
            bg_color = "#f39c12"
            text_color = "white"
        
        label.setStyleSheet(f"""
            background-color: {bg_color};
            color: {text_color};
            padding: 20px;
            border-radius: 8px;
            margin: 20px;
            min-height: 80px;
        """)

def main():
    app = QApplication(sys.argv)
    
    # Configurar estilo de la aplicación
    app.setStyle('Fusion')
    
    window = TEAPredictorApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
