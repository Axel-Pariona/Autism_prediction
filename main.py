from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
import numpy as np
import tflite_runtime.interpreter as tflite
from kivy.utils import platform
import os

# Ruta correcta para Android o PC
if platform == "android":
    from android.storage import app_storage_path
    app_path = app_storage_path()
    model_path = os.path.join(app_path, 'modelo_autismo.tflite')
else:
    model_path = 'modelo_autismo.tflite'

class TEAPredictor(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Cargar modelo TFLite usando la ruta correcta
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Botón y resultado
        self.result_label = Label(text="Resultado aparecerá aquí", font_size=20)
        self.add_widget(self.result_label)

        predict_button = Button(text="Predecir con datos simulados", size_hint=(1, 0.2), font_size=18)
        predict_button.bind(on_press=self.predecir)
        self.add_widget(predict_button)

    def predecir(self, instance):
        # Simula entrada (ajusta la dimensión si tu modelo lo requiere)
        dummy_input = np.array([[0.5] * self.input_details[0]['shape'][1]], dtype=np.float32)

        # Ejecutar inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        pred_idx = int(np.argmax(output_data))
        etiquetas = ['Desarrollo típico', 'TEA - Nivel 1', 'TEA - Nivel 2', 'TEA - Nivel 3', 'Indeterminado']
        resultado = etiquetas[pred_idx] if pred_idx < len(etiquetas) else "Resultado desconocido"
        self.result_label.text = f"Diagnóstico: {resultado}"

class AutismoApp(App):
    def build(self):
        return TEAPredictor()

if __name__ == "__main__":
    AutismoApp().run()
