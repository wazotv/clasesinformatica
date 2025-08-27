#!/usr/bin/env python3
"""
Sistema de Detección de Cámara Optimizado para Raspberry Pi 4
=============================================================

Este código está optimizado específicamente para Raspberry Pi 4 con las siguientes mejoras:
- Uso de GPIO para control de lámparas UV
- Optimización de rendimiento para CPU ARM
- Interfaz web ligera usando Flask
- Configuración específica para cámara Pi Camera
- Gestión eficiente de memoria
- Control remoto vía web

Autor: Sistema UVC-BOT
Versión: 1.0 para Raspberry Pi 4
"""

import datetime
import os
import threading
import time
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from tflite_runtime.interpreter import Interpreter

# Configuración específica para Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("✅ GPIO disponible - Configurando pines para lámparas UV")
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ GPIO no disponible - Modo simulación")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uvc_bot_raspberry.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== CONFIGURACIÓN RASPBERY PI =====
# Usar ruta del archivo para encontrar el modelo sin depender del cwd
MODEL_PATH = str(Path(__file__).with_name("detect.tflite"))
UMBRAL_INIT = 0.5
V_MIN, V_MAX = 0.0, 30.0  # Velocidades reducidas para Pi
W_MIN, W_MAX = -0.8, 0.8
V_RES, W_RES = 10.0, 0.2
DT = 0.15  # Intervalo aumentado para reducir carga CPU
P_HORIZON = 0.8

# Configuración de GPIO para lámparas UV
GPIO_PINS = {
    'lamps_down': [17, 18, 27, 22],  # Lámparas 1-4 (abajo)
    'lamps_up': [23, 24, 25, 8],     # Lámparas 5-8 (arriba)
    'status_led': 12,                 # LED de estado
    'emergency_stop': 16              # Botón de parada de emergencia
}

# Configuración de cámara Pi
CAMERA_CONFIG = {
    'resolution': (640, 480),  # Resolución reducida para mejor rendimiento
    'fps': 15,                 # FPS limitado para ahorrar recursos
    'rotation': 0,             # Rotación de la cámara
    'flip': False              # Voltear imagen
}

# ===== INICIALIZACIÓN GPIO =====
def setup_gpio():
    """Configurar GPIO para control de lámparas UV"""
    if not GPIO_AVAILABLE:
        logger.warning("GPIO no disponible - usando modo simulación")
        return False
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Configurar pines de lámparas como salidas
        for pin in GPIO_PINS['lamps_down'] + GPIO_PINS['lamps_up']:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)  # Inicialmente apagadas
        
        # Configurar LED de estado
        GPIO.setup(GPIO_PINS['status_led'], GPIO.OUT)
        GPIO.output(GPIO_PINS['status_led'], GPIO.HIGH)
        
        # Configurar botón de parada de emergencia
        GPIO.setup(GPIO_PINS['emergency_stop'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        logger.info("✅ GPIO configurado correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error configurando GPIO: {e}")
        return False

def cleanup_gpio():
    """Limpiar configuración GPIO"""
    if GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
            logger.info("GPIO limpiado")
        except:
            pass

# ===== CONTROL DE LÁMPARAS UV =====
class UVLampController:
    def __init__(self):
        self.gpio_available = setup_gpio()
        self.lamp_states = {
            'down': [False] * 4,
            'up': [False] * 4
        }
        self.emergency_stop = False
        
    def toggle_lamp(self, position, index):
        """Activar/desactivar lámpara específica"""
        if position not in ['down', 'up'] or index < 0 or index > 3:
            return False
            
        self.lamp_states[position][index] = not self.lamp_states[position][index]
        state = self.lamp_states[position][index]
        
        if self.gpio_available:
            pin = GPIO_PINS[f'lamps_{position}'][index]
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        
        status = "ON" if state else "OFF"
        logger.info(f"Lámpara {position} {index+1}: {status}")
        return True
    
    def toggle_all_lamps(self, state=None):
        """Activar/desactivar todas las lámparas"""
        if state is None:
            # Invertir estado actual
            current_state = any(self.lamp_states['down'] + self.lamp_states['up'])
            state = not current_state
        
        for position in ['down', 'up']:
            for i in range(4):
                self.lamp_states[position][i] = state
                if self.gpio_available:
                    pin = GPIO_PINS[f'lamps_{position}'][i]
                    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        
        status = "ENCENDIDAS" if state else "APAGADAS"
        logger.info(f"Todas las lámparas: {status}")
        return True
    
    def emergency_stop_lamps(self):
        """Parada de emergencia - apagar todas las lámparas"""
        self.toggle_all_lamps(False)
        self.emergency_stop = True
        logger.warning("🚨 PARADA DE EMERGENCIA ACTIVADA")
    
    def get_lamp_status(self):
        """Obtener estado actual de todas las lámparas"""
        return {
            'down': self.lamp_states['down'].copy(),
            'up': self.lamp_states['up'].copy(),
            'emergency_stop': self.emergency_stop
        }

# ===== DETECTOR OPTIMIZADO PARA PI =====
class PiDetector:
    def __init__(self, model_path):
        try:
            # Establecer hilos mediante argumento del constructor para mejor compatibilidad
            self.interp = Interpreter(model_path=model_path, num_threads=4)
            self.interp.allocate_tensors()
            
            inp = self.interp.get_input_details()[0]
            out = self.interp.get_output_details()
            _, h, w, _ = inp['shape']
            
            self.idx_in = inp['index']
            self.isf = inp['dtype'] == np.float32
            self.shape = (w, h)
            self.out_idx = [out[i]['index'] for i in range(3)]
            
            logger.info(f"✅ Detector inicializado - Input: {self.shape}")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando detector: {e}")
            raise

    def preprocess(self, frame):
        """Preprocesamiento optimizado para Pi"""
        try:
            # Redimensionar para mejor rendimiento
            img = cv2.resize(frame, self.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalización optimizada
            if self.isf:
                arr = img.astype(np.float32)[None]
                return (arr - 127.5) / 127.5
            else:
                return img.astype(np.uint8)[None]
                
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return None

    def detect(self, data):
        """Detección con manejo de errores"""
        try:
            self.interp.set_tensor(self.idx_in, data)
            self.interp.invoke()
            
            boxes = self.interp.get_tensor(self.out_idx[0])[0]
            classes = self.interp.get_tensor(self.out_idx[1])[0].astype(int)
            scores = self.interp.get_tensor(self.out_idx[2])[0]
            
            return boxes, classes, scores
            
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return [], [], []

# ===== NMS OPTIMIZADO =====
def non_max_suppression(boxes, scores, thr=0.3):
    """NMS optimizado para Pi"""
    if not boxes:
        return []
    
    try:
        B, S = np.array(boxes), np.array(scores)
        x1, y1, x2, y2 = B.T
        areas = (x2 - x1) * (y2 - y1)
        order = S.argsort()[::-1]
        keep = []
        
        while order.size:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            inds = np.where(ovr <= thr)[0]
            order = order[inds + 1] if inds.size else np.array([], int)
            
        return keep
        
    except Exception as e:
        logger.error(f"Error en NMS: {e}")
        return []

# ===== CAPTURA DE CÁMARA OPTIMIZADA =====
class PiCameraCapture:
    def __init__(self):
        self.cap = None
        self.camera_available = False
        self.setup_camera()
    
    def setup_camera(self):
        """Configurar cámara Pi optimizada"""
        try:
            # Intentar usar cámara Pi específica
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                # Fallback a cámara USB
                self.cap = cv2.VideoCapture(1)
            
            if self.cap.isOpened():
                # Configurar parámetros optimizados
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['resolution'][0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['resolution'][1])
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
                
                self.camera_available = True
                logger.info("✅ Cámara configurada correctamente")
            else:
                logger.error("❌ No se pudo abrir la cámara")
                
        except Exception as e:
            logger.error(f"❌ Error configurando cámara: {e}")
    
    def read_frame(self):
        """Leer frame con optimizaciones"""
        if not self.camera_available or self.cap is None:
            return None, None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            
            # Aplicar rotación si es necesaria
            if CAMERA_CONFIG['rotation'] != 0:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, CAMERA_CONFIG['rotation'], 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))
            
            # Aplicar flip si es necesario
            if CAMERA_CONFIG['flip']:
                frame = cv2.flip(frame, 1)
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error leyendo frame: {e}")
            return None, None
    
    def release(self):
        """Liberar recursos de cámara"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# ===== SISTEMA PRINCIPAL =====
class UVCRobotSystem:
    def __init__(self):
        self.lamp_controller = UVLampController()
        self.camera = PiCameraCapture()
        self.detector = None
        self.running = False
        self.shared_data = {
            'frame': None,
            'detections': [],
            'detected': False,
            'fps': 0,
            'threshold': UMBRAL_INIT,
            'camera_status': 'Desconectada',
            'system_status': 'Iniciando'
        }
        self.lock = threading.Lock()
        
        # Inicializar detector
        try:
            self.detector = PiDetector(MODEL_PATH)
            logger.info("✅ Sistema inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
    
    def start_processing(self):
        """Iniciar procesamiento en hilo separado"""
        if self.running:
            return
            
        self.running = True
        with self.lock:
            self.shared_data['system_status'] = 'Operando'
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("🔄 Procesamiento iniciado")
    
    def stop_processing(self):
        """Detener procesamiento"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        with self.lock:
            self.shared_data['system_status'] = 'Detenido'
        logger.info("⏹️ Procesamiento detenido")
    
    def _processing_loop(self):
        """Bucle principal de procesamiento"""
        fps_counter = 0
        last_time = time.time()
        
        while self.running:
            try:
                t0 = time.time()
                
                # Leer frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Actualizar estado de cámara
                with self.lock:
                    self.shared_data['camera_status'] = 'Conectada'
                    self.shared_data['frame'] = frame.copy()
                
                # Detección
                if self.detector:
                    preprocessed = self.detector.preprocess(frame)
                    if preprocessed is not None:
                        boxes, classes, scores = self.detector.detect(preprocessed)
                        
                        # Filtrar detecciones
                        detections = []
                        threshold = self.shared_data['threshold']
                        
                        for b, cls, sc in zip(boxes, classes, scores):
                            if cls == 0 and sc >= threshold:  # Clase 0 = objeto detectado
                                y1, x1, y2, x2 = (b * [frame.shape[0], frame.shape[1],
                                                       frame.shape[0], frame.shape[1]]).astype(int)
                                if x2 - x1 > 30 and y2 - y1 > 30:  # Tamaño mínimo
                                    detections.append((x1, y1, x2, y2, sc))
                        
                        # Aplicar NMS
                        if detections:
                            idxs = non_max_suppression([d[:4] for d in detections], [d[4] for d in detections])
                            detections = [detections[i] for i in idxs]
                        
                        # Actualizar datos compartidos
                        with self.lock:
                            self.shared_data['detections'] = detections
                            self.shared_data['detected'] = bool(detections)
                
                # Calcular FPS
                fps_counter += 1
                if time.time() - last_time >= 1.0:
                    with self.lock:
                        self.shared_data['fps'] = fps_counter
                    fps_counter = 0
                    last_time = time.time()
                
                # Control de tiempo
                elapsed = time.time() - t0
                sleep_time = max(0, DT - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error en bucle de procesamiento: {e}")
                time.sleep(0.1)
    
    def get_status(self):
        """Obtener estado completo del sistema"""
        with self.lock:
            status = self.shared_data.copy()
            status['lamp_status'] = self.lamp_controller.get_lamp_status()
            status['timestamp'] = datetime.datetime.now().isoformat()
        return status
    
    def cleanup(self):
        """Limpiar recursos"""
        self.stop_processing()
        self.camera.release()
        cleanup_gpio()
        logger.info("🧹 Recursos limpiados")

# ===== INTERFAZ WEB CON FLASK =====
app = Flask(__name__)
robot_system = None

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """API para obtener estado del sistema"""
    if robot_system:
        return jsonify(robot_system.get_status())
    return jsonify({'error': 'Sistema no inicializado'})

@app.route('/api/lamp/<position>/<int:index>', methods=['POST'])
def toggle_lamp(position, index):
    """API para controlar lámparas"""
    if robot_system:
        success = robot_system.lamp_controller.toggle_lamp(position, index)
        return jsonify({'success': success})
    return jsonify({'error': 'Sistema no inicializado'})

@app.route('/api/lamp/all', methods=['POST'])
def toggle_all_lamps():
    """API para controlar todas las lámparas"""
    if robot_system:
        data = request.get_json()
        state = data.get('state') if data else None
        success = robot_system.lamp_controller.toggle_all_lamps(state)
        return jsonify({'success': success})
    return jsonify({'error': 'Sistema no inicializado'})

@app.route('/api/threshold', methods=['POST'])
def update_threshold():
    """API para actualizar umbral de detección"""
    if robot_system:
        data = request.get_json()
        threshold = data.get('threshold', UMBRAL_INIT)
        with robot_system.lock:
            robot_system.shared_data['threshold'] = float(threshold)
        return jsonify({'success': True})
    return jsonify({'error': 'Sistema no inicializado'})

@app.route('/video_feed')
def video_feed():
    """Stream de video"""
    def generate():
        while True:
            if robot_system and robot_system.shared_data['frame'] is not None:
                with robot_system.lock:
                    frame = robot_system.shared_data['frame'].copy()
                    detections = robot_system.shared_data['detections']
                
                # Dibujar detecciones
                for x1, y1, x2, y2, score in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{score:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Codificar frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== TEMPLATE HTML =====
def create_html_template():
    """Crear template HTML para la interfaz web"""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UVC-BOT Raspberry Pi</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
        }
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
        }
        .video-container {
            text-align: center;
        }
        .video-feed {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            border: 3px solid rgba(255,255,255,0.3);
        }
        .lamp-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }
        .lamp-btn {
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255,255,255,0.2);
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        .lamp-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        .lamp-btn.active {
            background: #4CAF50;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            background: rgba(255,255,255,0.1);
        }
        .control-btn {
            width: 100%;
            padding: 12px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            background: #2196F3;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        .control-btn:hover {
            background: #1976D2;
        }
        .emergency-btn {
            background: #f44336;
        }
        .emergency-btn:hover {
            background: #d32f2f;
        }
    </style>
    <script>
      // Evitar zoom doble en iOS (mejor UX)
      document.addEventListener('gesturestart', function (e) { e.preventDefault(); });
    </script>
    
    
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 UVC-BOT Raspberry Pi</h1>
            <p>Sistema de Desinfección UV Inteligente</p>
        </div>
        
        <div class="grid">
            <!-- Panel de Control de Lámparas -->
            <div class="panel">
                <h3>💡 Control de Lámparas UV</h3>
                
                <h4>Lámparas Abajo (1-4)</h4>
                <div class="lamp-grid" id="lamps-down"></div>
                
                <h4>Lámparas Arriba (5-8)</h4>
                <div class="lamp-grid" id="lamps-up"></div>
                
                <button class="control-btn" onclick="toggleAllLamps()">
                    🔆 Todas las Lámparas
                </button>
                
                <button class="control-btn emergency-btn" onclick="emergencyStop()">
                    🚨 Parada de Emergencia
                </button>
            </div>
            
            <!-- Panel de Video -->
            <div class="panel">
                <div class="video-container">
                    <h3>📹 Visión de Cámara</h3>
                    <img src="/video_feed" class="video-feed" alt="Video Feed">
                    
                    <div class="status">
                        <strong>FPS:</strong> <span id="fps">0</span> |
                        <strong>Detecciones:</strong> <span id="detections">0</span> |
                        <strong>Umbral:</strong> <input type="range" id="threshold" min="0" max="1" step="0.05" value="0.5">
                        <span id="threshold-value">0.5</span>
                    </div>
                </div>
            </div>
            
            <!-- Panel de Estado -->
            <div class="panel">
                <h3>📊 Estado del Sistema</h3>
                
                <div class="status">
                    <strong>Cámara:</strong> <span id="camera-status">Desconectada</span>
                </div>
                
                <div class="status">
                    <strong>Sistema:</strong> <span id="system-status">Iniciando</span>
                </div>
                
                <div class="status">
                    <strong>Última Actualización:</strong> <span id="timestamp">-</span>
                </div>
                
                <h4>🎮 Control Manual</h4>
                <button class="control-btn" onclick="sendCommand('forward')">⬆️ Adelante</button>
                <button class="control-btn" onclick="sendCommand('backward')">⬇️ Atrás</button>
                <button class="control-btn" onclick="sendCommand('left')">⬅️ Izquierda</button>
                <button class="control-btn" onclick="sendCommand('right')">➡️ Derecha</button>
                <button class="control-btn" onclick="sendCommand('stop')">⏹️ Parar</button>
            </div>
        </div>
    </div>

    <script>
        // Variables globales
        let lampStates = {down: [false, false, false, false], up: [false, false, false, false]};
        
        // Inicializar interfaz
        function initInterface() {
            createLampButtons();
            updateStatus();
            setInterval(updateStatus, 1000);
            
            // Configurar slider de umbral
            document.getElementById('threshold').addEventListener('input', function(e) {
                const value = e.target.value;
                document.getElementById('threshold-value').textContent = value;
                updateThreshold(value);
            });
        }
        
        // Crear botones de lámparas
        function createLampButtons() {
            const downContainer = document.getElementById('lamps-down');
            const upContainer = document.getElementById('lamps-up');
            
            for (let i = 0; i < 4; i++) {
                const downBtn = document.createElement('button');
                downBtn.className = 'lamp-btn';
                downBtn.textContent = `L${i+1}`;
                downBtn.onclick = () => toggleLamp('down', i);
                downContainer.appendChild(downBtn);
                
                const upBtn = document.createElement('button');
                upBtn.className = 'lamp-btn';
                upBtn.textContent = `L${i+5}`;
                upBtn.onclick = () => toggleLamp('up', i);
                upContainer.appendChild(upBtn);
            }
        }
        
        // Actualizar estado
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps || 0;
                    document.getElementById('detections').textContent = data.detections?.length || 0;
                    document.getElementById('camera-status').textContent = data.camera_status || 'Desconectada';
                    document.getElementById('system-status').textContent = data.system_status || 'Iniciando';
                    document.getElementById('timestamp').textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '-';
                    
                    // Actualizar estados de lámparas
                    if (data.lamp_status) {
                        lampStates = data.lamp_status;
                        updateLampButtons();
                    }
                })
                .catch(error => console.error('Error actualizando estado:', error));
        }
        
        // Actualizar botones de lámparas
        function updateLampButtons() {
            const buttons = document.querySelectorAll('.lamp-btn');
            let buttonIndex = 0;
            
            // Lámparas abajo
            for (let i = 0; i < 4; i++) {
                buttons[buttonIndex].className = `lamp-btn ${lampStates.down[i] ? 'active' : ''}`;
                buttonIndex++;
            }
            
            // Lámparas arriba
            for (let i = 0; i < 4; i++) {
                buttons[buttonIndex].className = `lamp-btn ${lampStates.up[i] ? 'active' : ''}`;
                buttonIndex++;
            }
        }
        
        // Controlar lámpara individual
        function toggleLamp(position, index) {
            fetch(`/api/lamp/${position}/${index}`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus();
                    }
                })
                .catch(error => console.error('Error controlando lámpara:', error));
        }
        
        // Controlar todas las lámparas
        function toggleAllLamps() {
            const allOn = lampStates.down.every(state => state) && lampStates.up.every(state => state);
            fetch('/api/lamp/all', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({state: !allOn})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus();
                }
            })
            .catch(error => console.error('Error controlando todas las lámparas:', error));
        }
        
        // Parada de emergencia
        function emergencyStop() {
            if (confirm('¿Está seguro de activar la parada de emergencia?')) {
                fetch('/api/lamp/all', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({state: false})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus();
                        alert('Parada de emergencia activada');
                    }
                })
                .catch(error => console.error('Error en parada de emergencia:', error));
            }
        }
        
        // Actualizar umbral
        function updateThreshold(value) {
            fetch('/api/threshold', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({threshold: parseFloat(value)})
            })
            .catch(error => console.error('Error actualizando umbral:', error));
        }
        
        // Enviar comando de movimiento
        function sendCommand(command) {
            console.log('Comando enviado:', command);
            // Aquí se implementaría la lógica para enviar comandos al robot
        }
        
        // Inicializar cuando se carga la página
        window.onload = initInterface;
    </script>
</body>
</html>
    """
    
    with open(template_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info("✅ Template HTML creado")

# ===== FUNCIÓN PRINCIPAL =====
def main():
    """Función principal"""
    global robot_system
    
    try:
        logger.info("🚀 Iniciando UVC-BOT Raspberry Pi...")
        
        # Crear template HTML
        create_html_template()
        
        # Inicializar sistema
        robot_system = UVCRobotSystem()
        
        # Iniciar procesamiento
        robot_system.start_processing()
        
        # Configurar parada de emergencia
        if GPIO_AVAILABLE:
            def emergency_callback(channel):
                robot_system.lamp_controller.emergency_stop_lamps()
            
            GPIO.add_event_detect(GPIO_PINS['emergency_stop'], GPIO.FALLING, 
                                callback=emergency_callback, bouncetime=300)
        
        logger.info("✅ Sistema iniciado correctamente")
        logger.info("🌐 Interfaz web disponible en: http://localhost:5000")
        
        # Iniciar servidor Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupción del usuario")
    except Exception as e:
        logger.error(f"❌ Error en función principal: {e}")
    finally:
        if robot_system:
            robot_system.cleanup()
        logger.info("👋 Sistema cerrado")

if __name__ == "__main__":
    main()

