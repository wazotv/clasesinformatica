UVC-BOT Raspberry Pi 4
======================

Sistema de detección con cámara y control de lámparas UV optimizado para Raspberry Pi 4.

Requisitos
----------
- Raspberry Pi 4 (32 o 64 bits)
- Raspberry Pi OS actualizado
- Cámara (CSI o USB)
- Python 3

Preparación de la Raspberry Pi
------------------------------
1. Habilita la cámara.
   - En sistemas con legacy: `sudo raspi-config` → Interface Options → Camera
   - En Bookworm con libcamera: prueba `libcamera-hello` para validar la cámara
2. Reinicia: `sudo reboot`

Despliegue
----------
1. Copia el proyecto a la Pi, p. ej. `/home/pi/uvc-bot-rpi`.
2. Copia el modelo `detect.tflite` junto a `app.py`.

Instalación
-----------
```bash
cd /home/pi/uvc-bot-rpi
chmod +x setup_pi.sh
./setup_pi.sh
```

Ejecución (desarrollo)
----------------------
```bash
source .venv/bin/activate
python app.py
```
Abrir en navegador: `http://<IP_PI>:5000`

Servicio systemd (opcional)
---------------------------
```bash
sudo cp /home/pi/uvc-bot-rpi/uvc-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable uvc-bot
sudo systemctl start uvc-bot
```

Notas
-----
- OpenCV se instala vía `apt` (python3-opencv).
- `tflite-runtime` se instala según arquitectura en `setup_pi.sh`.
- Pines GPIO configurables en `GPIO_PINS` de `app.py`.

