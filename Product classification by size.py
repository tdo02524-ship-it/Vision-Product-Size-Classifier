import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

SERIAL_BAUD = 115200
SERIAL_PORT = None  # None -> auto detect

DEFAULT_SMALL_THR_MM = 300
DEFAULT_MED_THR_MM   = 500

# Chống spam gửi class (ms)
SEND_COOLDOWN_MS = 600

def open_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    return cap if (cap and cap.isOpened()) else None


def put_text(img, text, org, scale=0.75, thick=2, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def classify_size_mm(long_px, mm_per_pixel, small_thr_mm, med_thr_mm):
    mmpp = float(mm_per_pixel) if mm_per_pixel > 0 else 1.0
    long_mm = float(long_px) * mmpp
    if long_mm <= small_thr_mm:
        return "Small", long_mm
    elif long_mm <= med_thr_mm:
        return "Fit", long_mm
    else:
        return "Big", long_mm

def detect_objects_classic(frame_bgr, mm_per_pixel, small_thr_mm, med_thr_mm):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        long_px = float(max(w, h))
        cls, size_mm = classify_size_mm(long_px, mm_per_pixel, small_thr_mm, med_thr_mm)
        detections.append({
            "cls": cls,
            "bbox": (int(x), int(y), int(w), int(h)),
            "size_mm": float(size_mm),
            "area": float(area)
        })
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


def auto_find_arduino_port():
    if serial is None:
        return None
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    keywords = ["Arduino", "CH340", "USB Serial", "CP210", "CP210x",
                "Silicon Labs", "ESP32", "UART"]
    for p in ports:
        desc = (p.description or "")
        hwid = (p.hwid or "")
        if any(k.lower() in desc.lower() for k in keywords) or any(k.lower() in hwid.lower() for k in keywords):
            return p.device

    if len(ports) == 1:
        return ports[0].device

    return None

class SerialReader(QtCore.QThread):
    line_received = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)
    def __init__(self, port, baud, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.ser = None
        self._running = True

    def connect_serial(self):
        if serial is None:
            self.status.emit("pyserial chưa cài (pip install pyserial).")
            return False
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.status.emit(f"OK: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            self.ser = None
            self.status.emit(f"OFF: {e}")
            return False

    def run(self):
        if not self.connect_serial():
            return

        buf = b""
        while self._running:
            try:
                data = self.ser.read(128)
                if data:
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            s = line.decode(errors="ignore").strip()
                        except Exception:
                            s = ""
                        if s:
                            self.line_received.emit(s)
                else:
                    self.msleep(10)
            except Exception as e:
                self.status.emit(f"LỖI: {e}")
                break

        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def write_text(self, txt: str):
        if not self.ser:
            return False
        try:
            self.ser.write(txt.encode("ascii"))
            return True
        except Exception as e:
            self.status.emit(f"Gửi lỗi: {e}")
            return False

    def stop(self):
        self._running = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân loại sản phẩm - Camera + ESP32 (SCAN handshake)")
        self.resize(1400, 820)
        # ---- state ----
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.last_frame_bgr = None
        # mm/pixel: nếu 0 -> fallback 1px = 1mm
        self.mm_per_pixel = 0.0
        self.small_thr = DEFAULT_SMALL_THR_MM
        self.med_thr = DEFAULT_MED_THR_MM
        # counters
        self.count_small = 0
        self.count_fit = 0
        self.count_big = 0
        self.count_total = 0
        self.conveyor_state = "STOP"
        # serial
        self.serial_thread = None
        self.serial_ok = False
        self.serial_port_used = ""
        # anti-spam send
        self._last_send_ms = 0
        # UI
        self._build_ui()
        self._wire_events()

        self.start_serial()
        self.set_info("Ready. ESP32 gửi 'SCAN' → PC phân loại → trả S/F/B.")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Ứng dụng xử lý ảnh đo kích thước và phân loại sản phẩm (ESP32)")
        f = QtGui.QFont()
        f.setPointSize(14)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding:10px; background:#e8f3ff; border-radius:10px;")
        root.addWidget(title)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)
        root.addLayout(top)

        style_gb = (
            "QGroupBox{font-weight:bold;}"
            "QGroupBox{border:1px solid #ccc; border-radius:10px; margin-top:8px;}"
            "QGroupBox::title{left:10px; padding:0 4px;}"
        )
        gb_cam = QtWidgets.QGroupBox("Camera / Phone")
        gb_cam.setStyleSheet(style_gb)
        cam = QtWidgets.QGridLayout(gb_cam)
        cam.setContentsMargins(12, 14, 12, 12)
        cam.setHorizontalSpacing(10)
        cam.setVerticalSpacing(8)

        self.btn_cam_on = QtWidgets.QPushButton("On Camera")
        self.btn_cam_off = QtWidgets.QPushButton("Off")
        self.btn_cam_on.setMinimumHeight(40)
        self.btn_cam_off.setMinimumHeight(40)

        self.edt_url = QtWidgets.QLineEdit()
        self.edt_url.setPlaceholderText("Dán URL camera điện thoại (DroidCam/IP Webcam...)")
        self.btn_phone_start = QtWidgets.QPushButton("Start Phone")
        self.btn_phone_start.setMinimumHeight(32)

        cam.addWidget(self.btn_cam_on, 0, 0, 1, 2)
        cam.addWidget(self.btn_cam_off, 0, 2, 1, 1)
        cam.addWidget(QtWidgets.QLabel("Phone URL:"), 1, 0)
        cam.addWidget(self.edt_url, 1, 1, 1, 2)
        cam.addWidget(self.btn_phone_start, 2, 2)

        top.addWidget(gb_cam, 2)

        # ---- Conveyor group ----
        gb_conv = QtWidgets.QGroupBox("Băng tải")
        gb_conv.setStyleSheet(style_gb)
        conv = QtWidgets.QVBoxLayout(gb_conv)
        conv.setContentsMargins(12, 14, 12, 12)
        conv.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        self.btn_conv_start = QtWidgets.QPushButton("START (Chạy băng tải)")
        self.btn_conv_stop = QtWidgets.QPushButton("STOP (Dừng băng tải)")
        self.btn_conv_start.setMinimumHeight(44)
        self.btn_conv_stop.setMinimumHeight(44)
        row.addWidget(self.btn_conv_start)
        row.addWidget(self.btn_conv_stop)

        self.lbl_serial = QtWidgets.QLabel("Serial: ...")
        self.lbl_serial.setStyleSheet("color:#333;")

        self.lbl_conv = QtWidgets.QLabel("Trạng thái: STOP")
        self.lbl_conv.setStyleSheet("font-weight:bold;")

        conv.addLayout(row)
        conv.addWidget(self.lbl_conv)
        conv.addWidget(self.lbl_serial)

        top.addWidget(gb_conv, 2)

        # ---- Counter group ----
        gb_cnt = QtWidgets.QGroupBox("Counter")
        gb_cnt.setStyleSheet(style_gb)
        cnt = QtWidgets.QGridLayout(gb_cnt)
        cnt.setContentsMargins(12, 14, 12, 12)
        cnt.setHorizontalSpacing(12)
        cnt.setVerticalSpacing(8)

        def mk_lcd():
            lcd = QtWidgets.QLCDNumber()
            lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
            lcd.setMinimumHeight(42)
            lcd.setStyleSheet("background:#222; color:#0f0; border-radius:8px; padding:6px;")
            return lcd

        self.lcd_big = mk_lcd()
        self.lcd_fit = mk_lcd()
        self.lcd_small = mk_lcd()
        self.lcd_total = mk_lcd()

        cnt.addWidget(QtWidgets.QLabel("Big"), 0, 0, alignment=Qt.AlignCenter)
        cnt.addWidget(QtWidgets.QLabel("Fit"), 0, 1, alignment=Qt.AlignCenter)
        cnt.addWidget(QtWidgets.QLabel("Small"), 0, 2, alignment=Qt.AlignCenter)
        cnt.addWidget(QtWidgets.QLabel("Total"), 0, 3, alignment=Qt.AlignCenter)

        cnt.addWidget(self.lcd_big, 1, 0)
        cnt.addWidget(self.lcd_fit, 1, 1)
        cnt.addWidget(self.lcd_small, 1, 2)
        cnt.addWidget(self.lcd_total, 1, 3)

        self.btn_reset = QtWidgets.QPushButton("Reset Counter")
        self.btn_reset.setMinimumHeight(36)
        cnt.addWidget(self.btn_reset, 2, 0, 1, 4)

        top.addWidget(gb_cnt, 3)

        # ---- Main: video + snapshot ----
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(10)
        root.addLayout(mid, 1)

        self.video_label = QtWidgets.QLabel("No Signal")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border-radius:12px;")
        self.video_label.setMinimumSize(820, 520)
        mid.addWidget(self.video_label, 3)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        mid.addLayout(right, 2)

        self.snapshot_label = QtWidgets.QLabel("Snapshot")
        self.snapshot_label.setAlignment(Qt.AlignCenter)
        self.snapshot_label.setStyleSheet("background:#000; color:#ddd; border-radius:12px; border:2px solid #bbb;")
        self.snapshot_label.setMinimumSize(520, 360)
        right.addWidget(self.snapshot_label)

        self.info = QtWidgets.QLabel("Info: ...")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("background:#fff4cc; padding:10px; border-radius:10px; border:1px solid #e0d39b;")
        right.addWidget(self.info)

        self.update_lcds()

    def _wire_events(self):
        self.btn_cam_on.clicked.connect(self.start_webcam)
        self.btn_cam_off.clicked.connect(self.stop_all)
        self.btn_phone_start.clicked.connect(self.start_phone)

        self.btn_conv_start.clicked.connect(self.conveyor_start)
        self.btn_conv_stop.clicked.connect(self.conveyor_stop)

        self.btn_reset.clicked.connect(self.reset_counters)

    # ---------- Serial ----------
    def start_serial(self):
        if serial is None:
            self.serial_ok = False
            self.lbl_serial.setText("Serial: pyserial chưa cài (pip install pyserial)")
            return

        port = SERIAL_PORT if SERIAL_PORT else auto_find_arduino_port()
        if not port:
            self.serial_ok = False
            self.lbl_serial.setText("Serial: Không tìm thấy ESP32 (kiểm tra COM)")
            return

        self.serial_port_used = port
        self.serial_thread = SerialReader(port, SERIAL_BAUD, self)
        self.serial_thread.status.connect(self.on_serial_status)
        self.serial_thread.line_received.connect(self.on_serial_line)
        self.serial_thread.start()

        self.lbl_serial.setText(f"Serial: Đang kết nối {port} @ {SERIAL_BAUD} ...")

    def on_serial_status(self, s: str):
        self.serial_ok = s.startswith("OK:")
        self.lbl_serial.setText(f"Serial: {s}")

    def send_to_esp32(self, text: str):
        if not self.serial_thread:
            self.set_info("Serial chưa sẵn sàng.")
            return False
        ok = self.serial_thread.write_text(text)
        if not ok:
            self.set_info("Gửi Serial lỗi. Kiểm tra COM/baud.")
        return ok

    def on_serial_line(self, line: str):
        s = line.strip().upper()
        # ✅ chỉ xử lý khi ESP32 gửi đúng dòng "SCAN"
        if s == "SCAN":
            self.handle_scan_trigger(send_serial=True)

    # ---------- Conveyor buttons ----------
    def conveyor_start(self):
        if self.send_to_esp32("1\n"):
            self.conveyor_state = "RUN"
            self.lbl_conv.setText("Trạng thái: RUN")
            self.set_info("Đã gửi START băng tải (1).")

    def conveyor_stop(self):
        if self.send_to_esp32("0\n"):
            self.conveyor_state = "STOP"
            self.lbl_conv.setText("Trạng thái: STOP")
            self.set_info("Đã gửi STOP băng tải (0).")

    # ---------- Camera ----------
    def start_webcam(self):
        self.release_cap()
        self.cap = open_capture(0)
        if not self.cap:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không mở được camera PC.")
            return
        self.timer.start(30)
        self.set_info("Camera PC đang chạy.")

    def start_phone(self):
        self.release_cap()
        url = self.edt_url.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Nhập URL camera điện thoại.")
            return
        self.cap = open_capture(url)
        if not self.cap:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không mở được stream phone.")
            return
        self.timer.start(30)
        self.set_info(f"Phone stream đang chạy: {url}")

    def stop_all(self):
        self.timer.stop()
        self.release_cap()
        self.video_label.setText("No Signal")
        self.snapshot_label.setText("Snapshot")
        self.set_info("Đã dừng camera/stream.")

    def release_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.last_frame_bgr = None

    # ---------- Frame update ----------
    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.last_frame_bgr = frame
        dets = detect_objects_classic(frame, self.mm_per_pixel, self.small_thr, self.med_thr)

        # vẽ bbox + nhãn
        for d in dets[:5]:
            x, y, w, h = d["bbox"]
            cls = d["cls"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            put_text(frame, f"{cls}", (x, max(15, y - 8)), scale=0.6, thick=2)

        self.show_frame(frame)

    # ---------- SNAPSHOT + SEND ----------
    def handle_scan_trigger(self, send_serial: bool = True):
        now_ms = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        if send_serial and (now_ms - self._last_send_ms < SEND_COOLDOWN_MS):
            return

        if self.last_frame_bgr is None:
            self.set_info("SCAN nhưng chưa có frame.")
            return

        frame = self.last_frame_bgr.copy()
        dets = detect_objects_classic(frame, self.mm_per_pixel, self.small_thr, self.med_thr)

        if not dets:
            self.set_info("SNAPSHOT: Không thấy vật.")
            return

        det = dets[0]
        cls = det["cls"]
        size_mm = float(det.get("size_mm", 0.0))
        x, y, w, h = det["bbox"]

        # crop + padding
        pad = 16
        H, W = frame.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            self.set_info("SNAPSHOT: Crop rỗng.")
            return

        snap = crop.copy()

        # text trên snapshot
        put_text(snap, f"Class: {cls}", (14, 38), scale=1.0, thick=2)
        put_text(snap, f"Size: {size_mm:.2f} mm", (14, 78), scale=0.9, thick=2)
        put_text(snap, f"Thr: Small<={self.small_thr:.2f}mm | Fit<={self.med_thr:.2f}mm",
                 (14, 118), scale=0.65, thick=2)

        self.show_snapshot(snap)

        # map class -> ký tự gửi ESP32
        if cls == "Small":
            cmd = "S\n"
            self.count_small += 1
        elif cls == "Fit":
            cmd = "F\n"
            self.count_fit += 1
        else:
            cmd = "B\n"
            self.count_big += 1

        self.count_total += 1
        self.update_lcds()

        if send_serial:
            ok = self.send_to_esp32(cmd)
            if ok:
                self._last_send_ms = now_ms
            self.set_info(f"SNAPSHOT OK → {cls} | {size_mm:.2f} mm | TX={cmd.strip()} | ok={ok}")
        else:
            self.set_info(f"SNAPSHOT OK → {cls} | {size_mm:.2f} mm | preview-only")

    # ---------- Render ----------
    def show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def show_snapshot(self, snap_bgr):
        snap_rgb = cv2.cvtColor(snap_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = snap_rgb.shape
        qimg = QtGui.QImage(snap_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(self.snapshot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.snapshot_label.setPixmap(pix)
    # ---------- Counters ----------
    def update_lcds(self):
        self.lcd_big.display(int(self.count_big))
        self.lcd_fit.display(int(self.count_fit))
        self.lcd_small.display(int(self.count_small))
        self.lcd_total.display(int(self.count_total))
    def reset_counters(self):
        self.count_small = self.count_fit = self.count_big = self.count_total = 0
        self.update_lcds()
        self.snapshot_label.setText("Snapshot")
        self.set_info("Đã reset counter.")
    def set_info(self, s: str):
        self.info.setText(f"Info: {s}")
    def closeEvent(self, event):
        self.timer.stop()
        self.release_cap()
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait(500)
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
