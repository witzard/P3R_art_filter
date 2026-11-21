# P3R Art Filter

เว็บแอปแปลงรูปคนให้เป็นโทนขาวดำหลายระดับบนพื้นหลังฟ้าสด (แรงบันดาลใจจาก Persona 3 Reload)
ภาพยังเป็นคนสมจริง ทำงานในเบราว์เซอร์ทั้งหมด รูปไม่ถูกส่งขึ้นเซิร์ฟเวอร์

- ตัดพื้นหลัง: โมเดล `silueta` (U2-Net ขนาดย่อ, จาก [rembg](https://github.com/danielgatis/rembg)) ผ่าน onnxruntime-web
- ฟิลเตอร์: OpenCV.js (ลบริ้วรอยบางส่วน, CLAHE, posterize แบบนุ่ม, เส้นขอบ, ใส่สีฟ้าในเงา)

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `index.html` | หน้าเว็บ: ช่องอัปโหลด, แถบเลื่อนปรับค่า, ปุ่มดาวน์โหลด |
| `app.js` | ตัดพื้นหลัง + ฟิลเตอร์ทั้งหมด |
| `models/silueta.onnx` | โมเดลตัดพื้นหลัง (~43 MB) |
| `vendor/` | onnxruntime-web 1.31.1 (MIT) และ OpenCV.js 5.0 (Apache-2.0) เก็บไว้ใน repo เอง |
| `vercel.json` | security headers (CSP ฯลฯ) และ cache |
| `python/p3r_filter.py` | เวอร์ชัน Python (CLI) ของฟิลเตอร์เดียวกัน |

## ความปลอดภัย

- รูปถูกประมวลผลในเบราว์เซอร์ทั้งหมด ไม่ส่งขึ้นเซิร์ฟเวอร์ ไฟล์ที่ดาวน์โหลดไม่มี EXIF/GPS ติดไป
- ไม่โหลดโค้ดจากเว็บอื่น (ไลบรารีอยู่ใน `vendor/`) และ CSP อนุญาตเฉพาะ `'self'`
- CSP ต้องมี `'unsafe-eval'` เพราะ OpenCV.js (Emscripten embind) สร้างฟังก์ชันด้วย `new Function`
- จำกัดไฟล์ไม่เกิน 30 MB และ 50 ล้านพิกเซล

## รันในเครื่อง

```bash
python -m http.server 8000
# เปิด http://localhost:8000
```

## Deploy บน Vercel

เป็นเว็บ static ไม่ต้อง build: Import repo นี้ใน Vercel แล้วเลือก Framework Preset = **Other** แล้วกด Deploy

## เวอร์ชัน Python

```bash
pip install opencv-python numpy rembg onnxruntime
python python/p3r_filter.py input.jpg output.png --levels 5
```
