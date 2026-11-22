# P3R Art Filter

เว็บแอปแปลงรูปคนให้เป็นโทนขาวดำหลายระดับบนพื้นหลังฟ้าสด (แรงบันดาลใจจาก Persona 3 Reload)
ภาพยังเป็นคนสมจริง ทำงานในเบราว์เซอร์ทั้งหมด รูปไม่ถูกส่งขึ้นเซิร์ฟเวอร์

- ตัดพื้นหลัง: โมเดล `silueta` (U2-Net ขนาดย่อ, จาก [rembg](https://github.com/danielgatis/rembg)) ผ่าน onnxruntime-web
- ฟิลเตอร์: OpenCV.js (ลบริ้วรอยบางส่วน, CLAHE, posterize แบบนุ่ม, เส้นขอบ, ใส่สีฟ้าในเงา)

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `index.html` | หน้าเว็บ: ช่องอัปโหลด, แถบเลื่อนปรับค่า, ปุ่มดาวน์โหลด |
| `app.js` | หน้าเว็บ: slider (+/−), แปรงระบายเพิ่ม/ลบพื้นหลัง, undo, ซูม, คุยกับ worker |
| `worker.js` | ประมวลผลทั้งหมดใน Web Worker (ตัดพื้นหลัง + ฟิลเตอร์) หน้าเว็บจึงไม่ค้าง |
| `models/silueta.onnx` | โมเดลตัดพื้นหลัง (~43 MB) |
| `models/face_parsing_resnet18.onnx` | โมเดลแยกผม/ผิว/ตา/คิ้ว/ปาก (BiSeNet, ~53 MB) จาก [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) เทรนบน CelebAMask-HQ ซึ่งอนุญาตเฉพาะงานที่ไม่ใช่เชิงพาณิชย์ |
| `vendor/` | onnxruntime-web 1.31.1 (MIT) และ OpenCV.js 5.0 (Apache-2.0) เก็บไว้ใน repo เอง |
| `vercel.json` | security headers (CSP ฯลฯ) และ cache |
| `python/p3r_filter.py` | เวอร์ชัน Python (CLI) รุ่นเก่า (ยังไม่มี cartoon pipeline) |

## Pipeline (worker.js)

1. ตัดพื้นหลัง (U2-Net silueta) + edge cleanup (Lab, distance transform, connected components) + แปรงแก้ mask
2. หาหัวจาก silhouette (จุดที่คอแคบลง) แล้วรัน face parsing บนภาพหัว → แผนที่ผม/ผิว/ตา-คิ้ว-ปาก/ช่องปาก ถ้าไม่เจอหน้าจะไม่ใช้การแยกบริเวณ (fallback)
3. Cartoon simplification: bilateral หลายรอบ + median
4. ผม: median + closing/opening รวมเส้นเป็นก้อน, ไฮไลต์ถูกกดเป็นโทนกลาง
5. หน้า: bilateral เพิ่มบนผิวยกเว้นตา/คิ้ว/ปาก, ฟัน: closing เฉพาะในช่องปากเพื่อลบร่องระหว่างฟัน
6. Cel shading แยกตามบริเวณ (quantize → median merge → ลบ region เล็กตามสัดส่วนพื้นที่ภาพ → palette), เงาผิวไม่ต่ำกว่าเทากลาง
7. คืนรายละเอียดเฉพาะบริเวณ (ผิว ~0, ผ้าปานกลาง, ผมเฉพาะโครงหยาบ)
8. เส้น 3 ระดับ: silhouette (ขอบตัว + ขอบผมกับหน้า), structural (Canny ครึ่งความละเอียด + กรองความยาวตามบริเวณ + black-hat สำหรับแว่น/เปลือกตา), fine detail (ถูกกด)
9. สี: Grayscale หรือ P3R blue palette (navy → royal → cyan → cool white ตามความสว่าง, ผิวซีดกว่า)

## ความปลอดภัย

- รูปถูกประมวลผลในเบราว์เซอร์ทั้งหมด ไม่ส่งขึ้นเซิร์ฟเวอร์ ไฟล์ที่ดาวน์โหลดไม่มี EXIF/GPS ติดไป
- ไม่โหลดโค้ดจากเว็บอื่น (ไลบรารีอยู่ใน `vendor/`) และ CSP อนุญาตเฉพาะ `'self'`
- CSP ต้องมี `'unsafe-eval'` เพราะ OpenCV.js (Emscripten embind) สร้างฟังก์ชันด้วย `new Function`
- จำกัดไฟล์ไม่เกิน 30 MB และ 50 ล้านพิกเซล

## ความเร็ว

- พรีวิวขณะปรับค่าใช้ภาพ 1000px, ปุ่มดาวน์โหลดสร้างภาพใหม่ที่ 2000px (ค่าพิกเซลทุกตัวปรับตามขนาด หน้าตาจึงเหมือนกัน)
- หลังระบายแต่ละเส้นจะรวมภาพใหม่ทันที (~30 ms) แล้วค่อยคำนวณโทนทั้งหมดเมื่อหยุดระบาย 0.7 วินาที

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
