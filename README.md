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
2. หาหัวจาก silhouette (จุดที่คอแคบลง) แล้วรัน face parsing บนภาพหัว (รวมคอ) → ผม/ผิว/หน้า/คอ/จมูก/ตา/คิ้ว/ริมฝีปาก/ช่องปาก; ขอบ crop ค่อย ๆ จาง; ไม่เจอหน้า = ไม่แยกบริเวณ (fallback)
3. Cartoon simplification: bilateral หลายรอบ + median; ผม/ผิวเกลี่ยเพิ่มเฉพาะในกรอบของบริเวณนั้น
4. แถบขอบ silhouette แทนด้วยค่าเฉลี่ยเนื้อด้านใน (normalized convolution) กัน halo
5. **Nested tone masks แยกตามบริเวณ** (`toneRegion`): M_k = v ≥ t_k → closing/opening → ลบเกาะและรูเล็ก (ตามสัดส่วนพื้นที่ภาพ) → blur+threshold ให้ขอบเรียบ (anti-aliased) → บังคับซ้อนกัน; ผม ผิว เสื้อผ้า คำนวณแยกกัน ไม่รวมข้ามขอบ
   - ผม: เกณฑ์จาก percentile ความสว่างของผมเอง, สว่างสุดแค่โทนกลาง, ขอบนอกผมใช้ palette ผม
   - หน้า: normalize ความสว่างผิว, ไม่เกิน 3 ระดับ, เงาไม่ต่ำกว่าเทากลาง, ลบเงาเล็กบนแก้ม
   - คิ้ว = ก้อนเข้มก้อนเดียว, ริมฝีปาก = โทนเรียบ, ฟัน = ก้อนสว่างก้อนเดียวในช่องปากสีเข้ม
6. คืนรายละเอียดเฉพาะผ้า (เล็กน้อย) และโครงหยาบของผม
7. เส้นตามลำดับความสำคัญ: silhouette > ตา/ปาก/กราม/ใต้จมูก/แว่น (black-hat รอบตา หรือทั้งหน้าถ้าหาตาไม่เจอ) > ขอบก้อนโทนผม > รอยพับผ้า (Canny ยาว + approxPolyDP, วาดแบบ anti-aliased) > ผิว (ไม่มี)
8. สี: Grayscale หรือ P3R blue palette (navy → royal → cyan → cool white ตามความสว่าง, ผิวซีดกว่า)

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
