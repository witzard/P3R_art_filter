// P3R Art Filter: ทำงานในเบราว์เซอร์ทั้งหมด
// - ตัดพื้นหลัง: โมเดล silueta (U2-Net) ผ่าน onnxruntime-web
// - ฟิลเตอร์: OpenCV.js (พอร์ตจาก python/p3r_filter.py)

const MODEL_URL = 'models/silueta.onnx';
const ORT_WASM = new URL('vendor/', location.href).href;
const MAX_FILE_MB = 30;          // กันไฟล์ใหญ่ผิดปกติ
const MAX_PIXELS = 50e6;         // กันภาพที่ขยายแล้วกินแรมมหาศาล (decompression bomb)
const MAX_SIZE = 1400; // ย่อภาพด้านยาวสุดเหลือเท่านี้ ให้ปรับ slider ได้ลื่น

// [key, label, min, max, step, default]
const PARAMS = [
  ['smooth',    'ลบริ้วรอย',                 0,   1,   0.01, 0.45],
  ['levels',    'จำนวนระดับเงา',             2,   10,  1,    5],
  ['posterize', 'ความเป็นก้อนเงา',            0,   1,   0.01, 0.55],
  ['detail',    'รายละเอียดผม/ผ้า',          0,   2,   0.05, 0.9],
  ['ink',       'เส้นขอบ',                   0,   1,   0.01, 0.35],
  ['tint',      'สีฟ้าในเงา',                0,   0.5, 0.01, 0.12],
  ['black',     'จุดดำสุด (กันดำสนิท)',       0,   80,  1,    22],
  ['white',     'จุดขาวสุด (กันขาวโพลน)',     180, 255, 1,    238],
  ['gamma',     'แกมมา (>1 มืดลง)',          0.5, 2,   0.05, 1.0],
  ['shards',    'จำนวนเศษกระจก',             0,   80,  1,    25],
];

const $ = (id) => document.getElementById(id);
let cv = null;
let session = null;
let state = null;   // ข้อมูลของภาพที่อัปโหลด (คำนวณครั้งเดียว)
let seed = 7;
let timer = null;

// ------------------------------------------------------------ UI
function buildSliders() {
  const box = $('sliders');
  for (const [key, label, min, max, step, def] of PARAMS) {
    const row = document.createElement('div');
    row.className = 'row';
    const lab = document.createElement('label');
    const name = document.createElement('span'); name.textContent = label;
    const val = document.createElement('span'); val.id = 'v_' + key; val.textContent = def;
    lab.append(name, val);
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', id: 'p_' + key, min, max, step, value: def });
    row.append(lab, input);
    box.appendChild(row);
    input.addEventListener('input', (e) => {
      $('v_' + key).textContent = e.target.value;
      schedule();
    });
  }
}

function getParams() {
  const p = {};
  for (const [key] of PARAMS) p[key] = parseFloat($('p_' + key).value);
  p.bg = $('bg').value;
  return p;
}

function resetParams() {
  for (const [key, , , , , def] of PARAMS) {
    $('p_' + key).value = def;
    $('v_' + key).textContent = def;
  }
  $('bg').value = '#12acf4';
  schedule();
}

function schedule() {
  if (!state) return;
  clearTimeout(timer);
  timer = setTimeout(render, 120);
}

function setStatus(t) { $('status').textContent = t; }

// ------------------------------------------------------------ helpers
function newF32(w, h, data) {
  const m = new cv.Mat(h, w, cv.CV_32F);
  if (data) m.data32F.set(data);
  return m;
}

function gaussF(arr, w, h, sigma) {
  const m = newF32(w, h, arr), d = new cv.Mat();
  cv.GaussianBlur(m, d, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT);
  const out = d.data32F.slice();
  m.delete(); d.delete();
  return out;
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentileFromHist(hist, total, q) {
  const target = total * q;
  let acc = 0;
  for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc >= target) return i; }
  return 255;
}

// ------------------------------------------------------------ 1. ตัดพื้นหลัง
async function segment(canvas, w, h) {
  const S = 320;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const x = c.getContext('2d');
  x.imageSmoothingQuality = 'high';
  x.drawImage(canvas, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data;

  let mx = 1;
  for (let i = 0; i < d.length; i += 4) mx = Math.max(mx, d[i], d[i + 1], d[i + 2]);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const t = new Float32Array(3 * S * S);
  for (let p = 0; p < S * S; p++)
    for (let ch = 0; ch < 3; ch++)
      t[ch * S * S + p] = (d[p * 4 + ch] / mx - mean[ch]) / std[ch];

  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) });
  const pred = res[session.outputNames[0]].data;
  let lo = Infinity, hi = -Infinity;
  for (const v of pred) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const norm = new Float32Array(S * S);
  for (let i = 0; i < norm.length; i++) norm[i] = (pred[i] - lo) / Math.max(hi - lo, 1e-6);

  const small = newF32(S, S, norm), big = new cv.Mat();
  cv.resize(small, big, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);

  // ทำความสะอาด mask: opening + blur + threshold (เหมือน rembg post_process)
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.morphologyEx(big, big, cv.MORPH_OPEN, k);
  cv.GaussianBlur(big, big, new cv.Size(0, 0), 2, 2);
  const mask = big.data32F.slice();
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i] > 0.5 ? 1 : 0;
  small.delete(); big.delete(); k.delete();
  return mask;
}

// ลบพื้นหลังเดิมที่ติดอยู่ตามช่องผม/ขอบตัว (เช็คเฉพาะแถบขอบ)
function refineEdges(rgb, mask, w, h, bandPx = 30) {
  const lab = new cv.Mat();
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
  const L = lab.data;
  const hist = [new Array(256).fill(0), new Array(256).fill(0), new Array(256).fill(0)];
  let n = 0;
  for (let i = 0; i < w * h; i++) {
    if (mask[i] < 0.05) { hist[0][L[i*3]]++; hist[1][L[i*3+1]]++; hist[2][L[i*3+2]]++; n++; }
  }
  if (n < 300) { lab.delete(); return mask; }
  const ref = [0, 1, 2].map((c) => percentileFromHist(hist[c], n, 0.5));

  const inside = new cv.Mat(h, w, cv.CV_8U);
  for (let i = 0; i < w * h; i++) inside.data[i] = mask[i] >= 0.5 ? 1 : 0;
  const dist = new cv.Mat();
  cv.distanceTransform(inside, dist, cv.DIST_L2, 5);
  const D = dist.data32F;

  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const dl = L[i*3] - ref[0], da = L[i*3+1] - ref[1], db = L[i*3+2] - ref[2];
    const key = Math.min(1, Math.max(0, (Math.sqrt(dl*dl + da*da + db*db) - 10) / 18));
    const near = Math.min(1, Math.max(0, 1 - D[i] / bandPx));
    out[i] = mask[i] * (1 - near * (1 - key));
  }
  lab.delete(); inside.delete(); dist.delete();
  return gaussF(out, w, h, 1.0);
}

// ------------------------------------------------------------ เตรียมข้อมูลตอนอัปโหลด
async function prepare(file) {
  if (!file.type.startsWith('image/')) throw new Error('ไฟล์นี้ไม่ใช่รูปภาพ');
  if (file.size > MAX_FILE_MB * 1024 * 1024) throw new Error(`ไฟล์ใหญ่เกิน ${MAX_FILE_MB} MB`);
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  if (bmp.width * bmp.height > MAX_PIXELS) { bmp.close(); throw new Error('ภาพความละเอียดสูงเกินไป'); }
  const s = Math.min(1, MAX_SIZE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, h);

  const srcRGBA = cv.matFromImageData(rgba);
  const rgb = new cv.Mat();
  cv.cvtColor(srcRGBA, rgb, cv.COLOR_RGBA2RGB);
  srcRGBA.delete();

  setStatus('กำลังตัดพื้นหลัง...');
  await new Promise((r) => setTimeout(r, 20));
  let mask = await segment(c, w, h);
  mask = refineEdges(rgb, mask, w, h);

  // แถบขอบสำหรับ inpaint (กันขอบขาว)
  const band = new cv.Mat(h, w, cv.CV_8U);
  const inner = new cv.Mat(h, w, cv.CV_8U);
  for (let i = 0; i < w * h; i++) {
    band.data[i] = mask[i] > 0.02 && mask[i] <= 0.95 ? 255 : 0;
    inner.data[i] = mask[i] > 0.95 ? 255 : 0;
  }
  const k5 = cv.Mat.ones(5, 5, cv.CV_8U);
  cv.dilate(band, band, k5);
  for (let i = 0; i < w * h; i++) if (inner.data[i]) band.data[i] = 0;
  inner.delete();

  // mask สุดท้ายสำหรับรวมภาพ: หดเข้านิด + ขอบนุ่ม
  const mm = newF32(w, h, mask), me = new cv.Mat();
  cv.erode(mm, me, k5);
  cv.GaussianBlur(me, me, new cv.Size(0, 0), 1.0, 1.0);
  const maskE = me.data32F.slice();
  mm.delete(); me.delete(); k5.delete();

  // ค่าที่ไม่ขึ้นกับ slider: blur ของภาพสี และ high-pass ของ gray ต้นฉบับ
  const rgbF = new cv.Mat(); rgb.convertTo(rgbF, cv.CV_32FC3);
  const rgbBlur = new cv.Mat();
  cv.GaussianBlur(rgbF, rgbBlur, new cv.Size(0, 0), 1.5, 1.5);
  const g0m = new cv.Mat(); cv.cvtColor(rgb, g0m, cv.COLOR_RGB2GRAY);
  const g0 = Float32Array.from(g0m.data);
  const g0b = gaussF(g0, w, h, 2.5);
  const hp = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) hp[i] = g0[i] - g0b[i];

  // b1/b2 bilateral ของภาพสีไม่ขึ้นกับ slider เช่นกัน
  const b1 = new cv.Mat(), b2 = new cv.Mat();
  cv.bilateralFilter(rgb, b1, 9, 45, 9);
  cv.bilateralFilter(b1, b2, 7, 30, 7);

  const st = {
    w, h, rgba, rgb, mask, maskE, band, hp,
    rgbF: rgbF.data32F.slice(), rgbBlur: rgbBlur.data32F.slice(), smooth: b2.data.slice(),
    bgCache: null,
  };
  rgbF.delete(); rgbBlur.delete(); g0m.delete(); b1.delete(); b2.delete();
  return st;
}

// ------------------------------------------------------------ พื้นหลัง
function makeBackground(w, h, hex, shards, sd) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = hex;
  ctx.fillRect(0, 0, w, h);
  // vignette อ่อน ๆ
  const img = ctx.getImageData(0, 0, w, h), d = img.data;
  for (let y = 0; y < h; y++) {
    const dy = (y - h * 0.4) / h;
    for (let x = 0; x < w; x++) {
      const dx = (x - w / 2) / w;
      const f = 1.06 - 0.18 * Math.sqrt(dx * dx + dy * dy);
      const i = (y * w + x) * 4;
      d[i] = Math.min(255, d[i] * f); d[i+1] = Math.min(255, d[i+1] * f); d[i+2] = Math.min(255, d[i+2] * f);
    }
  }
  ctx.putImageData(img, 0, 0);
  // เศษกระจก
  const rng = mulberry32(sd);
  const s = Math.min(w, h);
  ctx.filter = 'blur(0.7px)';
  for (let k = 0; k < shards; k++) {
    const cx = rng() * w, cy = rng() * h;
    const r = (0.008 + rng() * 0.027) * s;
    const n = 3 + Math.floor(rng() * 3);
    const ang = Array.from({ length: n }, () => rng() * Math.PI * 2).sort((a, b) => a - b);
    const tg = 230 + rng() * 25, tb = 140 + rng() * 70, a = 0.25 + rng() * 0.35;
    ctx.fillStyle = `rgba(${tb | 0},${tg | 0},255,${a.toFixed(2)})`;
    ctx.beginPath();
    ang.forEach((t, j) => {
      const rr = r * (0.5 + rng() * 0.8);
      const px = cx + rr * Math.cos(t), py = cy + rr * Math.sin(t);
      j ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
  }
  ctx.filter = 'none';
  return ctx.getImageData(0, 0, w, h).data;
}

// ------------------------------------------------------------ ฟิลเตอร์หลัก
function render() {
  const t0 = performance.now();
  const p = getParams();
  const { w, h } = state;
  const N = w * h;

  if ($('showOrig').checked) {
    const out = $('out'); out.width = w; out.height = h;
    out.getContext('2d').putImageData(state.rgba, 0, 0);
    return;
  }

  // 2. ลบริ้วรอยบางส่วน + คืน texture ผิว
  const s = p.smooth;
  const soft = new cv.Mat(h, w, cv.CV_8UC3);
  const SD = soft.data, F = state.rgbF, B = state.rgbBlur, SM = state.smooth;
  for (let i = 0; i < N * 3; i++) {
    const v = F[i] * (1 - s) + SM[i] * s + (F[i] - B[i]) * 0.35 * (1 - s);
    SD[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  const gray = new cv.Mat();
  cv.cvtColor(soft, gray, cv.COLOR_RGB2GRAY);
  soft.delete();
  cv.inpaint(gray, gray, state.band, 5, cv.INPAINT_TELEA);

  // 3. โทนแสงเงา
  const clahe = new cv.CLAHE(1.6, new cv.Size(8, 8));
  const cl = new cv.Mat();
  clahe.apply(gray, cl);
  clahe.delete();
  const hist = new Array(256).fill(0);
  let n = 0;
  for (let i = 0; i < N; i++) if (state.mask[i] > 0.5) { hist[cl.data[i]]++; n++; }
  let lo = 0, hi = 255;
  if (n > 100) { lo = percentileFromHist(hist, n, 0.02); hi = percentileFromHist(hist, n, 0.98); }
  const g = new Float32Array(N);
  const g8 = new cv.Mat(h, w, cv.CV_8U);
  for (let i = 0; i < N; i++) {
    let v = (cl.data[i] - lo) / Math.max(hi - lo, 1);
    v = Math.pow(v < 0 ? 0 : v > 1 ? 1 : v, p.gamma);
    g[i] = v;
    g8.data[i] = Math.round(v * 255);
  }
  cl.delete();
  const gsm = new cv.Mat();
  cv.bilateralFilter(g8, gsm, 9, 30, 9);
  g8.delete();
  const gs = gaussF(Float32Array.from(gsm.data), w, h, 1.2);
  gsm.delete();

  const steps = p.levels - 1;
  const tone = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = (gs[i] / 255) * steps;
    const base = Math.floor(x);
    let t = Math.min(1, Math.max(0, (x - base - 0.35) / 0.3));
    t = t * t * (3 - 2 * t);
    const q = (base + t) / steps;
    const o = q * p.posterize + g[i] * (1 - p.posterize);
    tone[i] = p.black + o * (p.white - p.black);
  }

  // 4. เส้นขอบบาง ๆ
  if (p.ink > 0) {
    const md = new cv.Mat(), th = new cv.Mat();
    cv.medianBlur(gray, md, 5);
    cv.adaptiveThreshold(md, th, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 9);
    const e = gaussF(Float32Array.from(th.data), w, h, 0.8);
    for (let i = 0; i < N; i++) tone[i] *= 1 - (1 - e[i] / 255) * p.ink;
    md.delete(); th.delete();
  }
  gray.delete();

  // 5. รายละเอียดผม/ผ้า
  const lo2 = p.black * 0.6;
  for (let i = 0; i < N; i++) {
    const v = tone[i] + state.hp[i] * p.detail;
    tone[i] = v < lo2 ? lo2 : v > p.white ? p.white : v;
  }

  // 6. พื้นหลัง (cache ไว้ถ้าค่าไม่เปลี่ยน)
  const bgKey = `${p.bg}|${p.shards}|${seed}`;
  if (!state.bgCache || state.bgCache.key !== bgKey)
    state.bgCache = { key: bgKey, data: makeBackground(w, h, p.bg, p.shards, seed) };
  const bg = state.bgCache.data;

  // 7. ใส่สีฟ้าในเงา แล้วรวมภาพ
  const blue = hexToRgb(p.bg);
  const out = new ImageData(w, h);
  const O = out.data, M = state.maskE;
  for (let i = 0; i < N; i++) {
    const tv = tone[i], sw = 1 - tv / 255, m = M[i];
    for (let c = 0; c < 3; c++) {
      const pv = tv * (1 - p.tint * sw) + blue[c] * 0.35 * p.tint * sw;
      O[i * 4 + c] = pv * m + bg[i * 4 + c] * (1 - m);
    }
    O[i * 4 + 3] = 255;
  }
  const cvs = $('out');
  cvs.width = w; cvs.height = h;
  cvs.getContext('2d').putImageData(out, 0, 0);
  setStatus(`เสร็จ (${w}×${h}, ${Math.round(performance.now() - t0)} ms)`);
}

// ------------------------------------------------------------ เริ่มต้น
async function loadCV() {
  let c = window.cv;
  if (c instanceof Promise) c = await c;
  if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
  return c;
}

async function init() {
  buildSliders();
  $('reset').onclick = resetParams;
  $('reseed').onclick = () => { seed = (Math.random() * 1e9) | 0; schedule(); };
  $('bg').addEventListener('input', schedule);
  $('showOrig').addEventListener('change', schedule);
  $('download').onclick = () => {
    $('out').toBlob((b) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'p3r_filter.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
  };
  $('file').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      setStatus('กำลังเตรียมภาพ...');
      if (state) { state.rgb.delete(); state.band.delete(); state = null; }
      $('download').disabled = true;
      state = await prepare(f);
      render();
      $('download').disabled = false;
    } catch (err) {
      console.error(err);
      setStatus('เกิดข้อผิดพลาด: ' + err.message);
    }
  });

  try {
    cv = await loadCV();
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    $('file').disabled = false;
    setStatus('พร้อมแล้ว เลือกรูปเพื่อเริ่ม');
  } catch (err) {
    console.error(err);
    setStatus('โหลดไม่สำเร็จ: ' + err.message);
  }
}

init();
