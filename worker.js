// Web Worker: ประมวลผลทั้งหมดแยกจากหน้าเว็บ ไม่ให้หน้าเว็บค้าง
// - ตัดพื้นหลัง: โมเดล silueta (U2-Net) ผ่าน onnxruntime-web
// - ฟิลเตอร์: OpenCV.js (พอร์ตจาก python/p3r_filter.py)
// ค่าที่เป็น "พิกเซล" ทั้งหมดอ้างอิงภาพขนาด 1400px แล้วปรับตามขนาดจริง (st.f)
// ภาพพรีวิวกับภาพดาวน์โหลดจึงหน้าตาเหมือนกัน

importScripts('vendor/ort.wasm.min.js', 'vendor/opencv.js');

const BASE = 1400;
let cv = null;
let session = null;
let src = null;      // ImageBitmap ความละเอียดสูงสุด (สำหรับดาวน์โหลด)
let pred = null;     // ผลโมเดล 320x320 (ใช้สร้าง mask ได้ทุกขนาด)
let preview = null;  // state ของภาพพรีวิว

// ------------------------------------------------------------ helpers
const odd = (n) => Math.max(3, Math.round(n) | 1);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

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

function resizeF(arr, sw, sh, w, h) {
  const a = newF32(sw, sh, arr), b = new cv.Mat();
  cv.resize(a, b, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
  const out = b.data32F.slice();
  a.delete(); b.delete();
  return out;
}

// ------------------------------------------------------------ 1. ตัดพื้นหลัง
async function runModel(bitmap) {
  const S = 320;
  const c = new OffscreenCanvas(S, S);
  const x = c.getContext('2d', { willReadFrequently: true });
  x.imageSmoothingQuality = 'high';
  x.drawImage(bitmap, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data;
  let mx = 1;
  for (let i = 0; i < d.length; i += 4) mx = Math.max(mx, d[i], d[i + 1], d[i + 2]);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const t = new Float32Array(3 * S * S);
  for (let p = 0; p < S * S; p++)
    for (let ch = 0; ch < 3; ch++)
      t[ch * S * S + p] = (d[p * 4 + ch] / mx - mean[ch]) / std[ch];
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) });
  const out = res[session.outputNames[0]].data;
  let lo = Infinity, hi = -Infinity;
  for (const v of out) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const norm = new Float32Array(S * S);
  for (let i = 0; i < norm.length; i++) norm[i] = (out[i] - lo) / Math.max(hi - lo, 1e-6);
  return norm;
}

// mask ของโมเดลที่ขนาด w×h + ทำความสะอาด (opening + blur + threshold)
function rawMask(w, h, f) {
  const big = newF32(w, h, resizeF(pred, 320, 320, w, h));
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(odd(3 * f), odd(3 * f)));
  cv.morphologyEx(big, big, cv.MORPH_OPEN, k);
  cv.GaussianBlur(big, big, new cv.Size(0, 0), 2 * f, 2 * f);
  const m = big.data32F.slice();
  for (let i = 0; i < m.length; i++) m[i] = m[i] > 0.5 ? 1 : 0;
  big.delete(); k.delete();
  return m;
}

// ข้อมูลสำหรับ refine ที่ไม่ขึ้นกับ slider
function edgeData(rgb, raw, w, h) {
  const lab = new cv.Mat();
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
  const L = lab.data, N = w * h;
  const hist = [new Array(256).fill(0), new Array(256).fill(0), new Array(256).fill(0)];
  let n = 0;
  for (let i = 0; i < N; i++)
    if (raw[i] < 0.05) { hist[0][L[i*3]]++; hist[1][L[i*3+1]]++; hist[2][L[i*3+2]]++; n++; }
  if (n < 300) { lab.delete(); return null; }
  const ref = [0, 1, 2].map((c) => percentileFromHist(hist[c], n, 0.5));
  const wallDist = new Float32Array(N), dark = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const dl = L[i*3] - ref[0], da = L[i*3+1] - ref[1], db = L[i*3+2] - ref[2];
    wallDist[i] = Math.sqrt(dl*dl + da*da + db*db);
    dark[i] = L[i*3] < ref[0] * 0.55 ? 1 : 0;
  }
  lab.delete();
  const inside = new cv.Mat(h, w, cv.CV_8U);
  for (let i = 0; i < N; i++) inside.data[i] = raw[i] >= 0.5 ? 1 : 0;
  const dist = new cv.Mat();
  cv.distanceTransform(inside, dist, cv.DIST_L2, 5);
  const edgeDist = dist.data32F.slice();
  inside.delete(); dist.delete();
  return { wallDist, dark, edgeDist };
}

// ลบพื้นหลังเดิมที่ติดอยู่ตามช่องผม/ขอบตัว
// - ลบเฉพาะส่วน "บาง" ที่สีเหมือนผนัง (ช่องผม, halo)
// - ส่วนหนาที่สีไม่ใช่ผนังแท้ (จมูก ริมฝีปาก) และจุดไฮไลต์ที่ไม่ต่อกับพื้นหลัง จะไม่ถูกลบ
function refineEdges(st, bandPx, protectPx) {
  const { w, h, raw, edge, f } = st, N = w * h;
  if (!edge || bandPx <= 0) return raw.slice();
  const { wallDist, dark, edgeDist } = edge;
  const removed = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const key = clamp01((wallDist[i] - 10) / 18);
    const near = clamp01(1 - edgeDist[i] / bandPx);
    removed[i] = near * (1 - key);
  }
  const region = new cv.Mat(h, w, cv.CV_8U);
  const setRegion = () => { for (let i = 0; i < N; i++) region.data[i] = removed[i] > 0.3 && raw[i] > 0.5 ? 1 : 0; };
  const k3 = cv.Mat.ones(3, 3, cv.CV_8U);

  if (protectPx > 0) {
    setRegion();
    const r = Math.max(1, Math.round(protectPx));
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * r + 1, 2 * r + 1));
    const thick = new cv.Mat();
    cv.morphologyEx(region, thick, cv.MORPH_OPEN, k);
    cv.dilate(thick, thick, k3);
    const tf = gaussF(Float32Array.from(thick.data), w, h, 1.5 * f);
    for (let i = 0; i < N; i++) {
      const notWall = clamp01((wallDist[i] - 8) / 8);
      removed[i] *= 1 - Math.min(1, tf[i] * 1.5) * notWall;
    }
    k.delete(); thick.delete();
  }

  setRegion();
  const labels = new cv.Mat();
  const nLab = cv.connectedComponents(region, labels, 8, cv.CV_32S);
  if (nLab > 1) {
    const seedOut = new cv.Mat(h, w, cv.CV_8U), seedDark = new cv.Mat(h, w, cv.CV_8U);
    for (let i = 0; i < N; i++) { seedOut.data[i] = raw[i] < 0.5 ? 1 : 0; seedDark.data[i] = dark[i]; }
    const k5 = cv.Mat.ones(odd(5 * f), odd(5 * f), cv.CV_8U), k7 = cv.Mat.ones(odd(7 * f), odd(7 * f), cv.CV_8U);
    cv.dilate(seedOut, seedOut, k5);
    cv.dilate(seedDark, seedDark, k7);
    const ok = new Uint8Array(nLab); ok[0] = 1;
    const lb = labels.data32S;
    for (let i = 0; i < N; i++) if (region.data[i] && (seedOut.data[i] || seedDark.data[i])) ok[lb[i]] = 1;
    const bad = new cv.Mat(h, w, cv.CV_8U);
    for (let i = 0; i < N; i++) bad.data[i] = ok[lb[i]] ? 0 : 1;
    cv.dilate(bad, bad, k3);
    const bf = gaussF(Float32Array.from(bad.data), w, h, 1.0 * f);
    for (let i = 0; i < N; i++) removed[i] *= 1 - Math.min(1, bf[i]);
    seedOut.delete(); seedDark.delete(); k5.delete(); k7.delete(); bad.delete();
  }
  labels.delete(); region.delete(); k3.delete();

  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = raw[i] * (1 - removed[i]);
  return out;
}

// ใส่ส่วนที่ผู้ใช้ระบายเอง: edits (Int8, -127..127) บวก = เพิ่มตัวคน, ลบ = ลบออก
function applyEdits(mask, edits) {
  if (!edits) return mask;
  const out = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const e = edits[i] / 127, m = mask[i];
    out[i] = e > 0 ? m + (1 - m) * e : e < 0 ? m * (1 + e) : m;
  }
  return out;
}

// mask สำหรับรวมภาพ: หดเข้านิด + ขอบนุ่ม
function compositeMask(st, mask) {
  const k = cv.Mat.ones(odd(5 * st.f), odd(5 * st.f), cv.CV_8U);
  const mm = newF32(st.w, st.h, mask), me = new cv.Mat();
  cv.erode(mm, me, k);
  cv.GaussianBlur(me, me, new cv.Size(0, 0), 1.0 * st.f, 1.0 * st.f);
  const out = me.data32F.slice();
  mm.delete(); me.delete(); k.delete();
  return out;
}

// mask สุดท้าย + แถบขอบสำหรับ inpaint (คำนวณใหม่เฉพาะเมื่อค่าขอบหรือการระบายเปลี่ยน)
function buildMask(st, p, edits, editsVer) {
  const key = `${p.band}|${p.protect}|${editsVer}`;
  if (st.maskKey === key) return;
  const { w, h, f } = st, N = w * h;
  if (!st.refined || st.refinedKey !== `${p.band}|${p.protect}`) {
    st.refined = gaussF(refineEdges(st, p.band * f, p.protect * f), w, h, 1.0 * f);
    st.refinedKey = `${p.band}|${p.protect}`;
  }
  const mask = applyEdits(st.refined, edits);

  const band = new cv.Mat(h, w, cv.CV_8U);
  const inner = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    band.data[i] = mask[i] > 0.02 && mask[i] <= 0.95 ? 255 : 0;
    inner[i] = mask[i] > 0.95 ? 1 : 0;
  }
  const k5 = cv.Mat.ones(odd(5 * f), odd(5 * f), cv.CV_8U);
  cv.dilate(band, band, k5);
  k5.delete();
  for (let i = 0; i < N; i++) if (inner[i]) band.data[i] = 0;

  if (st.band) st.band.delete();
  Object.assign(st, { mask, maskE: compositeMask(st, mask), band, maskKey: key });
}

// ------------------------------------------------------------ เตรียม state ที่ขนาดหนึ่ง
function prepare(bitmap, maxSize) {
  const s = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * s), h = Math.round(bitmap.height * s);
  const f = Math.max(w, h) / BASE;
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h);

  const srcRGBA = cv.matFromImageData(rgba);
  const rgb = new cv.Mat();
  cv.cvtColor(srcRGBA, rgb, cv.COLOR_RGBA2RGB);
  srcRGBA.delete();

  const raw = rawMask(w, h, f);
  const edge = edgeData(rgb, raw, w, h);

  const rgbF = new cv.Mat(); rgb.convertTo(rgbF, cv.CV_32FC3);
  const rgbBlur = new cv.Mat();
  cv.GaussianBlur(rgbF, rgbBlur, new cv.Size(0, 0), 1.5 * f, 1.5 * f);
  const g0m = new cv.Mat(); cv.cvtColor(rgb, g0m, cv.COLOR_RGB2GRAY);
  const g0 = Float32Array.from(g0m.data);
  const g0b = gaussF(g0, w, h, 2.5 * f);
  const hp = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) hp[i] = g0[i] - g0b[i];

  const b1 = new cv.Mat(), b2 = new cv.Mat();
  cv.bilateralFilter(rgb, b1, Math.round(9 * f) || 1, 45, 9 * f);
  cv.bilateralFilter(b1, b2, Math.round(7 * f) || 1, 30, 7 * f);

  const st = {
    w, h, f, raw, edge, hp, band: null, maskKey: null, refined: null, refinedKey: null,
    rgbF: rgbF.data32F.slice(), rgbBlur: rgbBlur.data32F.slice(), smooth: b2.data.slice(),
    bgCache: null, person: null,
  };
  rgb.delete(); rgbF.delete(); rgbBlur.delete(); g0m.delete(); b1.delete(); b2.delete();
  return st;
}

function freeState(st) {
  if (st && st.band) st.band.delete();
}

// ------------------------------------------------------------ พื้นหลัง
function makeBackground(w, h, hex, shards, sd) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = hex;
  ctx.fillRect(0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h), d = img.data;
  for (let y = 0; y < h; y++) {
    const dy = (y - h * 0.4) / h;
    for (let x = 0; x < w; x++) {
      const dx = (x - w / 2) / w;
      const k = 1.06 - 0.18 * Math.sqrt(dx * dx + dy * dy);
      const i = (y * w + x) * 4;
      d[i] = Math.min(255, d[i] * k); d[i+1] = Math.min(255, d[i+1] * k); d[i+2] = Math.min(255, d[i+2] * k);
    }
  }
  ctx.putImageData(img, 0, 0);
  const rng = mulberry32(sd);
  const s = Math.min(w, h);
  ctx.filter = `blur(${(0.7 * Math.max(w, h) / BASE).toFixed(2)}px)`;
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

// ------------------------------------------------------------ ชั้นตัวคน (โทนแสงเงา)
function renderPerson(st, p) {
  const { w, h, f } = st, N = w * h;

  // ลบริ้วรอยบางส่วน + คืน texture ผิว
  const s = p.smooth;
  const soft = new cv.Mat(h, w, cv.CV_8UC3);
  const SD = soft.data, F = st.rgbF, B = st.rgbBlur, SM = st.smooth;
  for (let i = 0; i < N * 3; i++) {
    const v = F[i] * (1 - s) + SM[i] * s + (F[i] - B[i]) * 0.35 * (1 - s);
    SD[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  const gray = new cv.Mat();
  cv.cvtColor(soft, gray, cv.COLOR_RGB2GRAY);
  soft.delete();
  cv.inpaint(gray, gray, st.band, 5 * f, cv.INPAINT_TELEA);

  // โทนแสงเงา
  const clahe = new cv.CLAHE(1.6, new cv.Size(8, 8));
  const cl = new cv.Mat();
  clahe.apply(gray, cl);
  clahe.delete();
  const hist = new Array(256).fill(0);
  let n = 0;
  for (let i = 0; i < N; i++) if (st.mask[i] > 0.5) { hist[cl.data[i]]++; n++; }
  let lo = 0, hi = 255;
  if (n > 100) { lo = percentileFromHist(hist, n, 0.02); hi = percentileFromHist(hist, n, 0.98); }
  const g = new Float32Array(N);
  const g8 = new cv.Mat(h, w, cv.CV_8U);
  for (let i = 0; i < N; i++) {
    const v = Math.pow(clamp01((cl.data[i] - lo) / Math.max(hi - lo, 1)), p.gamma);
    g[i] = v;
    g8.data[i] = Math.round(v * 255);
  }
  cl.delete();
  const gsm = new cv.Mat();
  cv.bilateralFilter(g8, gsm, Math.round(9 * f) || 1, 30, 9 * f);
  g8.delete();
  const gs = gaussF(Float32Array.from(gsm.data), w, h, 1.2 * f);
  gsm.delete();

  const steps = p.levels - 1;
  const tone = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = (gs[i] / 255) * steps;
    const base = Math.floor(x);
    let t = clamp01((x - base - 0.35) / 0.3);
    t = t * t * (3 - 2 * t);
    const q = (base + t) / steps;
    const o = q * p.posterize + g[i] * (1 - p.posterize);
    tone[i] = p.black + o * (p.white - p.black);
  }

  // เส้นขอบบาง ๆ
  if (p.ink > 0) {
    const md = new cv.Mat(), th = new cv.Mat();
    cv.medianBlur(gray, md, odd(5 * f));
    cv.adaptiveThreshold(md, th, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, odd(11 * f), 9);
    const e = gaussF(Float32Array.from(th.data), w, h, 0.8 * f);
    for (let i = 0; i < N; i++) tone[i] *= 1 - (1 - e[i] / 255) * p.ink;
    md.delete(); th.delete();
  }
  gray.delete();

  // รายละเอียดผม/ผ้า + สีฟ้าในเงา
  const lo2 = p.black * 0.6;
  const blue = hexToRgb(p.bg);
  const person = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    let v = tone[i] + st.hp[i] * p.detail;
    v = v < lo2 ? lo2 : v > p.white ? p.white : v;
    const sw = 1 - v / 255;
    for (let c = 0; c < 3; c++) person[i * 3 + c] = v * (1 - p.tint * sw) + blue[c] * 0.35 * p.tint * sw;
  }
  return person;
}

function composite(st, p, seed, maskE) {
  const bgKey = `${p.bg}|${p.shards}|${seed}`;
  if (!st.bgCache || st.bgCache.key !== bgKey)
    st.bgCache = { key: bgKey, data: makeBackground(st.w, st.h, p.bg, p.shards, seed) };
  const bg = st.bgCache.data, P = st.person, N = st.w * st.h;
  const out = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const m = maskE[i];
    out[i*4]   = P[i*3]   * m + bg[i*4]   * (1 - m);
    out[i*4+1] = P[i*3+1] * m + bg[i*4+1] * (1 - m);
    out[i*4+2] = P[i*3+2] * m + bg[i*4+2] * (1 - m);
    out[i*4+3] = 255;
  }
  return out;
}

function maskBytes(maskE) {
  const m = new Uint8Array(maskE.length);
  for (let i = 0; i < m.length; i++) m[i] = maskE[i] * 255;
  return m;
}

// ------------------------------------------------------------ รับคำสั่งจากหน้าเว็บ
async function loadCV() {
  let c = self.cv;
  if (c instanceof Promise) c = await c;
  if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
  return c;
}

const handlers = {
  async init() {
    cv = await loadCV();
    ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(new URL('models/silueta.onnx', self.location.href).href,
      { executionProviders: ['wasm'] });
    return {};
  },

  async load({ bitmap, previewMax }) {
    if (src) src.close();
    freeState(preview);
    preview = null;
    src = bitmap;
    pred = await runModel(src);
    preview = prepare(src, previewMax);
    return { w: preview.w, h: preview.h };
  },

  // เรนเดอร์เต็มขั้นตอนที่ขนาดพรีวิว
  render({ p, seed, edits, editsVer }) {
    const st = preview;
    buildMask(st, p, edits, editsVer);
    st.person = renderPerson(st, p);
    st.personParams = p;
    const img = composite(st, p, seed, st.maskE);
    const mask = maskBytes(st.maskE);
    return { result: { w: st.w, h: st.h, img, mask }, transfer: [img.buffer, mask.buffer] };
  },

  // หลังระบาย: รวมภาพใหม่เร็ว ๆ โดยใช้ชั้นตัวคนเดิม
  composite({ p, seed, edits }) {
    const st = preview;
    if (!st.person) return handlers.render({ p, seed, edits, editsVer: -1 });
    if (!st.refined) buildMask(st, p, null, 0);
    const maskE = compositeMask(st, applyEdits(st.refined, edits));
    const img = composite(st, p, seed, maskE);
    const mask = maskBytes(maskE);
    return { result: { w: st.w, h: st.h, img, mask }, transfer: [img.buffer, mask.buffer] };
  },

  // ภาพความละเอียดเต็มสำหรับดาวน์โหลด
  export({ p, seed, edits, exportMax }) {
    const st = prepare(src, exportMax);
    let e = null;
    if (edits) {
      const ef = resizeF(Float32Array.from(edits), preview.w, preview.h, st.w, st.h);
      e = new Int8Array(ef.length);
      for (let i = 0; i < ef.length; i++) e[i] = Math.round(ef[i]);
    }
    buildMask(st, p, e, 'export');
    st.person = renderPerson(st, p);
    const img = composite(st, p, seed, st.maskE);
    freeState(st);
    return { result: { w: st.w, h: st.h, img }, transfer: [img.buffer] };
  },
};

self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data;
  try {
    const t0 = performance.now();
    const r = await handlers[type](payload || {});
    const result = r && r.result ? r.result : r;
    self.postMessage({ id, ok: true, result, ms: Math.round(performance.now() - t0) }, (r && r.transfer) || []);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
