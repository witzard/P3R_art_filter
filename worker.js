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
let parseSession = null;  // face parsing (BiSeNet, CelebAMask-HQ 19 classes)
let parse = null;         // { box:[x0,y0,x1,y1] (0..1), groups:{hair,skin,mouth,feat,cloth} 512x512 }

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

// ------------------------------------------------------------ face parsing (hair / skin / features / mouth)
// หา "หัว" จาก silhouette: ไล่จากบนลงล่างจนถึงช่วงคอที่แคบลงแล้วกว้างขึ้นอีก
function headBox(mask, w, h) {
  const widths = new Float32Array(h);
  let y0 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    let c = 0;
    for (let x = 0; x < w; x++) if (mask[y * w + x] >= 0.5) c++;
    widths[y] = c;
    if (c) { if (y0 < 0) y0 = y; y1 = y; }
  }
  if (y0 < 0 || y1 - y0 < 20) return null;
  const H = y1 - y0, k = Math.max(3, Math.floor(H / 60));
  const sm = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let a = 0, n = 0;
    for (let j = y - (k >> 1); j <= y + (k >> 1); j++) if (j >= 0 && j < h) { a += widths[j]; n++; }
    sm[y] = a / k;
  }
  let M = 0, neck = -1, mn = 1e9, inNeck = false;
  for (let y = y0; y < y0 + Math.floor(0.8 * H); y++) {
    const v = sm[y];
    if (!inNeck) {
      M = Math.max(M, v);
      if (y - y0 > 0.05 * H && v < 0.8 * M) { inNeck = true; mn = v; neck = y; }
    } else if (v < mn) { mn = v; neck = y; } else if (v > 1.15 * mn) break;
  }
  const hy0 = y0, hy1 = neck >= 0 ? neck : y0 + Math.floor(0.4 * H);
  let xmin = w, xmax = -1;
  for (let y = hy0; y < hy1; y++) for (let x = 0; x < w; x++)
    if (mask[y * w + x] >= 0.5) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; }
  const cx = xmax >= 0 ? (xmin + xmax) / 2 : w / 2, cy = (hy0 + hy1) / 2;
  const side = 1.25 * Math.max(hy1 - hy0, xmax >= 0 ? xmax - xmin : 0);
  const X0 = Math.max(0, Math.floor(cx - side / 2)), Y0 = Math.max(0, Math.floor(cy - side / 2));
  const X1 = Math.min(w, Math.floor(cx + side / 2)), Y1 = Math.min(h, Math.floor(cy + side / 2));
  if (X1 - X0 < 16 || Y1 - Y0 < 16) return null;
  return [X0 / w, Y0 / h, X1 / w, Y1 / h];
}

// รัน face parsing บนภาพหัวที่ crop มา แล้วรวม class เป็น 5 กลุ่ม (ค่าความน่าจะเป็น 0..1)
async function runParse(bitmap, box) {
  const S = 512;
  const c = new OffscreenCanvas(S, S);
  const x = c.getContext('2d', { willReadFrequently: true });
  x.imageSmoothingQuality = 'high';
  const sx = box[0] * bitmap.width, sy = box[1] * bitmap.height;
  x.drawImage(bitmap, sx, sy, (box[2] - box[0]) * bitmap.width, (box[3] - box[1]) * bitmap.height, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const t = new Float32Array(3 * S * S);
  for (let p = 0; p < S * S; p++)
    for (let ch = 0; ch < 3; ch++) t[ch * S * S + p] = (d[p * 4 + ch] / 255 - mean[ch]) / std[ch];
  const res = await parseSession.run({ [parseSession.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) });
  const o = res[parseSession.outputNames[0]].data, P = S * S;
  const G = { hair: [17], skin: [1, 7, 8, 10, 14, 15], mouth: [11], feat: [2, 3, 4, 5, 6, 12, 13], cloth: [16, 18] };
  const groups = {};
  for (const k in G) groups[k] = new Float32Array(P);
  const e = new Float32Array(19);
  for (let i = 0; i < P; i++) {
    let mx = -Infinity;
    for (let c2 = 0; c2 < 19; c2++) { e[c2] = o[c2 * P + i]; if (e[c2] > mx) mx = e[c2]; }
    let sum = 0;
    for (let c2 = 0; c2 < 19; c2++) { e[c2] = Math.exp(e[c2] - mx); sum += e[c2]; }
    for (const k in G) { let a = 0; for (const c2 of G[k]) a += e[c2]; groups[k][i] = a / sum; }
  }
  return groups;
}

// แผนที่บริเวณ (hair/skin/mouth/feat/cloth) ที่ขนาดของ state นี้
function buildRegions(st) {
  const { w, h, raw } = st, N = w * h;
  const reg = { hair: new Float32Array(N), skin: new Float32Array(N), mouth: new Float32Array(N),
                feat: new Float32Array(N), cloth: new Float32Array(N) };
  st.reg = reg; st.hasRegions = false;
  if (!parse) return;
  const X0 = Math.round(parse.box[0] * w), Y0 = Math.round(parse.box[1] * h);
  const cw = Math.max(1, Math.round(parse.box[2] * w) - X0), ch = Math.max(1, Math.round(parse.box[3] * h) - Y0);
  for (const k in reg) {
    const r = resizeF(parse.groups[k], 512, 512, cw, ch);
    for (let y = 0; y < ch && Y0 + y < h; y++)
      for (let x = 0; x < cw && X0 + x < w; x++) {
        const i = (Y0 + y) * w + X0 + x;
        reg[k][i] = clamp01(r[y * cw + x]) * raw[i];
      }
  }
  // ตรวจว่าเป็นหน้าคนจริง (มีผิวพอสมควร) ไม่งั้นใช้แบบไม่มีบริเวณ (fallback)
  let sk = 0;
  for (let i = 0; i < N; i++) sk += reg.skin[i];
  if (sk < 0.01 * cw * ch) { for (const k in reg) reg[k].fill(0); return; }
  st.hasRegions = true;
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
  const __d1 = inside.data;
  for (let i = 0; i < N; i++) __d1[i] = raw[i] >= 0.5 ? 1 : 0;
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
  const __d2 = region.data;
  const setRegion = () => { for (let i = 0; i < N; i++) __d2[i] = removed[i] > 0.3 && raw[i] > 0.5 ? 1 : 0; };
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
    const __d3 = seedOut.data;
    const __d4 = seedDark.data;
    for (let i = 0; i < N; i++) { __d3[i] = raw[i] < 0.5 ? 1 : 0; __d4[i] = dark[i]; }
    const k5 = cv.Mat.ones(odd(5 * f), odd(5 * f), cv.CV_8U), k7 = cv.Mat.ones(odd(7 * f), odd(7 * f), cv.CV_8U);
    cv.dilate(seedOut, seedOut, k5);
    cv.dilate(seedDark, seedDark, k7);
    const ok = new Uint8Array(nLab); ok[0] = 1;
    const lb = labels.data32S;
    const __d5 = region.data;
    const __d6 = seedOut.data;
    const __d7 = seedDark.data;
    for (let i = 0; i < N; i++) if (__d5[i] && (__d6[i] || __d7[i])) ok[lb[i]] = 1;
    const bad = new cv.Mat(h, w, cv.CV_8U);
    const __d8 = bad.data;
    for (let i = 0; i < N; i++) __d8[i] = ok[lb[i]] ? 0 : 1;
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
  const __d9 = band.data;
  for (let i = 0; i < N; i++) {
    __d9[i] = mask[i] > 0.02 && mask[i] <= 0.95 ? 255 : 0;
    inner[i] = mask[i] > 0.95 ? 1 : 0;
  }
  const k5 = cv.Mat.ones(odd(5 * f), odd(5 * f), cv.CV_8U);
  cv.dilate(band, band, k5);
  k5.delete();
  const __d10 = band.data;
  for (let i = 0; i < N; i++) if (inner[i]) __d10[i] = 0;

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

  const g0m = new cv.Mat(); cv.cvtColor(rgb, g0m, cv.COLOR_RGB2GRAY);
  const g0 = Float32Array.from(g0m.data);
  const ga = gaussF(g0, w, h, 1.5 * f), gb = gaussF(g0, w, h, 2.5 * f), gc = gaussF(g0, w, h, 4.5 * f);
  const hp = new Float32Array(w * h), hpc = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { hp[i] = g0[i] - gb[i]; hpc[i] = ga[i] - gc[i]; }

  const st = {
    w, h, f, raw, edge, g0, hp, hpc, rgb8: rgb.data.slice(),
    band: null, maskKey: null, refined: null, refinedKey: null,
    bgCache: null, person: null, simpKey: null, simp: null,
  };
  rgb.delete(); g0m.delete();
  buildRegions(st);
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

// ------------------------------------------------------------ ชั้นตัวคน: cartoon / cel-shading pipeline
function u8Mat(w, h, src) {
  const m = new cv.Mat(h, w, cv.CV_8U);
  const __d11 = m.data;
  for (let i = 0; i < src.length; i++) { const v = src[i]; __d11[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
  return m;
}
function ellipse(n) { return cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(n, n)); }
function median8(m, k) { const d = new cv.Mat(); cv.medianBlur(m, d, k); m.delete(); return d; }

// cel shading ของบริเวณหนึ่ง: quantize -> median merge -> ลบ region เล็ก -> palette
// ทำที่ครึ่งความละเอียด (เงาเป็นก้อนใหญ่อยู่แล้ว) แล้วขยายกลับ: เร็วขึ้น ~4 เท่า
function cel(st, vv, Lr, kFull, minFrac, loFrac, p) {
  const W = st.w, Hh = st.h;
  const w = Math.max(1, W >> 1), h = Math.max(1, Hh >> 1), N = w * h;
  const k = odd(kFull / 2);
  const vm = newF32(W, Hh, vv), vs = new cv.Mat();
  cv.resize(vm, vs, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  const V = vs.data32F;
  const step = Math.floor(255 / Math.max(Lr - 1, 1));
  let lab = new cv.Mat(h, w, cv.CV_8U);
  const LD0 = lab.data;
  for (let i = 0; i < N; i++) LD0[i] = Math.min(Math.floor(V[i] * Lr), Lr - 1) * step;
  vm.delete(); vs.delete();
  lab = median8(median8(lab, k), k);
  if (p.smallRegion > 0) {
    const minArea = p.smallRegion * minFrac * N;
    const fill = new cv.Mat();
    cv.medianBlur(lab, fill, odd(k * 2.5));
    const small = new Uint8Array(N);
    const bin = new cv.Mat(h, w, cv.CV_8U), labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
    const LD = lab.data, BD = bin.data;
    for (let l = 0; l < Lr; l++) {
      const val = l * step;
      let present = false;
      for (let i = 0; i < N; i++) { const b = LD[i] === val ? 1 : 0; BD[i] = b; if (b) present = true; }
      if (!present) continue;
      const n = cv.connectedComponentsWithStats(bin, labels, stats, cents, 8, cv.CV_32S);
      const A = stats.data32S, bad = new Uint8Array(n);
      let any = false;
      for (let c = 1; c < n; c++) if (A[c * 5 + 4] < minArea) { bad[c] = 1; any = true; }
      if (any) { const L32 = labels.data32S; for (let i = 0; i < N; i++) if (bad[L32[i]]) small[i] = 1; }
    }
    const FD = fill.data;
    for (let i = 0; i < N; i++) if (small[i]) LD[i] = FD[i];
    fill.delete(); bin.delete(); labels.delete(); stats.delete(); cents.delete();
  }
  const pal = [];
  for (let l = 0; l < Lr; l++)
    pal.push(p.black + (p.white - p.black) * (loFrac + (1 - loFrac) * Math.pow(l / Math.max(Lr - 1, 1), 0.9)));
  const outS = new cv.Mat(h, w, cv.CV_32F), OD = outS.data32F, LD2 = lab.data;
  for (let i = 0; i < N; i++) OD[i] = pal[Math.min(Lr - 1, Math.round(LD2[i] / step))];
  lab.delete();
  const big = new cv.Mat();
  cv.resize(outS, big, new cv.Size(W, Hh), 0, 0, cv.INTER_LINEAR);
  const out = big.data32F.slice();
  outS.delete(); big.delete();
  return out;
}

function renderPerson(st, p) {
  const { w, h, f } = st, N = w * h;
  const R = st.reg, mask = st.mask;
  const hair = gaussF(R.hair, w, h, 2 * f), skin = gaussF(R.skin, w, h, 2 * f);
  const feat = gaussF(R.feat, w, h, 1.5 * f), mouth = R.mouth;
  let mx = (a) => { let m = 0; for (let i = 0; i < a.length; i += 7) if (a[i] > m) m = a[i]; return m; };
  const hasHair = st.hasRegions && mx(hair) > 0.3, hasSkin = st.hasRegions && mx(skin) > 0.3;
  const hasMouth = st.hasRegions && mx(mouth) > 0.3;
  const s = p.cartoon, hc = p.hairCleanup, fs = p.facialSmoothing, fp = p.featurePreservation;

  // 1. Cartoon simplification: bilateral หลายรอบบนภาพสี (เก็บผลไว้ตามค่า cartoon)
  if (st.simpKey !== s) {
    let a = new cv.Mat(h, w, cv.CV_8UC3); a.data.set(st.rgb8);
    const passes = 1 + Math.round(3 * s);
    for (let i = 0; i < passes; i++) {
      const b = new cv.Mat();
      cv.bilateralFilter(a, b, Math.max(1, Math.round(7 * f)), 20 + 30 * s, 6 * f);
      a.delete(); a = b;
    }
    let g = new cv.Mat();
    cv.cvtColor(a, g, cv.COLOR_RGB2GRAY);
    a.delete();
    g = median8(g, odd((3 + 4 * s) * f));
    st.simp = g.data.slice(); st.simpKey = s;
    g.delete();
  }
  const Sf = Float32Array.from(st.simp);

  // 2. Hair: รวมเส้นผมเป็นก้อน (median + closing/opening + bilateral)
  if (hasHair && hc > 0) {
    let H = u8Mat(w, h, st.simp);
    H = median8(H, odd((5 + 10 * hc) * f));
    const k = ellipse(odd(3 + 6 * hc * f));
    cv.morphologyEx(H, H, cv.MORPH_CLOSE, k);
    cv.morphologyEx(H, H, cv.MORPH_OPEN, k);
    const Hb = new cv.Mat();
    cv.bilateralFilter(H, Hb, Math.max(1, Math.round(9 * f)), 40, 9 * f);
    const __d18 = Hb.data;
    for (let i = 0; i < N; i++) { const a = hair[i] * hc; Sf[i] = Sf[i] * (1 - a) + __d18[i] * a; }
    H.delete(); Hb.delete(); k.delete();
  }

  // 3. Face: เกลี่ยผิวเพิ่ม แต่เก็บตา/คิ้ว/ปาก
  if (hasSkin && fs > 0) {
    const g = u8Mat(w, h, st.simp), F1 = new cv.Mat(), F2 = new cv.Mat();
    const d = Math.max(1, Math.round(11 * f));
    cv.bilateralFilter(g, F1, d, 25 + 35 * fs, 11 * f);
    cv.bilateralFilter(F1, F2, d, 25 + 35 * fs, 11 * f);
    const __d19 = F2.data;
    for (let i = 0; i < N; i++) { const a = skin[i] * fs * (1 - feat[i] * fp); Sf[i] = Sf[i] * (1 - a) + __d19[i] * a; }
    g.delete(); F1.delete(); F2.delete();
  }
  // ฟัน: ปิดร่องมืดระหว่างฟันเฉพาะในช่องปาก เก็บขอบปากไว้
  if (hasMouth) {
    // ทำเฉพาะกรอบรอบปาก
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
      if (mouth[y * w + x] > 0.3) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const pad = Math.round(12 * f);
    x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(w - 1, x1 + pad); y1 = Math.min(h - 1, y1 + pad);
    const rw = x1 - x0 + 1, rh = y1 - y0 + 1;
    if (rw > 4 && rh > 4) {
      const g = new cv.Mat(rh, rw, cv.CV_8U), inner = new cv.Mat(rh, rw, cv.CV_8U);
      const GD = g.data, ID = inner.data;
      for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
        const i = (y0 + y) * w + x0 + x, v = Sf[i];
        GD[y * rw + x] = v < 0 ? 0 : v > 255 ? 255 : v;
        ID[y * rw + x] = mouth[i] > 0.5 ? 1 : 0;
      }
      const T = new cv.Mat(), k = ellipse(odd(9 * f));
      cv.morphologyEx(g, T, cv.MORPH_CLOSE, k);
      const Tm = median8(T, odd(5 * f));
      const k3 = cv.Mat.ones(odd(3 * f), odd(3 * f), cv.CV_8U);
      cv.erode(inner, inner, k3);
      const inn = gaussF(Float32Array.from(inner.data), rw, rh, 1.5 * f);
      const TD = Tm.data;
      for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
        const i = (y0 + y) * w + x0 + x, j = y * rw + x, a = inn[j];
        Sf[i] = Sf[i] * (1 - a) + Math.max(Sf[i], TD[j] * 0.92) * a;
      }
      g.delete(); inner.delete(); Tm.delete(); k.delete(); k3.delete();
    }
  }

  // 4. Tone normalization (percentile ของตัวคน) + gamma
  const hist = new Array(256).fill(0);
  let n = 0;
  for (let i = 0; i < N; i++) if (mask[i] > 0.5) { hist[Math.max(0, Math.min(255, Math.round(Sf[i])))]++; n++; }
  let lo = 0, hi = 255;
  if (n > 100) { lo = percentileFromHist(hist, n, 0.02); hi = percentileFromHist(hist, n, 0.98); }
  const med = n > 100 ? percentileFromHist(hist, n, 0.5) : 128;
  const v = new Float32Array(N);
  for (let i = 0; i < N; i++) v[i] = Math.pow(clamp01((Sf[i] - lo) / Math.max(hi - lo, 1)), p.gamma);

  // 5. Cel shading แยกตามบริเวณ
  const L = p.levels;
  let tone = cel(st, v, L, odd((3 + 10 * p.shadowMerge) * f), 0.0012, 0, p);
  if (hasSkin) {
    const vb = gaussF(v, w, h, 14 * f), vs = new Float32Array(N);
    for (let i = 0; i < N; i++) vs[i] = v[i] * (1 - 0.7 * fs) + vb[i] * 0.7 * fs;
    const ts = cel(st, vs, Math.min(L, 3), odd((7 + 16 * fs) * f), 0.004, 0.35, p); // เงาผิวไม่ดำสนิท
    for (let i = 0; i < N; i++) { const a = skin[i] * (1 - feat[i] * fp); tone[i] = tone[i] * (1 - a) + ts[i] * a; }
  }
  if (hasHair && hc > 0) {
    const vb = gaussF(v, w, h, 6 * f), vh = new Float32Array(N);
    for (let i = 0; i < N; i++) vh[i] = (v[i] * (1 - 0.5 * hc) + vb[i] * 0.5 * hc) * (1 - 0.6 * hc); // ผมเป็นก้อนเข้ม ไฮไลต์เป็นโทนกลาง
    const th = cel(st, vh, hc > 0.4 ? Math.min(L, 3) : L, odd((7 + 22 * hc) * f), 0.004, 0, p);
    for (let i = 0; i < N; i++) tone[i] = tone[i] * (1 - hair[i]) + th[i] * hair[i];
  }
  tone = gaussF(tone, w, h, (1 - p.shadingHardness) * 4 * f + 0.6 * f);
  const hd = p.shadingHardness;
  for (let i = 0; i < N; i++)
    tone[i] = tone[i] * (0.85 + 0.15 * hd) + (p.black + v[i] * (p.white - p.black)) * (0.15 - 0.15 * hd);

  // 6. คืนรายละเอียดแบบเลือกบริเวณ: ผิวแทบไม่คืน, ผ้าคืนปานกลาง, ผมคืนเฉพาะโครงหยาบ
  for (let i = 0; i < N; i++) {
    const cloth = clamp01(mask[i] - hair[i] - skin[i] - feat[i] - mouth[i]);
    const wd = skin[i] * 0.03 + cloth * 0.35 + feat[i] * 0.3;
    tone[i] += p.detail * (st.hp[i] * wd + st.hpc[i] * hair[i] * p.hairDetail * 0.6);
  }

  // 7. Line art หลายระดับ
  const mb = new cv.Mat(h, w, cv.CV_8U);
  const __d22 = mb.data;
  for (let i = 0; i < N; i++) __d22[i] = mask[i] > 0.5 ? 1 : 0;
  // A. silhouette: ขอบตัว + ขอบผมกับหน้า
  const t = Math.max(1, Math.round((1 + p.silhouette) * f * 1.5));
  const dil = new cv.Mat(), ero = new cv.Mat();
  const kd = cv.Mat.ones(2 * t + 1, 2 * t + 1, cv.CV_8U), ke = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.dilate(mb, dil, kd); cv.erode(mb, ero, ke);
  const ring = new Float32Array(N);
  const __d23 = ero.data;
  const __d24 = mb.data;
  const __d25 = dil.data;
  for (let i = 0; i < N; i++) ring[i] = (__d25[i] - __d23[i]) * __d24[i];
  let sil = gaussF(ring, w, h, 0.7 * f);
  const sA = Math.min(1, p.silhouette * 0.7);
  for (let i = 0; i < N; i++) sil[i] *= sA;
  if (hasHair && hasSkin) {
    const hb = new cv.Mat(h, w, cv.CV_8U), hd2 = new cv.Mat();
    const __d26 = hb.data;
    for (let i = 0; i < N; i++) __d26[i] = hair[i] > 0.5 ? 1 : 0;
    const kh = cv.Mat.ones(odd(2 * t), odd(2 * t), cv.CV_8U);
    cv.dilate(hb, hd2, kh);
    const hr = new Float32Array(N);
    const __d27 = hb.data;
    const __d28 = hd2.data;
    for (let i = 0; i < N; i++) hr[i] = (__d28[i] - __d27[i]) * (skin[i] > 0.3 ? 1 : 0);
    const hrb = gaussF(hr, w, h, 0.7 * f), hA = Math.min(1, p.silhouette * 0.6);
    for (let i = 0; i < N; i++) sil[i] = Math.max(sil[i], hrb[i] * hA);
    hb.delete(); hd2.delete(); kh.delete();
  }
  dil.delete(); ero.delete(); kd.delete(); ke.delete();

  // B. structural: Canny ที่ครึ่งความละเอียด + กรองตามความยาวเส้นและบริเวณ
  const Su = u8Mat(w, h, Sf);
  const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1);
  const half = new cv.Mat(), edges = new cv.Mat();
  cv.resize(Su, half, new cv.Size(hw, hh), 0, 0, cv.INTER_AREA);
  cv.Canny(half, edges, 0.5 * med, 1.0 * med);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(edges, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
  const fn = new cv.Mat(h, w, cv.CV_8U);
  const __d29 = fn.data;
  for (let i = 0; i < N; i++) __d29[i] = feat[i] > 0.3 ? 1 : 0;
  const kf = cv.Mat.ones(odd(9 * f), odd(9 * f), cv.CV_8U);
  cv.dilate(fn, fn, kf);
  const sh = cv.Mat.zeros(hh, hw, cv.CV_8U);
  const baseLen = 30 * f, thick = Math.max(1, Math.round(0.8 * f));
  const color = new cv.Scalar(255);
  const fnD = fn.data;
  for (let c = 0; c < contours.size(); c++) {
    const cnt = contours.get(c), d = cnt.data32S, np2 = d.length >> 1;
    const mid = np2 >> 1;
    const x = Math.min(w - 1, d[2 * mid] * 2), y = Math.min(h - 1, d[2 * mid + 1] * 2), i = y * w + x;
    let ok = mask[i] >= 0.5, need = baseLen;
    if (ok) {
      if (st.hasRegions && fnD[i]) need *= 0.4 / Math.max(fp, 0.1);
      else if (hair[i] > 0.5) need *= 2 + 6 * (1 - p.hairDetail);
      else if (skin[i] > 0.3) ok = false; // ไม่มีเส้นภายในผิวหน้า
    }
    if (ok && cv.arcLength(cnt, false) * 2 >= need) cv.drawContours(sh, contours, c, color, thick);
    cnt.delete();
  }
  const sFull = new cv.Mat();
  cv.resize(sh, sFull, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
  let struct = gaussF(Float32Array.from(sFull.data), w, h, 0.7 * f);
  for (let i = 0; i < N; i++) struct[i] = (struct[i] / 255) * p.structural * (1 - hair[i] * (1 - p.hairDetail) * 0.6);
  half.delete(); edges.delete(); contours.delete(); hier.delete(); fn.delete(); kf.delete(); sh.delete(); sFull.delete();

  // B2. เส้นบาง ๆ ที่มืดบนหน้า (กรอบแว่น เปลือกตา ร่องปาก) ด้วย black-hat
  //     เผื่อกรณี face parsing หาไม่เจอ (เช่น หน้าด้านข้าง)
  if (fp > 0 && hasSkin) {
    const g1 = u8Mat(w, h, gaussF(st.g0, w, h, 0.8 * f)), bh = new cv.Mat(), kb = ellipse(odd(11 * f));
    cv.morphologyEx(g1, bh, cv.MORPH_BLACKHAT, kb);
    const zone = new cv.Mat(h, w, cv.CV_8U);
    const __d30 = zone.data;
    for (let i = 0; i < N; i++) __d30[i] = skin[i] > 0.3 ? 1 : 0;
    const kz = cv.Mat.ones(odd(41 * f), odd(41 * f), cv.CV_8U);
    cv.dilate(zone, zone, kz);
    const fl = new cv.Mat(h, w, cv.CV_8U);
    const __d31 = zone.data;
    const __d32 = bh.data;
    const __d33 = fl.data;
    for (let i = 0; i < N; i++)
      __d33[i] = __d32[i] > 16 && __d31[i] && mask[i] > 0.5 && hair[i] < 0.6 && mouth[i] < 0.3 ? 1 : 0;
    const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
    const nc = cv.connectedComponentsWithStats(fl, labels, stats, cents, 8, cv.CV_32S);
    const A = stats.data32S, keep = new Uint8Array(nc);
    for (let c = 1; c < nc; c++)
      keep[c] = Math.max(A[c * 5 + 2], A[c * 5 + 3]) >= 12 * f && A[c * 5 + 4] >= 10 * f * f ? 1 : 0;
    const L32 = labels.data32S, flf = new Float32Array(N);
    for (let i = 0; i < N; i++) flf[i] = keep[L32[i]];
    const flb = gaussF(flf, w, h, 0.7 * f);
    const __d34 = bh.data;
    for (let i = 0; i < N; i++) struct[i] = Math.max(struct[i], flb[i] * clamp01((__d34[i] - 10) / 25) * fp);
    g1.delete(); bh.delete(); kb.delete(); zone.delete(); kz.delete(); fl.delete(); labels.delete(); stats.delete(); cents.delete();
  }

  // C. fine detail: Canny ละเอียด แต่ถูกกดไว้เกือบหมด
  let fine = null;
  if (p.fineSuppress < 1) {
    const e = new cv.Mat();
    cv.Canny(Su, e, 0.6 * med, 1.2 * med);
    const sd = newF32(w, h, struct), sdd = new cv.Mat(), k5 = cv.Mat.ones(5, 5, cv.CV_8U);
    cv.dilate(sd, sdd, k5);
    const fr = new Float32Array(N);
    const __d35 = mb.data;
    const __d36 = e.data;
    const __d37 = sdd.data32F;
    for (let i = 0; i < N; i++) fr[i] = (__d36[i] / 255) * (1 - Math.min(1, __d37[i])) * __d35[i] * (1 - skin[i]);
    fine = gaussF(fr, w, h, 0.6 * f);
    const fA = (1 - p.fineSuppress) * 0.6;
    for (let i = 0; i < N; i++) fine[i] *= fA;
    e.delete(); sd.delete(); sdd.delete(); k5.delete();
  }
  Su.delete(); mb.delete();

  const blo = p.black * 0.5;
  for (let i = 0; i < N; i++) {
    let ln = Math.max(sil[i], struct[i]);
    if (fine) ln = Math.max(ln, fine[i]);
    ln = clamp01(ln) * 0.85;
    const tv = tone[i] * (1 - ln) + blo * ln;
    tone[i] = tv < blo ? blo : tv > p.white ? p.white : tv;
  }

  // 8. สี: grayscale (+ฟ้าจาง ๆ ในเงา) หรือ palette ฟ้าแบบ P3R ตามความสว่าง
  const person = new Float32Array(N * 3);
  if (p.colorMode === 'blue') {
    const stops = [0, 0.3, 0.6, 0.82, 1.0];
    const cols = [[8, 14, 40], [22, 52, 150], [40, 120, 215], [110, 210, 250], [236, 248, 255]];
    for (let i = 0; i < N; i++) {
      const tv = tone[i] / 255;
      let j = 0;
      while (j < stops.length - 2 && tv > stops[j + 1]) j++;
      const u = clamp01((tv - stops[j]) / (stops[j + 1] - stops[j]));
      const amt = p.tint * (1 - 0.5 * skin[i]); // ผิวซีดกว่าส่วนอื่น
      for (let c = 0; c < 3; c++) {
        const pc = cols[j][c] * (1 - u) + cols[j + 1][c] * u;
        person[i * 3 + c] = tone[i] * (1 - amt) + pc * amt;
      }
    }
  } else {
    const blue = hexToRgb(p.bg);
    for (let i = 0; i < N; i++) {
      const sw = 1 - tone[i] / 255;
      for (let c = 0; c < 3; c++) person[i * 3 + c] = tone[i] * (1 - p.tint * sw) + blue[c] * 0.35 * p.tint * sw;
    }
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
    try {
      parseSession = await ort.InferenceSession.create(
        new URL('models/face_parsing_resnet18.onnx', self.location.href).href, { executionProviders: ['wasm'] });
    } catch (e) { parseSession = null; }   // ไม่มีโมเดลก็ยังใช้งานได้ (ไม่มีการแยกผม/หน้า)
    return { parsing: !!parseSession };
  },

  async load({ bitmap, previewMax }) {
    if (src) src.close();
    freeState(preview);
    preview = null;
    src = bitmap;
    pred = await runModel(src);
    parse = null;
    if (parseSession) {
      const s = Math.min(1, previewMax / Math.max(src.width, src.height));
      const pw = Math.round(src.width * s), ph = Math.round(src.height * s);
      const box = headBox(rawMask(pw, ph, Math.max(pw, ph) / BASE), pw, ph);
      if (box) parse = { box, groups: await runParse(src, box) };
    }
    preview = prepare(src, previewMax);
    return { w: preview.w, h: preview.h, regions: preview.hasRegions };
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
