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
  const side = 1.5 * Math.max(hy1 - hy0, xmax >= 0 ? xmax - xmin : 0);
  const cx = xmax >= 0 ? (xmin + xmax) / 2 : w / 2, cy = (hy0 + hy1) / 2 + 0.08 * side; // เลื่อนลงให้รวมคอ แต่ยังเหลือขอบบนหัว
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
  const G = { hair: [17], skin: [1, 7, 8, 10, 14, 15], mouth: [11], feat: [2, 3, 4, 5, 6, 12, 13], cloth: [16, 18],
              brow: [2, 3], eye: [4, 5, 6], lip: [12, 13], nose: [10], neck: [14, 15], face: [1] };
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
  const keys = ['hair', 'skin', 'mouth', 'feat', 'cloth', 'brow', 'eye', 'lip', 'nose', 'neck', 'face'];
  const reg = {};
  for (const k of keys) reg[k] = new Float32Array(N);
  st.reg = reg; st.hasRegions = false;
  if (!parse) return;
  const X0 = Math.round(parse.box[0] * w), Y0 = Math.round(parse.box[1] * h);
  const X1 = Math.min(w, Math.round(parse.box[2] * w)), Y1 = Math.min(h, Math.round(parse.box[3] * h));
  const cw = Math.max(1, X1 - X0), ch = Math.max(1, Y1 - Y0);
  // ค่อย ๆ จางออกที่ขอบ crop (ไม่ใช่ขอบภาพ) กันรอยต่อเป็นเส้นตรง
  const fy = 0.08 * ch, fx = 0.08 * cw;
  const fade = (x, y) => {
    let d = 1;
    if (Y0 > 0) d = Math.min(d, y / fy);
    if (Y1 < h) d = Math.min(d, (ch - 1 - y) / fy);
    if (X0 > 0) d = Math.min(d, x / fx);
    if (X1 < w) d = Math.min(d, (cw - 1 - x) / fx);
    return clamp01(d);
  };
  for (const k of keys) {
    const r = resizeF(parse.groups[k], 512, 512, cw, ch), dst = reg[k];
    for (let y = 0; y < ch; y++)
      for (let x = 0; x < cw; x++) {
        const i = (Y0 + y) * w + X0 + x;
        dst[i] = clamp01(r[y * cw + x]) * fade(x, y) * raw[i];
      }
  }
  let sk = 0;
  for (let i = 0; i < N; i++) sk += reg.skin[i];
  if (sk < 0.01 * cw * ch) { for (const k of keys) reg[k].fill(0); return; }
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

// ------------------------------------------------------------ ROI helpers (ประมวลผลเฉพาะกรอบที่จำเป็น)
function bboxOf(arr, w, h, t, pad) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) { const r = y * w; for (let x = 0; x < w; x++) if (arr[r + x] > t) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } }
  if (x1 < 0) return null;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(w - 1, x1 + pad); y1 = Math.min(h - 1, y1 + pad);
  return { x0, y0, rw: x1 - x0 + 1, rh: y1 - y0 + 1 };
}
function subU8(src, w, bb) {
  const m = new cv.Mat(bb.rh, bb.rw, cv.CV_8U), D = m.data;
  for (let y = 0; y < bb.rh; y++) { const r = (bb.y0 + y) * w + bb.x0; for (let x = 0; x < bb.rw; x++) {
    const v = src[r + x]; D[y * bb.rw + x] = v < 0 ? 0 : v > 255 ? 255 : v; } }
  return m;
}

// ------------------------------------------------------------ morphology helpers (Uint8 0/1 arrays)
function morphU8(arr, w, h, op, k, shape = 'rect') {
  if (k < 2) return arr.slice();
  const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(arr);
  const ker = shape === 'ellipse' ? ellipse(k | 1) : cv.Mat.ones(k, k, cv.CV_8U);
  const d = new cv.Mat();
  if (op === 'dilate') cv.dilate(m, d, ker);
  else if (op === 'erode') cv.erode(m, d, ker);
  else cv.morphologyEx(m, d, op === 'close' ? cv.MORPH_CLOSE : cv.MORPH_OPEN, ker);
  const out = d.data.slice();
  m.delete(); d.delete(); ker.delete();
  return out;
}
function boundaryU8(B, w, h, width) {
  const d = morphU8(B, w, h, 'dilate', width), out = new Float32Array(B.length);
  for (let i = 0; i < B.length; i++) out[i] = d[i] - B[i];
  return out;
}
// ลบ component ที่เล็กกว่า minArea (value=1: เกาะ, value=0: รูภายใน region)
function removeSmall(M, R, w, h, minArea, value) {
  const N = w * h, b = new cv.Mat(h, w, cv.CV_8U), BD = b.data;
  for (let i = 0; i < N; i++) BD[i] = value ? M[i] : (!M[i] && R[i] ? 1 : 0);
  const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(b, labels, stats, cents, 8, cv.CV_32S);
  const A = stats.data32S, small = new Uint8Array(n);
  let any = false;
  for (let c = 1; c < n; c++) if (A[c * 5 + 4] < minArea) { small[c] = 1; any = true; }
  if (any) { const L32 = labels.data32S; for (let i = 0; i < N; i++) if (small[L32[i]]) M[i] = value ? 0 : 1; }
  b.delete(); labels.delete(); stats.delete(); cents.delete();
}
function percentileOf(vals, mask, q) {
  const hist = new Uint32Array(1024);
  let n = 0;
  for (let i = 0; i < vals.length; i++) if (mask[i]) { hist[Math.min(1023, Math.max(0, Math.floor(vals[i] * 1023)))]++; n++; }
  if (!n) return 0.5;
  let acc = 0;
  for (let b = 0; b < 1024; b++) { acc += hist[b]; if (acc >= n * q) return b / 1023; }
  return 1;
}

// โทนของบริเวณหนึ่งด้วย "nested tone masks": M_k = v >= t_k (ภายใน region)
// แต่ละ mask: closing/opening -> ลบเกาะและรูเล็ก (ตามสัดส่วนพื้นที่ภาพ) -> blur+threshold ให้ขอบเรียบ
// บังคับซ้อนกัน (M_k+1 อยู่ใน M_k) ทำที่ครึ่งความละเอียดแล้วขยายกลับ
function toneRegion(st, vv, region, th, pal, rClose, rOpen, minArea, sigma, hard) {
  const W = st.w, H = st.h, w = Math.max(1, W >> 1), h = Math.max(1, H >> 1), N = w * h;
  const vm = newF32(W, H, vv), vs = new cv.Mat();
  cv.resize(vm, vs, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  const V = vs.data32F.slice();
  vm.delete(); vs.delete();
  const rf = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) rf[i] = region[i];
  const rh = resizeF(rf, W, H, w, h), R = new Uint8Array(N);
  for (let i = 0; i < N; i++) R[i] = rh[i] > 0.5 ? 1 : 0;
  const rc = Math.round(rClose / 2), ro = Math.round(rOpen / 2), ma = minArea / 4, sg = Math.max(0.5, sigma / 2);
  const ramp = 1.5 + 6 * hard;
  const acc = new Float32Array(N).fill(pal[0]);
  let prev = R, alphaPrev = null;
  for (let k = 0; k < th.length; k++) {
    let M = new Uint8Array(N);
    for (let i = 0; i < N; i++) M[i] = V[i] >= th[k] && prev[i] ? 1 : 0;
    if (rc > 0) M = morphU8(M, w, h, 'close', 2 * rc + 1, 'ellipse');
    if (ro > 0) M = morphU8(M, w, h, 'open', 2 * ro + 1, 'ellipse');
    for (let i = 0; i < N; i++) M[i] &= R[i];
    removeSmall(M, R, w, h, ma, 1);
    removeSmall(M, R, w, h, ma, 0);
    const A = gaussF(Float32Array.from(M), w, h, sg);
    const alpha = new Float32Array(N), next = new Uint8Array(N), dp = pal[k + 1] - pal[k];
    for (let i = 0; i < N; i++) {
      let a = clamp01((A[i] - 0.5) * ramp + 0.5);
      a = Math.min(a, k === 0 ? prev[i] : alphaPrev[i]);
      acc[i] += a * dp; alpha[i] = a;
      next[i] = A[i] > 0.5 && prev[i] ? 1 : 0;
    }
    prev = next; alphaPrev = alpha;
  }
  return resizeF(acc, w, h, W, H);
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
  const bbHair = hasHair && hc > 0 ? bboxOf(hair, w, h, 0.01, Math.round(12 * f)) : null;
  if (bbHair) {
    let H = subU8(st.simp, w, bbHair);
    H = median8(H, odd((5 + 10 * hc) * f));
    const k = ellipse(odd(3 + 6 * hc * f));
    cv.morphologyEx(H, H, cv.MORPH_CLOSE, k);
    cv.morphologyEx(H, H, cv.MORPH_OPEN, k);
    const Hb = new cv.Mat();
    cv.bilateralFilter(H, Hb, Math.max(1, Math.round(9 * f)), 40, 9 * f);
    const HD = Hb.data, { x0, y0, rw, rh } = bbHair;
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      const i = (y0 + y) * w + x0 + x, a = hair[i] * hc;
      Sf[i] = Sf[i] * (1 - a) + HD[y * rw + x] * a;
    }
    H.delete(); Hb.delete(); k.delete();
  }

  // 3. Face: เกลี่ยผิวเพิ่ม แต่เก็บตา/คิ้ว/ปาก
  const bbSkin = hasSkin && fs > 0 ? bboxOf(skin, w, h, 0.01, Math.round(12 * f)) : null;
  if (bbSkin) {
    const g = subU8(st.simp, w, bbSkin), F1 = new cv.Mat(), F2 = new cv.Mat();
    const d = Math.max(1, Math.round(11 * f));
    cv.bilateralFilter(g, F1, d, 25 + 35 * fs, 11 * f);
    cv.bilateralFilter(F1, F2, d, 25 + 35 * fs, 11 * f);
    const FD = F2.data, { x0, y0, rw, rh } = bbSkin;
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      const i = (y0 + y) * w + x0 + x, a = skin[i] * fs * (1 - feat[i] * fp);
      Sf[i] = Sf[i] * (1 - a) + FD[y * rw + x] * a;
    }
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

  // ขอบ silhouette ปนสีพื้นหลังเดิม: แทนค่าในแถบขอบด้วยค่าเฉลี่ยของเนื้อด้านใน
  // (normalized convolution: เร็วกว่า inpaint มาก ผลใกล้เคียงกัน) กันขอบขาว/halo
  if (st.band) {
    const BD = st.band.data, inner = new Float32Array(N), num = new Float32Array(N);
    for (let i = 0; i < N; i++) { const a = !BD[i] && mask[i] > 0.5 ? 1 : 0; inner[i] = a; num[i] = Sf[i] * a; }
    const nb = gaussF(num, w, h, 4 * f), db = gaussF(inner, w, h, 4 * f);
    for (let i = 0; i < N; i++) if (BD[i] && db[i] > 1e-3) Sf[i] = nb[i] / db[i];
  }

  // 4. Tone normalization (percentile ของตัวคน) + gamma
  const hist = new Array(256).fill(0);
  let n = 0;
  for (let i = 0; i < N; i++) if (mask[i] > 0.5) { hist[Math.max(0, Math.min(255, Math.round(Sf[i])))]++; n++; }
  let lo = 0, hi = 255;
  if (n > 100) { lo = percentileFromHist(hist, n, 0.02); hi = percentileFromHist(hist, n, 0.98); }
  const v = new Float32Array(N);
  for (let i = 0; i < N; i++) v[i] = Math.pow(clamp01((Sf[i] - lo) / Math.max(hi - lo, 1)), p.gamma);

  // 5. โทนแบบ nested masks แยกตามบริเวณ (ไม่รวมข้ามขอบ ผม / หน้า / เสื้อผ้า)
  const L = p.levels, hard = p.shadingHardness, rng = p.white - p.black;
  const merge = p.shadowMerge, sup = p.smallRegion;
  const palOf = (cnt, a = 0, b = 1) => Array.from({ length: cnt }, (_, i) =>
    p.black + rng * (a + (b - a) * Math.pow(i / Math.max(cnt - 1, 1), 0.9)));
  const hairH = new Uint8Array(N), faceH = new Uint8Array(N), clothH = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    hairH[i] = hair[i] > 0.5 ? 1 : 0;
    faceH[i] = !hairH[i] && skin[i] + feat[i] + mouth[i] > 0.5 ? 1 : 0;
    clothH[i] = mask[i] > 0.5 && !hairH[i] && !faceH[i] ? 1 : 0;
  }
  const k4 = odd(4 * f);
  const hf = new Uint8Array(N);
  for (let i = 0; i < N; i++) hf[i] = hairH[i] | faceH[i];
  const core = morphU8(hf, w, h, 'erode', k4);
  const clothX = new Uint8Array(N);
  for (let i = 0; i < N; i++) clothX[i] = mask[i] > 0.02 && !core[i] ? 1 : 0;
  const over = (tr, region) => {
    const wg = gaussF(Float32Array.from(region), w, h, 1.0 * f);
    for (let i = 0; i < N; i++) tone[i] = tone[i] * (1 - wg[i]) + tr[i] * wg[i];
  };
  // เสื้อผ้า / ของ: ใช้ระดับเต็ม merge ปานกลาง
  const vc = gaussF(v, w, h, 1.5 * f);
  let tone = toneRegion(st, vc, clothX, Array.from({ length: L - 1 }, (_, k) => (k + 1) / L), palOf(L),
    (1 + 3 * merge) * f, (1 + 2 * merge) * f, sup * 0.0015 * N, (1 + 2 * merge) * f, hard);
  // ผม: เกณฑ์จากความสว่างของผมเอง, สว่างสุดแค่โทนกลาง-สว่าง, รวมก้อนใหญ่
  if (hasHair) {
    const vh = gaussF(v, w, h, (2 + 4 * hc) * f);
    const nl = hc > 0.3 ? Math.min(3, L) : Math.min(L, 4);
    const qs = nl === 2 ? [0.6] : nl === 3 ? [0.55, 0.9] : [0.4, 0.7, 0.92];
    const th = qs.map((q) => percentileOf(vh, hairH, q));
    // ขอบนอกของผม (mask มีแต่ความน่าจะเป็นผม < 0.5) ใช้ palette ของผมด้วย กันขอบสว่างแปลก ๆ
    const hairX = morphU8(hairH, w, h, 'dilate', odd(12 * f));
    for (let i = 0; i < N; i++) hairX[i] = hairX[i] && !faceH[i] && R.cloth[i] < 0.3 && mask[i] > 0.02 ? 1 : 0;
    over(toneRegion(st, vh, morphU8(hairX, w, h, 'dilate', k4), th, palOf(nl, 0, 0.62),
      (2 + 8 * hc) * f, (2 + 6 * hc) * f, (0.0015 + 0.006 * hc) * sup * N, (2 + 4 * hc) * f, hard), hairX);
  }
  // หน้า: normalize ความสว่างผิว, ไม่เกิน 3 ระดับ, เงาไม่ต่ำกว่าเทากลาง, ลบเงาเล็ก ๆ บนแก้ม
  if (hasSkin) {
    const vb = gaussF(v, w, h, 14 * f), vs = new Float32Array(N), sm = new Uint8Array(N);
    for (let i = 0; i < N; i++) { vs[i] = v[i] * (1 - 0.7 * fs) + vb[i] * 0.7 * fs; sm[i] = faceH[i] && skin[i] > 0.5 ? 1 : 0; }
    const a5 = percentileOf(vs, sm, 0.05), a95 = percentileOf(vs, sm, 0.95);
    for (let i = 0; i < N; i++) vs[i] = clamp01((vs[i] - a5) / Math.max(a95 - a5, 1e-3));
    const nl = Math.min(L, 3);
    over(toneRegion(st, vs, morphU8(faceH, w, h, 'dilate', k4), nl === 3 ? [0.22, 0.5] : [0.3], palOf(nl, 0.45, 1),
      (3 + 6 * fs) * f, (2 + 5 * fs) * f, (0.002 + 0.006 * fs) * sup * N + 1, (2 + 3 * fs) * f, hard), faceH);
  }
  // คิ้ว: ก้อนเข้มก้อนเดียว / ปาก: โทนเรียบ / ฟัน: ก้อนสว่างก้อนเดียวในช่องปากสีเข้ม
  if (st.hasRegions) {
    const thr = (src, t) => { const o = new Uint8Array(N); for (let i = 0; i < N; i++) o[i] = src[i] > t ? 1 : 0; return o; };
    if (mx(R.brow) > 0.3) {
      const bm = gaussF(Float32Array.from(morphU8(thr(R.brow, 0.45), w, h, 'close', odd(3 * f))), w, h, 0.8 * f);
      const bt = p.black + 0.18 * rng;
      for (let i = 0; i < N; i++) { const a = bm[i] * fp; tone[i] = tone[i] * (1 - a) + bt * a; }
    }
    if (mx(R.lip) > 0.3) {
      const lb = thr(R.lip, 0.5), lm = gaussF(Float32Array.from(lb), w, h, 0.8 * f);
      const lt = p.black + rng * (0.45 + 0.35 * percentileOf(v, lb, 0.5));
      for (let i = 0; i < N; i++) tone[i] = tone[i] * (1 - lm[i]) + lt * lm[i];
    }
    if (hasMouth) {
      const mm = thr(mouth, 0.5), t45 = percentileOf(v, mm, 0.45);
      let teeth = new Uint8Array(N);
      for (let i = 0; i < N; i++) teeth[i] = v[i] >= t45 && mm[i] ? 1 : 0;
      teeth = morphU8(teeth, w, h, 'close', odd(7 * f), 'ellipse');
      for (let i = 0; i < N; i++) teeth[i] &= mm[i];
      teeth = morphU8(teeth, w, h, 'open', odd(2 * f));
      const tm = gaussF(Float32Array.from(teeth), w, h, 1.0 * f), gm = gaussF(Float32Array.from(mm), w, h, 0.8 * f);
      const cav = p.black + 0.25 * rng, tt = p.black + 0.9 * rng;
      for (let i = 0; i < N; i++) {
        tone[i] = tone[i] * (1 - gm[i]) + cav * gm[i];
        tone[i] = tone[i] * (1 - tm[i]) + tt * tm[i];
      }
    }
  }
  for (let i = 0; i < N; i++) tone[i] = tone[i] * (0.9 + 0.1 * hard) + (p.black + v[i] * rng) * (0.1 - 0.1 * hard);

  // 6. คืนรายละเอียดเฉพาะบริเวณ: ผ้าเล็กน้อย, ผมเฉพาะโครงหยาบ, ผิวไม่คืน
  for (let i = 0; i < N; i++)
    tone[i] += p.detail * (st.hp[i] * clothH[i] * 0.15 + st.hpc[i] * hair[i] * p.hairDetail * 0.35);

  // 7. เส้นตามลำดับความสำคัญ: silhouette > ตา/ปาก/คิ้ว/แว่น > ขอบก้อนผม > รอยพับผ้า > ผิว (ไม่มี)
  const aa = (arr, sg = 0.6) => gaussF(arr instanceof Float32Array ? arr : Float32Array.from(arr), w, h, sg * f);
  const mbA = new Uint8Array(N);
  for (let i = 0; i < N; i++) mbA[i] = mask[i] > 0.5 ? 1 : 0;
  const t = Math.max(1, Math.round((1 + p.silhouette) * f * 1.5));
  const dl = morphU8(mbA, w, h, 'dilate', 2 * t + 1), er = morphU8(mbA, w, h, 'erode', 3);
  const ring = new Float32Array(N);
  for (let i = 0; i < N; i++) ring[i] = (dl[i] - er[i]) * mbA[i];
  const sil = aa(ring, 0.7);
  const sA = Math.min(1, p.silhouette * 0.75);
  for (let i = 0; i < N; i++) sil[i] *= sA;
  if (hasHair) {   // เส้นไรผมตัดกับหน้า
    const hr = boundaryU8(hairH, w, h, Math.max(2, t));
    for (let i = 0; i < N; i++) hr[i] *= faceH[i];
    const hrb = aa(hr, 0.7), hA = Math.min(1, p.silhouette * 0.6);
    for (let i = 0; i < N; i++) sil[i] = Math.max(sil[i], hrb[i] * hA);
  }
  const lw = Math.max(1, Math.round(1.6 * f));
  const lf = new Float32Array(N);
  const maxInto = (src, wgt) => { for (let i = 0; i < N; i++) { const v2 = src[i] * wgt; if (v2 > lf[i]) lf[i] = v2; } };
  const thrU = (src, t2) => { const o = new Uint8Array(N); for (let i = 0; i < N; i++) o[i] = src[i] > t2 ? 1 : 0; return o; };
  if (st.hasRegions) {
    if (mx(R.eye) > 0.3) maxInto(aa(boundaryU8(morphU8(thrU(R.eye, 0.45), w, h, 'close', odd(3 * f)), w, h, lw + 1)), 1);
    if (hasMouth) maxInto(aa(boundaryU8(thrU(mouth, 0.5), w, h, lw + 1)), 0.9);
    if (mx(R.lip) > 0.3) maxInto(aa(boundaryU8(thrU(R.lip, 0.5), w, h, lw)), 0.35);
    if (mx(R.neck) > 0.3 && mx(R.face) > 0.3) {   // เส้นกราม: ขอบระหว่างหน้ากับคอ
      const nd = morphU8(thrU(R.neck, 0.5), w, h, 'dilate', lw + 2), fc = thrU(R.face, 0.5);
      for (let i = 0; i < N; i++) nd[i] &= fc[i];
      maxInto(aa(nd), 0.7);
    }
    if (mx(R.nose) > 0.3) {   // จมูก: เส้นสั้น ๆ ใต้ปลายจมูก (ขอบล่างของบริเวณจมูก) แบบอนิเมะ
      const nb = thrU(R.nose, 0.5);
      let y0 = h, y1 = -1;
      for (let i = 0; i < N; i++) if (nb[i]) { const y = (i / w) | 0; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (y1 > y0) {
        const cut = y0 + 0.6 * (y1 - y0), bd = boundaryU8(nb, w, h, lw + 1);
        for (let i = 0; i < N; i++) if ((i / w | 0) < cut) bd[i] = 0;
        maxInto(aa(bd), 0.55);
      }
    }
  }
  // กรอบแว่น / เปลือกตา / ส่วนที่ parsing หาไม่เจอ: เส้นบางมืดด้วย black-hat
  if (fp > 0 && hasSkin) {
    // ถ้าเจอตา ใช้เฉพาะรอบตา/คิ้ว (แว่น เปลือกตา) กันรอยย่นบนแก้ม; ไม่เจอ (หน้าด้านข้าง) ใช้ทั้งหน้า
    let zone;
    if (st.hasRegions && mx(R.eye) > 0.3) {
      const eb = new Uint8Array(N);
      for (let i = 0; i < N; i++) eb[i] = R.eye[i] + R.brow[i] > 0.3 ? 1 : 0;
      zone = morphU8(eb, w, h, 'dilate', odd(31 * f));
    } else zone = morphU8(faceH, w, h, 'dilate', odd(41 * f));
    const bb = bboxOf(zone, w, h, 0, Math.round(8 * f));
    if (bb) {
      const { x0, y0, rw, rh } = bb, M = rw * rh;
      const g1 = subU8(st.g0, w, bb), bh = new cv.Mat(), kb = ellipse(odd(11 * f));
      cv.GaussianBlur(g1, g1, new cv.Size(0, 0), 0.8 * f, 0.8 * f);
      cv.morphologyEx(g1, bh, cv.MORPH_BLACKHAT, kb);
      const BH = bh.data, fl = new cv.Mat(rh, rw, cv.CV_8U), FL = fl.data;
      for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
        const i = (y0 + y) * w + x0 + x, j = y * rw + x;
        FL[j] = BH[j] > 16 && zone[i] && mbA[i] && hair[i] < 0.6 && mouth[i] < 0.3 ? 1 : 0;
      }
      const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
      const nc = cv.connectedComponentsWithStats(fl, labels, stats, cents, 8, cv.CV_32S);
      const A = stats.data32S, keep = new Uint8Array(nc);
      for (let c = 1; c < nc; c++) keep[c] = Math.max(A[c * 5 + 2], A[c * 5 + 3]) >= 12 * f && A[c * 5 + 4] >= 10 * f * f ? 1 : 0;
      const L32 = labels.data32S, flf = new Float32Array(M);
      for (let j = 0; j < M; j++) flf[j] = keep[L32[j]];
      const flb = gaussF(flf, rw, rh, 0.7 * f);
      for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
        const i = (y0 + y) * w + x0 + x, j = y * rw + x, v2 = flb[j] * clamp01((BH[j] - 10) / 25);
        if (v2 > lf[i]) lf[i] = v2;
      }
      g1.delete(); bh.delete(); kb.delete(); fl.delete(); labels.delete(); stats.delete(); cents.delete();
    }
  }
  for (let i = 0; i < N; i++) lf[i] *= 0.9 * fp;

  // ขอบก้อนโทนของผม (ปานกลาง)
  const hl = new Float32Array(N);
  if (hasHair) {
    const ht = new Float32Array(N);
    for (let i = 0; i < N; i++) ht[i] = tone[i] * hairH[i];
    const hb = newF32(w, h, gaussF(ht, w, h, 0.8 * f)), gx = new cv.Mat(), gy = new cv.Mat();
    cv.Sobel(hb, gx, cv.CV_32F, 1, 0); cv.Sobel(hb, gy, cv.CV_32F, 0, 1);
    const he = morphU8(hairH, w, h, 'erode', odd(5 * f)), GX = gx.data32F, GY = gy.data32F;
    const hk = 0.5 * p.structural * (0.4 + 0.6 * p.hairDetail);
    for (let i = 0; i < N; i++) hl[i] = clamp01(Math.sqrt(GX[i] * GX[i] + GY[i] * GY[i]) * he[i] / (rng * 0.6)) * hk;
    hb.delete(); gx.delete(); gy.delete();
  }
  // รอยพับผ้า: เส้น Canny ยาว ๆ บนภาพที่เกลี่ยแล้ว (ปานกลาง-ต่ำ)
  const cl = new Float32Array(N);
  {
    const Su = u8Mat(w, h, gaussF(Sf, w, h, 1.5 * f));
    const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1);
    const half = new cv.Mat(), ed = new cv.Mat();
    cv.resize(Su, half, new cv.Size(hw, hh), 0, 0, cv.INTER_AREA);
    const hh2 = new Array(256).fill(0), HD = half.data;
    for (let i = 0; i < HD.length; i++) hh2[HD[i]]++;
    const med = percentileFromHist(hh2, HD.length, 0.5);
    cv.Canny(half, ed, 0.66 * med, 1.33 * med);
    const cs = new cv.MatVector(), hi2 = new cv.Mat();
    cv.findContours(ed, cs, hi2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
    const ce = morphU8(clothH, w, h, 'erode', odd(5 * f));
    const minLen = 60 * f * (1 + 2 * p.fineSuppress);
    const keep = new cv.MatVector();
    for (let c = 0; c < cs.size(); c++) {
      const cnt = cs.get(c), d = cnt.data32S, mid = (d.length >> 2) << 1;
      const x = Math.min(w - 1, d[mid] * 2), y = Math.min(h - 1, d[mid + 1] * 2);
      if (ce[y * w + x] && cv.arcLength(cnt, false) * 2 >= minLen) {
        const ap = new cv.Mat(); cv.approxPolyDP(cnt, ap, 0.8, false); keep.push_back(ap); ap.delete();
      }
      cnt.delete();
    }
    const cimg = cv.Mat.zeros(hh, hw, cv.CV_8U);
    if (keep.size()) cv.polylines(cimg, keep, false, new cv.Scalar(255), 1, cv.LINE_AA);
    const cf = new Float32Array(hw * hh), CD = cimg.data;
    for (let i = 0; i < CD.length; i++) cf[i] = CD[i] / 255;
    const cu = gaussF(resizeF(cf, hw, hh, w, h), w, h, 0.5 * f), ck = 0.45 * p.structural;
    for (let i = 0; i < N; i++) cl[i] = cu[i] * ck;
    Su.delete(); half.delete(); ed.delete(); cs.delete(); hi2.delete(); keep.delete(); cimg.delete();
  }

  const blo = p.black * 0.5;
  for (let i = 0; i < N; i++) {
    const ln = clamp01(Math.max(sil[i], lf[i], hl[i], cl[i])) * 0.85;
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
