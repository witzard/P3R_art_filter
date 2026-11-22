// P3R Art Filter: ส่วนหน้าเว็บ (UI, แปรงระบาย, คุยกับ worker)
// การประมวลผลภาพทั้งหมดอยู่ใน worker.js หน้าเว็บจึงไม่ค้างระหว่างประมวลผล

const PREVIEW_MAX = 1000;  // พรีวิวขณะปรับค่า (เร็ว)
const EXPORT_MAX = 2000;   // ภาพที่ดาวน์โหลด (ความละเอียดเต็ม)
const MAX_FILE_MB = 30;
const MAX_PIXELS = 50e6;
const UNDO_LIMIT = 30;

// [key, label, min, max, step, default]
const PARAMS = [
  ['smooth',    'Skin smoothing',            0,   1,   0.01, 0.45],
  ['levels',    'Shadow levels',             2,   10,  1,    5],
  ['posterize', 'Posterize strength',        0,   1,   0.01, 0.55],
  ['detail',    'Hair/fabric detail',        0,   2,   0.05, 0.9],
  ['ink',       'Ink lines',                 0,   1,   0.01, 0.35],
  ['tint',      'Blue tint in shadows',      0,   0.5, 0.01, 0.12],
  ['black',     'Black point',               0,   80,  1,    22],
  ['white',     'White point',               180, 255, 1,    238],
  ['gamma',     'Gamma (>1 = darker)',       0.5, 2,   0.05, 1.0],
  ['shards',    'Glass shards',              0,   80,  1,    25],
  ['band',      'Edge cleanup (px)',         0,   60,  1,    30],
  ['protect',   'Protect nose/solid parts (px)', 0,   20,  1,    7],
];
const BRUSH_PARAMS = [
  ['brush', 'Brush size (px)', 4, 200, 2, 40],
  ['zoom',  'Zoom (%)',      100, 400, 25, 100],
];

const $ = (id) => document.getElementById(id);
const setStatus = (t) => { $('status').textContent = t; };

let seed = 7;
let loaded = false;
let W = 0, H = 0;              // ขนาดพรีวิว
let edits = null;              // Int8Array: + เพิ่มตัวคน, - ลบออก
let editsVer = 0, hasEdits = false;
let undoStack = [];
let lastImage = null, lastMask = null;
let origCanvas = null;

// ------------------------------------------------------------ worker
const worker = new Worker('worker.js');
let reqId = 0;
const waiting = new Map();
worker.onmessage = (e) => {
  const { id, ok, result, error, ms } = e.data;
  const w = waiting.get(id);
  waiting.delete(id);
  if (ok) w.res({ ...result, ms }); else w.rej(new Error(error));
};
worker.onerror = (e) => setStatus('Worker error: ' + e.message);
function call(type, payload, transfer = []) {
  return new Promise((res, rej) => {
    const id = ++reqId;
    waiting.set(id, { res, rej });
    worker.postMessage({ id, type, payload }, transfer);
  });
}

// คิวงาน: ถ้ากำลังประมวลผลอยู่ จะเก็บไว้แค่งานล่าสุด (render สำคัญกว่า composite)
let busy = false, pending = null;
function want(kind) {
  if (!loaded) return;
  pending = pending === 'render' || kind === 'render' ? 'render' : 'composite';
  pump();
}
async function pump() {
  if (busy || !pending) return;
  const kind = pending;
  pending = null;
  busy = true;
  setStatus('Processing...');
  try {
    const r = await call(kind, { p: getParams(), seed, edits: hasEdits ? edits : null, editsVer });
    show(r);
    setStatus(`Done (${r.w}×${r.h}, ${r.ms} ms)`);
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + err.message);
  }
  busy = false;
  pump();
}

let renderTimer = null;
function scheduleRender(delay = 120) {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => want('render'), delay);
}

// ------------------------------------------------------------ แสดงผล
function setCanvasSize(w, h) {
  for (const id of ['out', 'maskOverlay', 'strokes']) {
    const c = $(id);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
}

function show(r) {
  lastImage = new ImageData(new Uint8ClampedArray(r.img.buffer), r.w, r.h);
  lastMask = r.mask;
  drawMain();
  drawMaskOverlay();
}

function drawMain() {
  const ctx = $('out').getContext('2d');
  if ($('showOrig').checked && origCanvas) ctx.drawImage(origCanvas, 0, 0);
  else if (lastImage) ctx.putImageData(lastImage, 0, 0);
}

function currentMode() {
  return document.querySelector('input[name=mode]:checked').value;
}

function drawMaskOverlay() {
  const c = $('maskOverlay'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const visible = $('showMask').checked || currentMode() !== 'view';
  if (!visible || !lastMask) return;
  const img = ctx.createImageData(c.width, c.height), d = img.data;
  for (let i = 0; i < lastMask.length; i++) {
    const a = 255 - lastMask[i];
    if (a > 8) { d[i*4] = 255; d[i*4+1] = 0; d[i*4+2] = 0; d[i*4+3] = a * 0.35; }
  }
  ctx.putImageData(img, 0, 0);
}

// ------------------------------------------------------------ slider + ปุ่ม −/+
function makeSlider(box, [key, label, min, max, step, def], onChange) {
  const row = document.createElement('div');
  row.className = 'row';
  const lab = document.createElement('label');
  const name = document.createElement('span'); name.textContent = label;
  const val = document.createElement('span'); val.id = 'v_' + key; val.textContent = def;
  lab.append(name, val);

  const ctl = document.createElement('div');
  ctl.className = 'ctl';
  const input = document.createElement('input');
  Object.assign(input, { type: 'range', id: 'p_' + key, min, max, step, value: def });
  const dec = document.createElement('button'); dec.textContent = '−'; dec.type = 'button';
  const inc = document.createElement('button'); inc.textContent = '+'; inc.type = 'button';
  ctl.append(dec, input, inc);
  row.append(lab, ctl);
  box.appendChild(row);

  const decimals = (String(step).split('.')[1] || '').length;
  const update = () => { val.textContent = Number(input.value).toFixed(decimals); onChange(); };
  input.addEventListener('input', update);

  const bump = (dir) => {
    const v = Math.min(max, Math.max(min, parseFloat(input.value) + dir * step));
    input.value = v.toFixed(decimals);
    update();
  };
  // กดค้างเพื่อเพิ่ม/ลดต่อเนื่อง
  for (const [btn, dir] of [[dec, -1], [inc, 1]]) {
    let t1 = null, t2 = null;
    const stop = () => { clearTimeout(t1); clearInterval(t2); };
    btn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      bump(dir);
      t1 = setTimeout(() => { t2 = setInterval(() => bump(dir), 80); }, 400);
    });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) btn.addEventListener(ev, stop);
  }
}

function getParams() {
  const p = {};
  for (const [key] of PARAMS) p[key] = parseFloat($('p_' + key).value);
  p.bg = $('bg').value;
  return p;
}

function resetParams() {
  for (const [key, , , , step, def] of PARAMS) {
    $('p_' + key).value = def;
    $('v_' + key).textContent = Number(def).toFixed((String(step).split('.')[1] || '').length);
  }
  $('bg').value = '#12acf4';
  scheduleRender();
}

// ------------------------------------------------------------ แปรงระบาย
let stroke = null; // { pts, width, mode }

function toImageXY(e) {
  const r = $('strokes').getBoundingClientRect();
  return [(e.clientX - r.left) * (W / r.width), (e.clientY - r.top) * (H / r.height)];
}

function drawStroke(ctx, s, color) {
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = s.width;
  ctx.lineCap = ctx.lineJoin = 'round';
  if (s.pts.length === 1) {
    ctx.beginPath();
    ctx.arc(s.pts[0][0], s.pts[0][1], s.width / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(s.pts[0][0], s.pts[0][1]);
  for (const [x, y] of s.pts.slice(1)) ctx.lineTo(x, y);
  ctx.stroke();
}

function redrawStroke() {
  const c = $('strokes'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (stroke) drawStroke(ctx, stroke, stroke.mode === 'add' ? 'rgba(0,200,60,0.5)' : 'rgba(255,40,40,0.5)');
}

function commitStroke() {
  // วาดเส้นลง canvas ชั่วคราว แล้วแปลงเป็นค่า edits
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  drawStroke(ctx, stroke, '#fff');
  const a = ctx.getImageData(0, 0, W, H).data;
  const target = stroke.mode === 'add' ? 127 : -127;
  for (let i = 0; i < W * H; i++) {
    const k = a[i * 4 + 3] / 255;
    if (k > 0) edits[i] = Math.round(edits[i] * (1 - k) + target * k);
  }
  stroke = null;
  redrawStroke();
  editsChanged();
}

function pushUndo() {
  undoStack.push(edits.slice());
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
}

function editsChanged() {
  editsVer++;
  hasEdits = edits.some((v) => v !== 0);
  $('undo').disabled = undoStack.length === 0;
  $('clearEdits').disabled = !hasEdits;
  want('composite');      // แสดงผลเร็วทันที
  scheduleRender(700);    // แล้วค่อยคำนวณโทนใหม่ทั้งหมดเมื่อหยุดระบาย
}

function setupBrush() {
  const c = $('strokes');
  c.addEventListener('pointerdown', (e) => {
    const mode = currentMode();
    if (mode === 'view' || !loaded) return;
    e.preventDefault();
    c.setPointerCapture(e.pointerId);
    pushUndo();
    stroke = { pts: [toImageXY(e)], width: parseFloat($('p_brush').value), mode };
    redrawStroke();
  });
  c.addEventListener('pointermove', (e) => {
    if (!stroke) return;
    stroke.pts.push(toImageXY(e));
    redrawStroke();
  });
  const end = () => { if (stroke) commitStroke(); };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);

  for (const r of document.querySelectorAll('input[name=mode]')) {
    r.addEventListener('change', () => {
      c.classList.toggle('painting', currentMode() !== 'view');
      drawMaskOverlay();
    });
  }
  $('undo').onclick = () => {
    if (!undoStack.length) return;
    edits = undoStack.pop();
    editsChanged();
  };
  $('clearEdits').onclick = () => {
    pushUndo();
    edits.fill(0);
    editsChanged();
  };
}

// ------------------------------------------------------------ อัปโหลด / ดาวน์โหลด
async function onFile(f) {
  if (!f.type.startsWith('image/')) throw new Error('This file is not an image');
  if (f.size > MAX_FILE_MB * 1024 * 1024) throw new Error(`File is larger than ${MAX_FILE_MB} MB`);
  const bmp = await createImageBitmap(f, { imageOrientation: 'from-image' });
  if (bmp.width * bmp.height > MAX_PIXELS) { bmp.close(); throw new Error('Image resolution is too high'); }

  loaded = false;
  $('download').disabled = true;
  setStatus('Removing background...');

  // ภาพสำหรับ worker (ย่อเหลือ EXPORT_MAX)
  const s = Math.min(1, EXPORT_MAX / Math.max(bmp.width, bmp.height));
  const rc = document.createElement('canvas');
  rc.width = Math.round(bmp.width * s); rc.height = Math.round(bmp.height * s);
  const rctx = rc.getContext('2d');
  rctx.imageSmoothingQuality = 'high';
  rctx.drawImage(bmp, 0, 0, rc.width, rc.height);
  bmp.close();
  const big = await createImageBitmap(rc);

  // ภาพต้นฉบับขนาดพรีวิว (ไว้ดูเทียบ)
  const ps = Math.min(1, PREVIEW_MAX / Math.max(big.width, big.height));
  origCanvas = document.createElement('canvas');
  origCanvas.width = Math.round(big.width * ps);
  origCanvas.height = Math.round(big.height * ps);
  origCanvas.getContext('2d').drawImage(big, 0, 0, origCanvas.width, origCanvas.height);

  const r = await call('load', { bitmap: big, previewMax: PREVIEW_MAX }, [big]);
  W = r.w; H = r.h;
  setCanvasSize(W, H);
  edits = new Int8Array(W * H);
  editsVer = 0; hasEdits = false; undoStack = [];
  $('undo').disabled = true; $('clearEdits').disabled = true;
  loaded = true;
  want('render');
  $('download').disabled = false;
}

async function download() {
  $('download').disabled = true;
  setStatus('Rendering full resolution image...');
  try {
    const r = await call('export', { p: getParams(), seed, edits: hasEdits ? edits : null, exportMax: EXPORT_MAX });
    const c = document.createElement('canvas');
    c.width = r.w; c.height = r.h;
    c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(r.img.buffer), r.w, r.h), 0, 0);
    c.toBlob((b) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'p3r_filter.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
    setStatus(`Downloaded (${r.w}×${r.h}, ${r.ms} ms)`);
  } catch (err) {
    console.error(err);
    setStatus('Download failed: ' + err.message);
  }
  $('download').disabled = false;
}

// ------------------------------------------------------------ เริ่มต้น
async function init() {
  for (const p of PARAMS) makeSlider($('sliders'), p, () => scheduleRender());
  makeSlider($('brushSliders'), BRUSH_PARAMS[0], () => {});
  makeSlider($('brushSliders'), BRUSH_PARAMS[1], () => {
    $('stage').style.width = $('p_zoom').value + '%';
  });
  setupBrush();

  $('reset').onclick = resetParams;
  $('reseed').onclick = () => { seed = (Math.random() * 1e9) | 0; want('composite'); };
  $('bg').addEventListener('input', () => scheduleRender());
  $('showOrig').addEventListener('change', drawMain);
  $('showMask').addEventListener('change', drawMaskOverlay);
  $('download').onclick = download;
  $('file').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try { await onFile(f); } catch (err) {
      console.error(err);
      setStatus('Error: ' + err.message);
    }
  });

  try {
    await call('init');
    $('file').disabled = false;
    setStatus('Ready. Choose a photo to start.');
  } catch (err) {
    console.error(err);
    setStatus('Failed to load: ' + err.message);
  }
}

init();
