/* ------------------------------------------------------------------
 * MINERVA · versione italiana
 * Identica a app.js — solo le stringhe visibili all'utente sono in italiano.
 * Pipeline (zscore -> splat -> ensemble ONNX -> calibrazione isotonica
 * -> soglia) invariata.
 * ------------------------------------------------------------------ */

(() => {
'use strict';

// --- 1. definizione delle feature ----------------------------------------

const FEATURE_META = {
  // Informazioni personali
  age: { label: 'Età',                          unit: 'anni',  kind: 'num' },
  sex: { label: 'Sesso',                                       kind: 'seg', options: [[1, 'Uomo'], [2, 'Donna']] },
  bmi: { label: 'Indice di massa corporea (IMC)', unit: 'kg/m²', kind: 'num' },

  // Storia clinica (toggle binari)
  previous_episodes:             { label: 'Episodi precedenti di pancreatite biliare', hint: 'Hai mai avuto un attacco di pancreatite in passato?',           kind: 'bin' },
  diabetes:                      { label: 'Diabete',                                   hint: 'Diabete di tipo 1 o di tipo 2',                                kind: 'bin' },
  chronic_pulmonary_disease:     { label: 'Malattia polmonare cronica',                hint: 'Per esempio BPCO, enfisema, bronchite cronica',                kind: 'bin' },
  hypertension:                  { label: 'Pressione alta',                            hint: 'Ipertensione diagnosticata',                                   kind: 'bin' },
  atrial_fibrillation:           { label: 'Battito cardiaco irregolare',               hint: 'Fibrillazione atriale (FA)',                                   kind: 'bin' },
  ischemic_heart_disease:        { label: 'Malattia cardiaca',                         hint: 'Coronaropatia o infarto pregresso',                            kind: 'bin' },
  chronic_kidney_disease:        { label: 'Malattia renale cronica',                   hint: 'Insufficienza renale cronica diagnosticata',                   kind: 'bin' },
  hematopoietic_disease:         { label: 'Malattia del sangue o del midollo',         hint: 'Per esempio leucemia, linfoma, anemia da malattia cronica',    kind: 'bin' },
  immunosuppressive_medications: { label: 'Farmaci immunosoppressori in uso',          hint: 'Steroidi, chemioterapia, terapie post-trapianto, biologici',   kind: 'bin' },

  // Esami di laboratorio
  wbc:                  { label: 'Globuli bianchi',           unit: '× 10³/μL', hint: 'Più alti in caso di infezione o infiammazione. Usa l\'unità del tuo referto.', kind: 'num' },
  neutrophils:          { label: 'Neutrofili',                unit: '× 10³/μL', hint: 'Un tipo di globulo bianco. Usa l\'unità del tuo referto.',                  kind: 'num' },
  platelets:            { label: 'Piastrine',                 unit: '× 10³/μL', hint: 'Cellule che aiutano la coagulazione. Usa l\'unità del tuo referto.',         kind: 'num' },
  inr:                  { label: 'Tempo di coagulazione (INR)', unit: '',         hint: 'Più alto = coagulazione più lenta',                                          kind: 'num' },
  crp:                  { label: 'Proteina C reattiva (PCR)', unit: 'mg/L',     hint: 'Indicatore generico di infiammazione',                                       kind: 'num' },
  ast:                  { label: 'Enzima epatico — AST',      unit: 'U/L',      hint: 'Aspartato aminotransferasi',                                                 kind: 'num' },
  alt:                  { label: 'Enzima epatico — ALT',      unit: 'U/L',      hint: 'Alanina aminotransferasi',                                                   kind: 'num' },
  total_bilirubin:      { label: 'Bilirubina totale',         unit: 'mg/dL',    hint: 'Valori alti possono causare ittero (pelle e occhi gialli)',                  kind: 'num' },
  conjugated_bilirubin: { label: 'Bilirubina diretta',        unit: 'mg/dL',    hint: 'La forma "elaborata" della bilirubina',                                      kind: 'num' },
  ggt:                  { label: 'Enzima epatico — GGT',      unit: 'U/L',      hint: 'Gamma-glutamil transpeptidasi',                                              kind: 'num' },
  serum_lipase:         { label: 'Lipasi',                    unit: 'U/L',      hint: 'Un enzima del pancreas; molto alta in caso di pancreatite',                  kind: 'num' },
  ldh:                  { label: 'LDH',                       unit: 'U/L',      hint: 'Lattato deidrogenasi — rilasciata dai tessuti danneggiati',                  kind: 'num' },

  // Reperti
  choledocholithiasis: {
    label: 'Calcoli nel coledoco',
    hint: 'Le immagini hanno mostrato calcoli che ostruiscono la via biliare?',
    kind: 'seg',
    options: [[1, 'No'], [2, 'Possibile'], [3, 'Sì']],
  },
  cholangitis: {
    label: 'Infezione delle vie biliari',
    hint: 'Colangite — febbre + ittero + dolore addominale',
    kind: 'bin',
  },
  ercp: {
    label: 'Procedura endoscopica (CPRE)',
    hint: 'Hai eseguito una procedura per liberare la via biliare?',
    kind: 'seg',
    options: [
      [1, 'Nessuna'],
      [2, 'Diagnostica'],
      [3, 'Con incisione'],
      [4, 'Con stent'],
      [5, 'Altra'],
    ],
  },
};

const SECTIONS = {
  demographics:  ['age', 'sex', 'bmi'],
  comorbidities: ['previous_episodes', 'diabetes', 'chronic_pulmonary_disease', 'hypertension',
                  'atrial_fibrillation', 'ischemic_heart_disease', 'chronic_kidney_disease',
                  'hematopoietic_disease', 'immunosuppressive_medications'],
  laboratory:    ['wbc', 'neutrophils', 'platelets', 'inr', 'crp', 'ast', 'alt',
                  'total_bilirubin', 'conjugated_bilirubin', 'ggt', 'serum_lipase', 'ldh'],
  findings:      ['choledocholithiasis', 'cholangitis', 'ercp'],
};

// --- 2. caricamento asset ------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const status = $('#status');

function setStatus(text, state = '') {
  status.textContent = text;
  if (state) status.dataset.state = state; else status.removeAttribute('data-state');
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`HTTP ${r.status} su ${url}`);
  return r.json();
}

let stats, mapping, calibration, samples, sessions = [];

async function loadEverything() {
  setStatus('Caricamento mappatura…');
  [stats, mapping, calibration, samples] = await Promise.all([
    fetchJSON('assets/feature_stats.json'),
    fetchJSON('assets/mapping.json'),
    fetchJSON('assets/calibration.json'),
    fetchJSON('assets/samples.json'),
  ]);

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  ort.env.wasm.numThreads = 1;

  const seedURLs = Array.from({ length: calibration.headline.n_seeds }, (_, i) => `assets/seed_${i}.onnx`);
  setStatus(`Caricamento di ${seedURLs.length} modelli ONNX…`);
  sessions = await Promise.all(seedURLs.map(url => ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })));
}

// --- 3. pipeline di inferenza --------------------------------------------

const FEATURE_ORDER = [];
const COORDS = [];

function buildFeatureOrder() {
  FEATURE_ORDER.length = 0;
  COORDS.length = 0;
  for (const f of mapping.mapping) {
    FEATURE_ORDER.push(f.feature);
    COORDS.push({ y: f.y, x: f.x });
  }
}

function readFormVector() {
  const vec = new Float32Array(FEATURE_ORDER.length);
  for (let i = 0; i < FEATURE_ORDER.length; i++) {
    const name = FEATURE_ORDER[i];
    const meta = FEATURE_META[name];
    const raw = formValues[name];
    if (meta.kind === 'num') {
      const s = stats.continuous[name];
      const v = (raw == null || Number.isNaN(raw)) ? s.median : raw;
      vec[i] = (v - s.mean) / (s.std || 1e-8);
    } else {
      vec[i] = raw == null ? 0 : raw;
    }
  }
  return vec;
}

function splat16x16(zVec) {
  const S = mapping.deepinsight.image_size;
  const img = new Float32Array(S * S);
  for (let f = 0; f < zVec.length; f++) {
    const yc = COORDS[f].y;
    const xc = COORDS[f].x;
    const x0 = Math.floor(xc), y0 = Math.floor(yc);
    const x1 = Math.min(x0 + 1, S - 1);
    const y1 = Math.min(y0 + 1, S - 1);
    const dx = xc - x0;
    const dy = yc - y0;
    const w00 = (1 - dx) * (1 - dy);
    const w10 = dx * (1 - dy);
    const w01 = (1 - dx) * dy;
    const w11 = dx * dy;
    const v = zVec[f];
    img[y0 * S + x0] += w00 * v;
    img[y0 * S + x1] += w10 * v;
    img[y1 * S + x0] += w01 * v;
    img[y1 * S + x1] += w11 * v;
  }
  let sum = 0; for (let k = 0; k < img.length; k++) sum += img[k];
  const mean = sum / img.length;
  let sq = 0; for (let k = 0; k < img.length; k++) { const d = img[k] - mean; sq += d * d; }
  const std = Math.sqrt(sq / img.length) + 1e-6;
  for (let k = 0; k < img.length; k++) img[k] = (img[k] - mean) / std;
  return img;
}

function softmaxPos(logits2) {
  const m = Math.max(logits2[0], logits2[1]);
  const e0 = Math.exp(logits2[0] - m);
  const e1 = Math.exp(logits2[1] - m);
  return e1 / (e0 + e1);
}

async function ensembleProb(img16x16) {
  const tensor = new ort.Tensor('float32', img16x16, [1, 1, 16, 16]);
  let sum = 0;
  for (const s of sessions) {
    const out = await s.run({ image: tensor });
    sum += softmaxPos(out.logits.data);
  }
  return sum / sessions.length;
}

function isotonicInterp(p, cal) {
  const xs = cal.x, ys = cal.y;
  if (p <= xs[0]) return ys[0];
  if (p >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= p) lo = mid; else hi = mid;
  }
  const t = (p - xs[lo]) / Math.max(xs[hi] - xs[lo], 1e-12);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

// --- 4. canvas -----------------------------------------------------------

const VIRIDIS_STOPS = [
  [0.00, [0.267, 0.005, 0.329]],
  [0.13, [0.283, 0.140, 0.458]],
  [0.27, [0.254, 0.265, 0.530]],
  [0.40, [0.207, 0.372, 0.553]],
  [0.53, [0.164, 0.471, 0.558]],
  [0.67, [0.128, 0.567, 0.551]],
  [0.80, [0.135, 0.659, 0.518]],
  [0.93, [0.478, 0.821, 0.318]],
  [1.00, [0.993, 0.906, 0.144]],
];

function viridis(t) {
  if (t <= 0) return VIRIDIS_STOPS[0][1];
  if (t >= 1) return VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1][1];
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const [t0, c0] = VIRIDIS_STOPS[i];
    const [t1, c1] = VIRIDIS_STOPS[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]), c0[2] + f * (c1[2] - c0[2])];
    }
  }
  return [0, 0, 0];
}

function paintDeepInsight(img) {
  const canvas = $('#di-canvas');
  const ctx = canvas.getContext('2d');
  const S = 16;
  const sorted = Array.from(img).sort((a, b) => a - b);
  const lo = sorted[Math.max(0, Math.floor(sorted.length * 0.01))];
  const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.99))];
  const denom = Math.max(hi - lo, 1e-8);
  const id = ctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    const t = Math.max(0, Math.min(1, (img[i] - lo) / denom));
    const c = viridis(t);
    id.data[i * 4 + 0] = Math.round(c[0] * 255);
    id.data[i * 4 + 1] = Math.round(c[1] * 255);
    id.data[i * 4 + 2] = Math.round(c[2] * 255);
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

// --- 5. risultato --------------------------------------------------------

function renderResult(probCal) {
  const pct = probCal * 100;
  const pctRounded = Math.max(0, Math.round(pct));
  $('#prob-value').textContent = pctRounded;
  $('#prob-plain-num').textContent = pctRounded;

  const thr = calibration.threshold;
  $('#risk-thr-marker').style.left = `${thr * 100}%`;
  $('#risk-needle').style.left = `${Math.min(100, Math.max(0, pct))}%`;

  const verdict = $('#result-verdict');
  let band = 'low';
  let label = "Rientra nell'intervallo tipico. Continua a seguire le indicazioni del tuo team medico.";
  if (probCal >= thr * 1.6) {
    band = 'high';
    label = "È al di sopra dell'intervallo abituale. Vale la pena parlarne con il tuo medico per un controllo più ravvicinato.";
  } else if (probCal >= thr) {
    band = 'mid';
    label = "Si trova sul versante più alto. Segnalalo alla prossima visita di controllo.";
  }
  verdict.textContent = label;
  verdict.dataset.band = band;
}

function renderIdleResult() {
  $('#prob-value').textContent = '—';
  $('#prob-plain-num').textContent = '—';
  $('#risk-needle').style.left = '0%';
  $('#risk-thr-marker').style.left = `${(calibration?.threshold ?? 0.5) * 100}%`;
  const v = $('#result-verdict');
  v.textContent = 'Compila il modulo per vedere la tua stima.';
  v.dataset.band = 'idle';
}

function renderHeadline() {
  // Versione paziente: metriche tecniche volutamente nascoste.
}

// --- 6. costruzione del modulo -------------------------------------------

const formValues = {};

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return '';
  const abs = Math.abs(v);
  if (Number.isInteger(v)) return String(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function escHTML(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function makeNumField(name) {
  const meta = FEATURE_META[name];
  const s = stats.continuous[name];
  const wrap = document.createElement('div');
  wrap.className = 'field';
  wrap.innerHTML = `
    <label class="field-label" for="f-${name}">
      <span>${escHTML(meta.label)}</span>
      ${meta.unit ? `<span class="field-unit">${escHTML(meta.unit)}</span>` : ''}
    </label>
    <input id="f-${name}" name="${name}" class="field-num hint" type="number"
           inputmode="decimal" step="any"
           placeholder="tipico: ${fmtNum(s.median)}"
           value="${fmtNum(s.median)}">
    ${meta.hint ? `<span class="field-hint">${escHTML(meta.hint)}</span>` : ''}
  `;
  formValues[name] = s.median;
  const input = wrap.querySelector('input');
  input.addEventListener('input', () => {
    const v = input.value === '' ? null : Number(input.value);
    formValues[name] = (v == null || Number.isNaN(v)) ? null : v;
    schedulePredict();
  });
  return wrap;
}

function makeSegField(name) {
  const meta = FEATURE_META[name];
  const sCat = stats.categorical[name];
  const def = sCat?.mode ?? meta.options[0][0];
  const wrap = document.createElement('div');
  wrap.className = 'field';
  const opts = meta.options.map(([v, lbl]) =>
    `<input id="f-${name}-${v}" name="${name}" type="radio" value="${v}"${v === def ? ' checked' : ''}>
     <label for="f-${name}-${v}">${escHTML(lbl)}</label>`).join('');
  wrap.innerHTML = `
    <span class="field-label"><span>${escHTML(meta.label)}</span></span>
    <div class="segment" role="radiogroup" aria-label="${escHTML(meta.label)}">${opts}</div>
    ${meta.hint ? `<span class="field-hint">${escHTML(meta.hint)}</span>` : ''}
  `;
  formValues[name] = def;
  wrap.querySelectorAll('input[type="radio"]').forEach(r => {
    r.addEventListener('change', () => {
      if (r.checked) { formValues[name] = Number(r.value); schedulePredict(); }
    });
  });
  return wrap;
}

function makeBinField(name) {
  const meta = FEATURE_META[name];
  const def = stats.categorical[name]?.mode ?? 0;
  const wrap = document.createElement('div');
  wrap.className = 'toggle-row';
  wrap.innerHTML = `
    <div class="toggle-text">
      <span class="toggle-label">${escHTML(meta.label)}</span>
      ${meta.hint ? `<span class="toggle-hint">${escHTML(meta.hint)}</span>` : ''}
    </div>
    <label class="toggle">
      <input type="checkbox" id="f-${name}" name="${name}"${def ? ' checked' : ''}>
      <span class="toggle-track"></span>
      <span class="toggle-knob"></span>
      <span class="toggle-state" id="state-${name}">${def ? 'Sì' : 'No'}</span>
    </label>
  `;
  formValues[name] = def;
  const cb = wrap.querySelector('input');
  const stateLbl = wrap.querySelector('.toggle-state');
  cb.addEventListener('change', () => {
    formValues[name] = cb.checked ? 1 : 0;
    stateLbl.textContent = cb.checked ? 'Sì' : 'No';
    schedulePredict();
  });
  return wrap;
}

function makeField(name) {
  const meta = FEATURE_META[name];
  if (meta.kind === 'num') return makeNumField(name);
  if (meta.kind === 'seg') return makeSegField(name);
  if (meta.kind === 'bin') return makeBinField(name);
}

function buildForm() {
  for (const [section, fields] of Object.entries(SECTIONS)) {
    const host = document.querySelector(`[data-fields="${section}"]`);
    fields.forEach(name => host.appendChild(makeField(name)));
  }
}

function buildDemoPicker() {
  const sel = $('#demo-picker');
  samples.samples.forEach((row, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    const sex = row.clinical.sex === 1 ? 'uomo' : 'donna';
    const ageBucket = Math.round(row.clinical.age / 10) * 10;
    opt.textContent = `Esempio ${i + 1} · ${sex} sui ${ageBucket} anni`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => {
    if (sel.value === '') return;
    const row = samples.samples[Number(sel.value)].clinical;
    populateForm(row);
  });
}

function populateForm(row) {
  for (const [name, meta] of Object.entries(FEATURE_META)) {
    const v = row[name];
    if (v == null) continue;
    if (meta.kind === 'num') {
      const input = document.getElementById(`f-${name}`);
      if (input) input.value = fmtNum(v);
      formValues[name] = Number(v);
    } else if (meta.kind === 'bin') {
      const cb = document.getElementById(`f-${name}`);
      const on = Number(v) === 1;
      if (cb) cb.checked = on;
      const stateLbl = document.getElementById(`state-${name}`);
      if (stateLbl) stateLbl.textContent = on ? 'Sì' : 'No';
      formValues[name] = on ? 1 : 0;
    } else if (meta.kind === 'seg') {
      const r = document.getElementById(`f-${name}-${Number(v)}`);
      if (r) r.checked = true;
      formValues[name] = Number(v);
    }
  }
  schedulePredict({ immediate: true });
}

function resetToTypical() {
  const row = {};
  for (const [name, meta] of Object.entries(FEATURE_META)) {
    if (meta.kind === 'num') row[name] = stats.continuous[name].median;
    else row[name] = stats.categorical[name].mode;
  }
  populateForm(row);
  $('#demo-picker').value = '';
}

// --- 7. orchestrazione ---------------------------------------------------

let predictTimer = null;
async function schedulePredict({ immediate = false } = {}) {
  if (predictTimer) clearTimeout(predictTimer);
  const delay = immediate ? 0 : 180;
  predictTimer = setTimeout(runPrediction, delay);
}

async function runPrediction() {
  if (!sessions.length) return;
  try {
    const zVec = readFormVector();
    const img = splat16x16(zVec);
    paintDeepInsight(img);
    const probRaw = await ensembleProb(img);
    const probCal = isotonicInterp(probRaw, calibration.isotonic);
    renderResult(probCal);
  } catch (e) {
    console.error(e);
    setStatus(`Qualcosa è andato storto: ${e.message}`, 'error');
  }
}

async function init() {
  try {
    await loadEverything();
    buildFeatureOrder();
    buildForm();
    buildDemoPicker();
    renderHeadline();
    renderIdleResult();
    $('#reset-btn').addEventListener('click', resetToTypical);
    schedulePredict({ immediate: true });
  } catch (e) {
    console.error(e);
    const status = $('#status');
    status.hidden = false;
    setStatus(`Impossibile caricare il modello: ${e.message}`, 'error');
  }
}

document.addEventListener('DOMContentLoaded', init);
})();
