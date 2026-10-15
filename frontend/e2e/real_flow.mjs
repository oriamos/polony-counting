// REAL end-to-end flow: real FastAPI backend + real Big-FISH + real Chromium UI.
//   cd frontend && npm run build && node e2e/real_flow.mjs
// Env: REAL_DATA_DIR (default: <scratch>/e2e-data/run-<ts>), REAL_PORT (8765), PY (pythonbin/python)
// Writes screenshots + results.json to e2e-artifacts/real/ (git-ignored).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '../..');
const OUT = path.resolve(__dirname, '../e2e-artifacts/real');
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const PY = process.env.PY || 'python';
const PORT = +(process.env.REAL_PORT || 8765);
const BASE = `http://127.0.0.1:${PORT}`;
const SCRATCH = process.env.REAL_SCRATCH || fs.mkdtempSync(path.join(os.tmpdir(), 'polony-e2e-'));
const DATA = process.env.REAL_DATA_DIR || path.join(SCRATCH, 'e2e-data', `run-${Date.now()}`);
const FIX = path.join(SCRATCH, 'fixtures');
fs.mkdirSync(DATA, { recursive: true });
fs.mkdirSync(FIX, { recursive: true });
const REAL_TIF = path.join(REPO, 'resources/dataset_files/test/test_img.tif');
const CHROME =
  process.env.CHROMIUM_PATH ||
  ['/opt/pw-browsers/chromium', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].find((p) => fs.existsSync(p));

const results = [];
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${info ? '  — ' + info : ''}`);
};
const sha256 = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ fixtures (synthetic two-file pair with edge/corner spots)
const TRUTH = {
  red: [[60, 50], [140, 80], [300, 60], [80, 220], [330, 250], [8, 8], [391, 291], [391, 150], [200, 8]],
  green: [[250, 150], [120, 160], [200, 250], [350, 130], [8, 291], [391, 8]],
  yellow: [[100, 120], [220, 80], [8, 150], [200, 292]],
};
function makeFixtures() {
  const code = `
import sys, json
sys.path.insert(0, ${JSON.stringify(path.join(REPO, 'tests/app'))})
import synthetic as s
T = json.loads(sys.argv[1])
red = [(x, y, 3000) for x, y in T['red']] + [(x + .4, y - .3, 3000) for x, y in T['yellow']]
green = [(x, y, 2800) for x, y in T['green']] + [(x - .4, y + .5, 2800) for x, y in T['yellow']]
sc = s.make_scene((300, 400), red, green, seed=7)
s.write_single(sys.argv[2], sc.red)
s.write_single(sys.argv[3], sc.green)
`;
  execFileSync(PY, ['-I', '-c', code, JSON.stringify(TRUTH), path.join(FIX, 'pair_chA.tif'), path.join(FIX, 'pair_chB.tif')], {
    env: { ...process.env, PYTHONPATH: path.join(REPO, 'src') },
  });
}
makeFixtures();

// ------------------------------------------------------------------ server control
let server = null;
async function startServer() {
  try {
    await fetch(`${BASE}/api/health`);
    throw new Error(`port ${PORT} already in use by another server; set REAL_PORT`);
  } catch (e) {
    if (/already in use/.test(String(e))) throw e;
  }
  const log = fs.openSync(path.join(OUT, 'server.log'), 'a');
  server = spawn(PY, ['-m', 'polony_app', '--data-dir', DATA, '--port', String(PORT)], {
    cwd: REPO,
    env: { ...process.env, PYTHONPATH: path.join(REPO, 'src') },
    stdio: ['ignore', log, log],
  });
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(`${BASE}/api/health`);
      if (r.ok) return await r.json();
    } catch {
      /* not yet */
    }
    await sleep(200);
  }
  throw new Error('server did not start');
}
async function stopServer() {
  if (!server) return;
  const p = server;
  const exited = new Promise((r) => p.once('exit', r));
  p.kill('SIGINT');
  const t = setTimeout(() => p.kill('SIGKILL'), 15000);
  await exited;
  clearTimeout(t);
  server = null;
  try {
    await fetch(`${BASE}/api/health`);
    return false;
  } catch {
    return true;
  }
}
const jget = async (u) => (await fetch(BASE + u)).json();
const getA = (id) => jget(`/api/analyses/${id}`);
const live = (a) => a.objects.filter((o) => !o.deleted);

// ------------------------------------------------------------------ browser helpers
const browser = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox'] });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
const page = await ctx.newPage();
page.setDefaultTimeout(20000);
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && pageErrors.push(m.text()));
const failedReqs = [];
page.on('response', (r) => {
  if (r.status() >= 400 && r.url().includes('/api/')) failedReqs.push(`${r.status()} ${r.request().method()} ${r.url().replace(BASE, '')}`);
});
const tid = (id) => page.locator(`[data-testid="${id}"]`);
const shot = (name) => page.screenshot({ path: path.join(OUT, name + '.png') });
const store = (fn, arg) => page.evaluate(([f, a]) => new Function('S', 'a', `return (${f})(S, a)`)(window.__polony.getState(), a), [fn.toString(), arg]);

async function waitImg() {
  await page.waitForFunction(() => document.querySelector('[data-testid=viewer]')?.dataset.imgReady === '1');
  await sleep(150);
}
async function viewState() {
  return page.evaluate(() => {
    const v = document.querySelector('[data-testid=viewer]');
    const r = v.getBoundingClientRect();
    return { s: +v.dataset.scale, tx: +v.dataset.tx, ty: +v.dataset.ty, left: r.left, top: r.top, w: r.width, h: r.height };
  });
}
async function toScreen(x, y) {
  const v = await viewState();
  return { x: v.left + v.tx + v.s * x, y: v.top + v.ty + v.s * y, v };
}
async function zoomTo(x, y, scale) {
  await page.evaluate(([x, y, s]) => {
    const S = window.__polony.getState();
    S.fit();
    window.__polony.getState().zoomToPoint(x, y, s);
  }, [x, y, scale]);
  await sleep(250);
}
async function setHidden(want) {
  const cur = (await tid('viewer').getAttribute('data-hidden')) === '1';
  if (cur !== want) {
    await page.keyboard.press('h');
    await page.waitForFunction((w) => document.querySelector('[data-testid=viewer]').dataset.hidden === (w ? '1' : '0'), want);
    await sleep(80);
  }
}
async function openAnalysis(id) {
  await page.goto(BASE + '/');
  await tid(`analysis-${id}`).click();
  await page.waitForSelector('[data-testid=viewer]');
  await waitImg();
}
async function uploadFiles(files, name) {
  await page.goto(BASE + '/');
  await tid('file-input').setInputFiles(files);
  if (name) await page.getByPlaceholder('Analysis name (optional)').fill(name);
  await page.getByRole('button', { name: 'Create analysis' }).click();
  await page.waitForSelector('[data-testid=mapping-dialog]');
  return page.evaluate(() => window.__polony.getState().current.id);
}
async function mapChannels(roles) {
  // roles: array of 'red'|'green'|'ignore' indexed by channel
  for (let i = 0; i < roles.length; i++) await tid(`role-${i}-${roles[i]}`).check();
  await tid('confirm-mapping').click();
  await page.waitForFunction(() => !document.querySelector('[data-testid=mapping-dialog]'));
  await sleep(300);
}
async function drawRect(tool, a, b) {
  await tid(tool).click();
  const p = await toScreen(...a);
  const q = await toScreen(...b);
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.move(q.x, q.y, { steps: 8 });
  await page.mouse.up();
  await sleep(500);
}
async function runDetection(id, timeoutMs = 600000) {
  await tid('tab-detect').click();
  await tid('run-detection').click();
  await page.waitForSelector('[data-testid=job]');
  const seen = new Set();
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const st = await page.locator('[data-testid=job]').getAttribute('data-status');
    const pb = page.locator('[role=progressbar]');
    if (await pb.count()) seen.add(+(await pb.first().getAttribute('aria-valuenow')));
    if (st === 'done' || st === 'failed' || st === 'cancelled') {
      const txt = await tid('job').innerText();
      return { status: st, seen: [...seen], txt, ms: Date.now() - t0 };
    }
    await sleep(100);
  }
  throw new Error('detection timeout');
}

/** Pixel measurements done inside the browser on SCREENSHOT pixels.
 *  ring: overlay ON over a black display (display-only lo/hi pushed to max) -> bbox centre of the near-white outline.
 *  blob: overlay OFF over the normal display -> intensity-weighted centroid inside a disc around the model position. */
async function clipFor(o) {
  const p = await toScreen(o.x, o.y);
  const v = p.v;
  const R = Math.round(Math.min(7.5 * v.s, v.h / 2 - 6, v.w / 2 - 6, 130));
  const x0 = Math.round(p.x - R), y0 = Math.round(p.y - R);
  const clip = { x: Math.max(v.left + 2, x0), y: Math.max(v.top + 2, y0), width: 2 * R, height: 2 * R };
  clip.width = Math.min(clip.width, v.left + v.w - 2 - clip.x);
  clip.height = Math.min(clip.height, v.top + v.h - 2 - clip.y);
  return { p, R, clip };
}
const PIX = `
  const load = async (b) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + b; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    return { d: g.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
  };`;
async function ringAt(o) {
  const { p, clip } = await clipFor(o);
  const buf = await page.screenshot({ clip });
  const r = await page.evaluate(
    new Function('args', `return (async () => { ${PIX}
      const I = await load(args.b64); const cx = args.px - args.clip.x, cy = args.py - args.clip.y;
      // any non-black pixel is overlay (the image is black). The review tick/dot sits in the upper-right of the ring:
      // take the horizontal extent from rows at/below the centre and the vertical extent from columns at/left of it.
      let x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1, n = 0;
      for (let y = 0; y < I.h; y++) for (let x = 0; x < I.w; x++) {
        if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) > 24) continue;
        const i = (y * I.w + x) * 4;
        if (Math.max(I.d[i], I.d[i + 1], I.d[i + 2]) >= 90) {
          n++;
          if (y + 0.5 >= cy - 2) { x0 = Math.min(x0, x); x1 = Math.max(x1, x + 1); }
          if (x + 0.5 <= cx + 2) { y0 = Math.min(y0, y); y1 = Math.max(y1, y + 1); }
        }
      }
      return n > 8 ? { x: args.clip.x + (x0 + x1) / 2, y: args.clip.y + (y0 + y1) / 2, n, w: x1 - x0 } : null; })()`),
    { b64: buf.toString('base64'), clip, px: p.x, py: p.y },
  );
  return { ring: r, p };
}
async function blobAt(o) {
  const { p, R, clip } = await clipFor(o);
  const buf = await page.screenshot({ clip });
  const r = await page.evaluate(
    new Function('args', `return (async () => { ${PIX}
      const I = await load(args.b64); const cx = args.px - args.clip.x, cy = args.py - args.clip.y; const rad = args.R * 0.85;
      const lum = (i) => (I.d[i] + I.d[i + 1] + I.d[i + 2]) / 3; const vals = [];
      for (let y = 0; y < I.h; y++) for (let x = 0; x < I.w; x++) if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= rad) vals.push(lum((y * I.w + x) * 4));
      vals.sort((a, b) => a - b); const bg = vals[Math.floor(vals.length * 0.3)] ?? 0;
      let sw = 0, sx = 0, sy = 0;
      for (let y = 0; y < I.h; y++) for (let x = 0; x < I.w; x++) { if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) > rad) continue;
        const w = Math.max(0, lum((y * I.w + x) * 4) - bg); sw += w; sx += w * (x + 0.5); sy += w * (y + 0.5); }
      return sw > 0 ? { x: args.clip.x + sx / sw, y: args.clip.y + sy / sw, w: sw } : null; })()`),
    { b64: buf.toString('base64'), clip, px: p.x, py: p.y, R },
  );
  return { spot: r, p };
}
async function viewerRect() {
  return page.evaluate(() => { const r = document.querySelector('[data-testid=viewer]').getBoundingClientRect(); return [r.left, r.top, r.width, r.height].map((v) => +v.toFixed(2)).join(','); });
}

async function alignmentRun(label, objs, scales, shotPrefix) {
  const rows = new Map();
  const key = (o, sc) => `${o.id}@${sc}`;
  await page.keyboard.press('Escape');
  await page.keyboard.press('3');
  await sleep(400);
  await waitImg();
  const rect0 = await viewerRect();
  // pass 1: rings on a black display (display-only sliders at maximum)
  await page.evaluate(() => window.__polony.getState().setDisplay({ lo: 65534, hi: 65535 }));
  await sleep(500);
  await waitImg();
  await setHidden(false);
  for (const sc of scales)
    for (const o of objs) {
      await zoomTo(o.x, o.y, sc);
      await sleep(120);
      const m = await ringAt(o);
      rows.set(key(o, sc), { label, id: o.id, cls: o.cls, x: o.x, y: o.y, scale: m.p.v.s, ring: m.ring, p: m.p });
    }
  await shot(`${shotPrefix}-black-display`);
  await page.evaluate(() => window.__polony.getState().resetDisplay());
  await sleep(500);
  await waitImg();
  // pass 2: blob centroid with overlays hidden
  await setHidden(true);
  for (const sc of scales)
    for (const o of objs) {
      await zoomTo(o.x, o.y, sc);
      await sleep(120);
      const m = await blobAt(o);
      const r = rows.get(key(o, sc));
      r.spot = m.spot;
      r.p2 = m.p;
    }
  await setHidden(false);
  const out = [];
  for (const r of rows.values()) {
    const s = r.scale;
    if (!r.spot || !r.ring) { out.push({ label, id: r.id, x: r.x, y: r.y, scale: s, note: `missing spot=${!!r.spot} ring=${!!r.ring}` }); continue; }
    if (Math.abs(r.p.x - r.p2.x) > 0.01 || Math.abs(r.p.y - r.p2.y) > 0.01) { out.push({ label, id: r.id, x: r.x, y: r.y, scale: s, note: 'view moved between passes' }); continue; }
    out.push({
      label, id: r.id, cls: r.cls, x: r.x, y: r.y, scale: s,
      ringModel: { dx: (r.ring.x - r.p.x) / s, dy: (r.ring.y - r.p.y) / s },
      ringImg: { dx: (r.ring.x - r.spot.x) / s, dy: (r.ring.y - r.spot.y) / s },
      imgModel: { dx: (r.spot.x - r.p.x) / s, dy: (r.spot.y - r.p.y) / s },
    });
  }
  const rect1 = await viewerRect();
  out.viewerRectStable = rect0 === rect1;
  return out;
}
const rowErr = (r, k) => (r[k] ? Math.hypot(r[k].dx, r[k].dy) : NaN);

// ================================================================== STEP 0: server up, health
process.on('exit', () => server && server.kill('SIGKILL'));
process.on('uncaughtException', (e) => { console.error(e); server && server.kill('SIGKILL'); process.exit(2); });
const health = await startServer();
check('real backend up', health.ok && health.bigfish_version, JSON.stringify(health));
const realHashBefore = sha256(REAL_TIF);

// ================================================================== STEP 1: upload + mapping (real image)
const rid = await uploadFiles([REAL_TIF], 'real test_img');
check('upload creates analysis needing mapping', (await getA(rid)).status === 'needs_channel_mapping', `id=${rid}`);
{
  const dlg = await tid('mapping-dialog').innerText();
  check("page 0 label hint '635 [W1]' shown as hint only", dlg.includes('635 [W1]') && /hint/i.test(dlg));
  check('no role preselected, confirm disabled until explicit choice', (await page.locator('[data-testid=mapping-dialog] input[type=radio]:checked').count()) === 0 && (await tid('confirm-mapping').isDisabled()));
  await sleep(1500);
  const thumbs = await page.evaluate(() => [...document.querySelectorAll('[data-testid=mapping-dialog] .thumb-box img')].map((i) => [i.complete, i.naturalWidth, getComputedStyle(i).display]));
  check('per-channel previews (/image/chN) load before mapping is confirmed', thumbs.length === 2 && thumbs.every((t) => t[0] && t[1] > 0 && t[2] !== 'none'), JSON.stringify(thumbs));
  await shot('01-mapping-dialog');
  const a0 = await getA(rid);
  check('backend: no role assigned/confirmed before the user acts', a0.image.channels.every((c) => c.role === null && !c.role_confirmed));
  await mapChannels(['red', 'green']);
  const a1 = await getA(rid);
  check('explicit mapping stored (ch0=red, ch1=green)', a1.status === 'ready' && a1.image.channels.map((c) => c.role).join() === 'red,green' && a1.image.channels.every((c) => c.role_confirmed));
}
await waitImg();
await shot('02-fit-composite');

// ================================================================== STEP 2: regions
await tid('tab-regions').click();
await drawRect('tool-draw-count-rect', [250, 300], [1250, 2100]);
let A = await getA(rid);
const creg = A.regions.find((r) => r.kind === 'counting');
{
  const xs = creg?.polygon.map((p) => p[0]) ?? [], ys = creg?.polygon.map((p) => p[1]) ?? [];
  const bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  check('counting rectangle drawn in UI (target 250,300-1250,2100)', creg && Math.abs(bb[0] - 250) < 6 && Math.abs(bb[1] - 300) < 6 && Math.abs(bb[2] - 1250) < 6 && Math.abs(bb[3] - 2100) < 6, JSON.stringify(bb.map((v) => +v.toFixed(1))));
}
await drawRect('tool-draw-excl-rect', [450, 600], [650, 800]);
A = await getA(rid);
const xreg = A.regions.find((r) => r.kind === 'exclusion');
check('exclusion rectangle drawn in UI', !!xreg && A.regions.length === 2, xreg ? JSON.stringify(xreg.polygon.map((p) => p.map((v) => +v.toFixed(0)))) : 'none');
await shot('03-regions');

// ================================================================== STEP 3: preview + full detection
await tid('tab-detect').click();
await tid('preview-btn').click();
{
  const a = await toScreen(300, 350), b = await toScreen(700, 750);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForSelector('[data-testid=preview-info]', { timeout: 60000 });
  await sleep(300);
  const n = await page.locator('[data-testid=preview-layer] > g').count();
  const info = await tid('preview-info').innerText();
  const A2 = await getA(rid);
  check('detect preview on a window shows temporary overlays and persists nothing', n >= 1 && A2.objects.length === 0, `${n} preview glyphs; info="${info.replace(/\n/g, ' ')}"`);
  await shot('04-preview');
}
const det = await runDetection(rid);
check('full real Big-FISH detection completes with progress shown', det.status === 'done', `seen progress values ${JSON.stringify(det.seen)}; ${det.ms} ms; "${det.txt.replace(/\n/g, ' ')}"`);
A = await getA(rid);
const nDet = live(A).length;
check('analysis in_review, objects present, not stale, package versions recorded', A.status === 'in_review' && nDet > 0 && !A.provenance.stale_detection && Object.keys(A.provenance.package_versions || {}).length > 0, `status=${A.status} live=${nDet} versions=${JSON.stringify(A.provenance.package_versions)}`);
check('no object outside counting region / inside exclusion is counted', live(A).every((o) => o.region_id === creg.id || o.region_id == null));
await tid('tab-review').click();
await page.keyboard.press('f');
await sleep(300);
await shot('05-detected-fit');
{
  const glyphs = await page.locator('[data-testid=objects-layer] g[data-cls]').count();
  check('overlays visible after detection', glyphs > 0, `${glyphs} glyphs rendered at fit zoom for ${nDet} objects`);
}

// ================================================================== STEP 4: edits (add/delete/move/reclassify/merge/split/undo/redo)
async function waitObjs(pred, ms = 6000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const a = await getA(rid);
    if (pred(a)) return a;
    await sleep(120);
  }
  return getA(rid);
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
{
  const cr = creg.polygon;
  const x0 = Math.min(...cr.map((p) => p[0])), x1 = Math.max(...cr.map((p) => p[0])), y0 = Math.min(...cr.map((p) => p[1])), y1 = Math.max(...cr.map((p) => p[1]));
  let free = null;
  for (let y = y0 + 150; y < y1 - 100 && !free; y += 40)
    for (let x = x0 + 100; x < x1 - 100 && !free; x += 40) {
      const ex = xreg.polygon;
      const inEx = x > Math.min(...ex.map((p) => p[0])) - 60 && x < Math.max(...ex.map((p) => p[0])) + 60 && y > Math.min(...ex.map((p) => p[1])) - 60 && y < Math.max(...ex.map((p) => p[1])) + 60;
      if (!inEx && live(A).every((o) => dist(o, { x, y }) > 40)) free = { x, y };
    }
  check('found free spot for manual add', !!free, JSON.stringify(free));
  // ---- add
  await zoomTo(free.x, free.y, 6);
  await page.keyboard.press('a');
  await tid('addcls-green').click();
  const ps = await toScreen(free.x, free.y);
  await page.mouse.click(ps.x, ps.y);
  A = await waitObjs((a) => live(a).length === nDet + 1);
  const added = live(A).find((o) => o.origin === 'manual');
  check('ADD: click adds a manual green spot at the clicked coordinate', live(A).length === nDet + 1 && added && added.cls === 'green' && dist(added, free) < 0.6 && added.region_id === creg.id, JSON.stringify(added && [added.x, added.y, added.cls, added.region_id]));
  await page.keyboard.press('v');
  // ---- move (drag)
  const from = await toScreen(added.x, added.y);
  const to = await toScreen(added.x + 10, added.y + 6);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 8 });
  await page.mouse.up();
  A = await waitObjs((a) => { const o = a.objects.find((q) => q.id === added.id); return o && dist(o, added) > 5; });
  const moved = A.objects.find((o) => o.id === added.id);
  check('MOVE: dragging moves the spot by ~(+10,+6) image px', dist(moved, { x: added.x + 10, y: added.y + 6 }) < 1.0, `now (${moved.x.toFixed(2)},${moved.y.toFixed(2)}) from (${added.x.toFixed(2)},${added.y.toFixed(2)})`);
  // ---- reclassify via inspector button and hotkey
  await page.waitForSelector('[data-testid=inspector]');
  await tid('cls-yellow').click();
  A = await waitObjs((a) => a.objects.find((o) => o.id === added.id).cls === 'yellow');
  check('RECLASSIFY: inspector button -> yellow', A.objects.find((o) => o.id === added.id).cls === 'yellow');
  await page.keyboard.press('r');
  A = await waitObjs((a) => a.objects.find((o) => o.id === added.id).cls === 'red');
  check('RECLASSIFY: hotkey R -> red', A.objects.find((o) => o.id === added.id).cls === 'red');
  await shot('06-edit-inspector');
  // ---- delete + undo + redo
  const nBefore = live(A).length;
  await page.keyboard.press('Delete');
  A = await waitObjs((a) => live(a).length === nBefore - 1);
  check('DELETE: Del key soft-deletes the selected spot (excluded from counts)', live(A).length === nBefore - 1 && A.objects.find((o) => o.id === added.id).deleted === true);
  const cDel = await jget(`/api/analyses/${rid}/counts`);
  await page.keyboard.press('Control+z');
  A = await waitObjs((a) => live(a).length === nBefore);
  check('UNDO restores the deleted spot', live(A).length === nBefore);
  await page.keyboard.press('Control+Shift+Z');
  A = await waitObjs((a) => live(a).length === nBefore - 1);
  check('REDO re-applies the delete', live(A).length === nBefore - 1 && (await jget(`/api/analyses/${rid}/counts`)).slide.total_objects_exclusive === cDel.slide.total_objects_exclusive);
  // ---- reclassify a detected spot
  const target = live(A).find((o) => o.origin === 'auto' && o.cls === 'red');
  await zoomTo(target.x, target.y, 8);
  const pt = await toScreen(target.x, target.y);
  await page.mouse.click(pt.x, pt.y);
  await page.waitForSelector('[data-testid=inspector]');
  await page.keyboard.press('g');
  A = await waitObjs((a) => a.objects.find((o) => o.id === target.id).cls === 'green');
  const t2 = A.objects.find((o) => o.id === target.id);
  check('RECLASSIFY detected spot red->green keeps origin/evidence and records history', t2.cls === 'green' && t2.red.detected === target.red.detected && t2.history.length > target.history.length, `origin=${t2.origin} history=${JSON.stringify(t2.history.slice(-2))}`);
  // ---- merge two nearby detected spots
  const cand = live(A).filter((o) => o.origin === 'auto' && o.id !== target.id);
  let best = null;
  for (let i = 0; i < cand.length; i++)
    for (let j = i + 1; j < cand.length; j++) {
      const d = dist(cand[i], cand[j]);
      if (d >= 9 && (!best || d < best.d)) best = { a: cand[i], b: cand[j], d };
    }
  check('found a nearby pair to merge', !!best, best ? `distance ${best.d.toFixed(1)} px` : '');
  const mid = { x: (best.a.x + best.b.x) / 2, y: (best.a.y + best.b.y) / 2 };
  await zoomTo(mid.x, mid.y, 12);
  const pa = await toScreen(best.a.x, best.a.y), pb = await toScreen(best.b.x, best.b.y);
  const nPre = live(A).length;
  await page.mouse.click(pa.x, pa.y);
  await page.keyboard.down('Shift');
  await page.mouse.click(pb.x, pb.y);
  await page.keyboard.up('Shift');
  await page.keyboard.press('m');
  A = await waitObjs((a) => live(a).length === nPre - 1);
  const merged = live(A).find((o) => o.history.some((h) => /merged/i.test(h)));
  check('MERGE: two spots become one at their centroid', live(A).length === nPre - 1 && merged && dist(merged, mid) < 1.0, merged ? `merged at (${merged.x.toFixed(1)},${merged.y.toFixed(1)}) vs midpoint (${mid.x.toFixed(1)},${mid.y.toFixed(1)}); sources deleted: ${A.objects.find((o) => o.id === best.a.id).deleted && A.objects.find((o) => o.id === best.b.id).deleted}` : 'none');
  // ---- split
  await zoomTo(merged.x, merged.y, 12);
  const pm = await toScreen(merged.x, merged.y);
  await page.mouse.click(pm.x, pm.y);
  await page.waitForSelector('[data-testid=inspector]');
  await page.keyboard.press('s');
  const s1 = await toScreen(merged.x - 5, merged.y), s2 = await toScreen(merged.x + 5, merged.y);
  await page.mouse.click(s1.x, s1.y);
  await page.mouse.click(s2.x, s2.y);
  A = await waitObjs((a) => live(a).length === nPre);
  const kids = live(A).filter((o) => o.history.some((h) => /split/i.test(h)));
  check('SPLIT: one spot becomes two at the clicked positions', live(A).length === nPre && kids.length === 2 && kids.some((k) => dist(k, { x: merged.x - 5, y: merged.y }) < 1) && kids.some((k) => dist(k, { x: merged.x + 5, y: merged.y }) < 1), kids.map((k) => `(${k.x.toFixed(1)},${k.y.toFixed(1)},${k.cls})`).join(' '));
  await shot('07-after-split');
  A = await getA(rid);
  check('undo stack recorded all edits', A.undo_depth >= 7, `undo_depth=${A.undo_depth} redo_depth=${A.redo_depth}`);
  // undo whole chain of merge+split then redo them (verifies multi-step undo/redo)
  const snapIds = JSON.stringify(live(A).map((o) => [o.id, o.x, o.y, o.cls]));
  await page.keyboard.press('Control+z');
  await page.keyboard.press('Control+z');
  await sleep(700);
  const mid1 = await getA(rid);
  await page.keyboard.press('Control+Shift+Z');
  await page.keyboard.press('Control+Shift+Z');
  await sleep(700);
  const A3 = await getA(rid);
  check('UNDO x2 then REDO x2: set changes then is restored identically', JSON.stringify(live(mid1).map((o) => [o.id, o.x, o.y, o.cls])) !== snapIds && JSON.stringify(live(A3).map((o) => [o.id, o.x, o.y, o.cls])) === snapIds, `live after undo x2=${live(mid1).length}, after redo x2=${live(A3).length}`);
}

// ================================================================== STEP 5: review + inspect + accept + approve + finalize
{
  await tid('tab-review').click();
  A = await getA(rid);
  const unc = live(A).filter((o) => o.cls === 'uncertain');
  const queue = await jget(`/api/analyses/${rid}/review-queue`);
  check('review queue lists flagged objects with reasons (UI count == API count)', (await page.locator('[data-testid^="queue-o"]').count()) === queue.length && queue.every((q) => q.reasons.length > 0), `uncertain=${unc.length}, queue=${queue.length}, reasons=${JSON.stringify([...new Set(queue.flatMap((q) => q.reasons))])}`);
  // resolve uncertain objects using their own evidence (both channels detected -> yellow, else the detecting channel)
  for (const o of unc) {
    const cls = o.red.detected && o.green.detected ? 'yellow' : o.green.detected ? 'green' : 'red';
    await zoomTo(o.x, o.y, 8);
    const p = await toScreen(o.x, o.y);
    await page.mouse.click(p.x, p.y);
    await page.waitForSelector('[data-testid=inspector]');
    await tid(`cls-${cls}`).click();
    await sleep(250);
  }
  await page.keyboard.press('Escape');
  await tid('tab-regions').click();
  // approve before inspecting must be refused with a human-readable reason
  await page.locator(`[data-testid=approve-${creg.id}]`).click();
  await page.waitForSelector(`[data-testid=region-error-${creg.id}]`);
  const refusal = await tid(`region-error-${creg.id}`).innerText();
  check('approval refusal (not inspected) shows a readable reason', /inspect/i.test(refusal) && refusal.length > 20, refusal);
  // finalize blockers visible
  await tid('tab-finish').click();
  const bad = await page.locator('[data-testid=checklist] li.bad').count();
  check('finalize disabled with blockers listed before approval', (await tid('finalize').isDisabled()) && bad >= 2, `${bad} unmet checklist items: ` + (await page.locator('[data-testid=checklist] li.bad').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ')).join(' | '));
  await shot('08-finish-blocked');
  // DRAFT export labelling
  const dr = await fetch(`${BASE}/api/analyses/${rid}/export?kind=objects`);
  const drt = await dr.text();
  check('pre-final export is DRAFT (filename + status column)', /DRAFT/.test(dr.headers.get('content-disposition')) && drt.split('\n')[1].trim().endsWith('DRAFT'), dr.headers.get('content-disposition'));
  // inspect, accept, approve
  await tid('tab-regions').click();
  await page.locator(`[data-testid=inspected-${creg.id}]`).click();
  await page.waitForFunction((i) => document.querySelector(`[data-testid="inspected-${i}"]`).checked, creg.id);
  await sleep(400);
  await page.locator(`[data-testid=accept-${creg.id}]`).click();
  await sleep(500);
  await page.locator(`[data-testid=approve-${creg.id}]`).click();
  await sleep(600);
  A = await getA(rid);
  const cr = A.regions.find((r) => r.id === creg.id);
  check('region inspected + accepted (all counted objects reviewed) + approved', cr.inspected && cr.approved && live(A).filter((o) => o.region_id === creg.id && !o.reviewed).length === 0, `reviewed ${live(A).filter((o) => o.reviewed).length}/${live(A).length}`);
  await tid('tab-finish').click();
  check('finalize enabled when no blockers', await tid('finalize').isEnabled());
  await tid('finalize').click();
  await page.waitForSelector('[data-testid=reopen]');
  A = await getA(rid);
  check('finalized: status finalized, finalized_at set, FINAL badge', A.status === 'finalized' && !!A.finalized_at && (await tid('draft-badge').first().innerText()) === 'FINAL');
  await shot('09-finalized');
}

// ================================================================== STEP 6: synthetic two-file pair (mapping swap, detection, alignment at edges/corners)
const pairFiles = [path.join(FIX, 'pair_chA.tif'), path.join(FIX, 'pair_chB.tif')];
const pid = await uploadFiles(pairFiles, 'synthetic pair');
{
  const dlg = await tid('mapping-dialog').innerText();
  check('pair: mapping needed, nothing preselected from file names', (await page.locator('[data-testid=mapping-dialog] input[type=radio]:checked').count()) === 0 && /pair_chA/.test(dlg));
  await mapChannels(['red', 'green']); // chA is (intended) red
  await waitImg();
  await tid('tab-regions').click();
  await drawRect('tool-draw-count-rect', [1, 1], [399, 299]);
  const Ap = await getA(pid);
  check('pair: whole-image counting region drawn', Ap.regions.length === 1, JSON.stringify(Ap.regions[0]?.polygon));
  // brightness of viewer image at ground-truth spots
  await setHidden(true);
  async function brightAt(view, pts) {
    await page.keyboard.press({ red: '1', green: '2', composite: '3' }[view]);
    await sleep(500);
    await waitImg();
    const out = [];
    for (const [x, y] of pts) {
      const p = await toScreen(x, y);
      const buf = await page.screenshot({ clip: { x: Math.round(p.x - 3), y: Math.round(p.y - 3), width: 6, height: 6 } });
      out.push(await page.evaluate(async (b64) => {
        const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
        const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data; let r = 0, gg = 0, b = 0, n = 0;
        for (let i = 0; i < d.length; i += 4) { r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++; }
        return [r / n, gg / n, b / n];
      }, buf.toString('base64')));
    }
    return out;
  }
  const mean = (a) => a.reduce((s, v) => s + v[0] + v[1] + v[2], 0) / (3 * a.length);
  await page.keyboard.press('f');
  await sleep(300);
  const redOnly = TRUTH.red.slice(0, 5), greenOnly = TRUTH.green.slice(0, 4);
  const m1 = { rR: mean(await brightAt('red', redOnly)), rG: mean(await brightAt('red', greenOnly)), gR: mean(await brightAt('green', redOnly)), gG: mean(await brightAt('green', greenOnly)) };
  check('mapping A=red,B=green: RED view shows A (red-only spots bright, green-only dark); GREEN view the opposite', m1.rR > m1.rG + 60 && m1.gG > m1.gR + 60, JSON.stringify(m1, (k, v) => (typeof v === 'number' ? +v.toFixed(1) : v)));
  const compo = await brightAt('composite', redOnly);
  check('composite: red-only spot is red-dominant', compo.every((c) => c[0] > c[1] + 30), JSON.stringify(compo.map((c) => c.map(Math.round))));
  await setHidden(false);
  // swap mapping through the UI
  await tid('tab-detect').click();
  await tid('edit-mapping').click();
  await mapChannels(['green', 'red']);
  await waitImg();
  await setHidden(true);
  await page.keyboard.press('f');
  await sleep(300);
  const m2 = { rR: mean(await brightAt('red', redOnly)), rG: mean(await brightAt('red', greenOnly)), gR: mean(await brightAt('green', redOnly)), gG: mean(await brightAt('green', greenOnly)) };
  check('after SWAPPING mapping (A=green,B=red) the RED view now shows B and GREEN view shows A', m2.rG > m2.rR + 60 && m2.gR > m2.gG + 60, JSON.stringify(m2, (k, v) => (typeof v === 'number' ? +v.toFixed(1) : v)));
  const compo2 = await brightAt('composite', redOnly);
  check('composite follows the swap (A spots now green-dominant)', compo2.every((c) => c[1] > c[0] + 30), JSON.stringify(compo2.map((c) => c.map(Math.round))));
  await setHidden(false);
  await shot('10-pair-swapped-composite');
  await tid('edit-mapping').click();
  await mapChannels(['red', 'green']);
  await waitImg();
  const det2 = await runDetection(pid, 300000);
  check('pair: Big-FISH detection done', det2.status === 'done', det2.txt.replace(/\n/g, ' '));
  const Ad = await getA(pid);
  const truthAll = [...TRUTH.red, ...TRUTH.green, ...TRUTH.yellow];
  const found = live(Ad);
  const nearest = found.map((o) => Math.min(...truthAll.map((t) => dist(o, { x: t[0], y: t[1] }))));
  const byClass = found.reduce((m, o) => ((m[o.cls] = (m[o.cls] || 0) + 1), m), {});
  check('pair: detections vs ground truth (software check only)', found.length > 0, `found ${found.length} (truth ${TRUTH.red.length + TRUTH.green.length + TRUTH.yellow.length}); by class ${JSON.stringify(byClass)}; max nearest-truth distance ${Math.max(...nearest).toFixed(2)} px`);
  const missing = truthAll.filter((t) => !found.some((o) => dist(o, { x: t[0], y: t[1] }) < 3));
  check('pair: every ground-truth spot (incl. corners/edges) has a detection within 3 px', missing.length === 0, missing.length ? `missing ${JSON.stringify(missing)}` : 'all matched');
  await page.keyboard.press('f');
  await shot('11-pair-detected');
  // ---- ALIGNMENT, pair: corners/edges + interior, two zoom levels
  const pick = (t) => found.filter((o) => dist(o, { x: t[0], y: t[1] }) < 3).sort((a, b) => dist(a, { x: t[0], y: t[1] }) - dist(b, { x: t[0], y: t[1] }))[0];
  const edgeTruth = [[8, 8], [391, 291], [8, 291], [391, 8], [8, 150], [391, 150], [200, 8], [200, 292], [100, 120], [60, 50], [330, 250]];
  const targets = edgeTruth.map(pick).filter(Boolean);
  await tid('labels-mode').selectOption('off');
  await page.keyboard.press('Escape');
  const rowsPair = await alignmentRun('pair', targets, [6, 12], 'align-pair');
  check('layout stable: toggling overlays / display does not move the viewer', rowsPair.viewerRectStable, '');
  globalThis.ALIGN = { pair: rowsPair };
}

// ================================================================== STEP 7: snapshot, STOP server, restart, verify identical state
const before = { real: await getA(rid), pair: await getA(pid) };
const beforeCounts = { real: await jget(`/api/analyses/${rid}/counts`), pair: await jget(`/api/analyses/${pid}/counts`) };
const beforeList = await jget('/api/analyses');
const stopped = await stopServer();
check('server stopped', stopped);
await startServer();
await page.goto(BASE + '/');
await page.waitForSelector(`[data-testid=analysis-${rid}]`);
const after = { real: await getA(rid), pair: await getA(pid) };
check('RESTART: real analysis JSON (objects, regions, approvals, history, finalized) byte-identical', JSON.stringify(after.real) === JSON.stringify(before.real), `${live(after.real).length} live objects, finalized_at=${after.real.finalized_at}`);
check('RESTART: pair analysis JSON identical', JSON.stringify(after.pair) === JSON.stringify(before.pair));
check('RESTART: counts identical', JSON.stringify(await jget(`/api/analyses/${rid}/counts`)) === JSON.stringify(beforeCounts.real));
const afterList = await jget('/api/analyses');
check('RESTART: analyses list identical (status, counts)', JSON.stringify(afterList.map((x) => [x.id, x.status, x.finalized_at, x.counts])) === JSON.stringify(beforeList.map((x) => [x.id, x.status, x.finalized_at, x.counts])), afterList.map((x) => `${x.id}:${x.status}:${JSON.stringify(x.counts && [x.counts.red_only, x.counts.green_only, x.counts.yellow])}`).join(' '));
await openAnalysis(rid);
{
  const ui = await store((S) => ({ status: S.current.status, n: S.current.objects.filter((o) => !o.deleted).length, regs: S.current.regions.length, appr: S.current.regions.filter((r) => r.approved).length }));
  check('RESTART: reopened analysis in UI shows same state', ui.status === 'finalized' && ui.n === live(before.real).length && ui.regs === 2 && ui.appr === 1, JSON.stringify(ui));
  await tid('tab-finish').click();
  check('RESTART: FINAL badge + Reopen available', (await tid('draft-badge').first().innerText()) === 'FINAL' && (await tid('reopen').count()) === 1);
  await shot('12-after-restart');
}

// ================================================================== STEP 8: exports via UI links + validation
const dlDir = path.join(OUT, 'downloads');
fs.mkdirSync(dlDir, { recursive: true });
const files = {};
for (const k of ['objects', 'regions', 'slide', 'provenance', 'actions', 'history', 'bundle']) {
  const [dl] = await Promise.all([page.waitForEvent('download'), tid(`export-${k}`).click()]);
  const p = path.join(dlDir, dl.suggestedFilename());
  await dl.saveAs(p);
  files[k] = p;
  check(`export ${k}: downloaded via UI link, filename carries FINAL, no DRAFT`, /_FINAL\./.test(dl.suggestedFilename()) && !/DRAFT/.test(dl.suggestedFilename()), dl.suggestedFilename());
}
const parseCsv = (txt) => {
  const rows = []; let row = [], cur = '', q = false;
  for (let i = 0; i < txt.length; i++) {
    const c = txt[i];
    if (q) { if (c === '"') { if (txt[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cur); cur = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && txt[i + 1] === '\n') i++; row.push(cur); rows.push(row); row = []; cur = ''; }
    else cur += c;
  }
  if (cur || row.length) { row.push(cur); rows.push(row); }
  const h = rows[0];
  return rows.slice(1).filter((r) => r.length > 1).map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
};
{
  const obj = parseCsv(fs.readFileSync(files.objects, 'utf8'));
  const exp = live(after.real);
  check('objects.csv: one row per non-deleted object, all status FINAL, ids match', obj.length === exp.length && obj.every((r) => r.status === 'FINAL') && new Set(obj.map((r) => r.id)).size === new Set(exp.map((o) => o.id)).size, `${obj.length} rows vs ${exp.length} live objects`);
  check('objects.csv: per-channel evidence columns present', ['red_detected', 'red_peak', 'green_snr', 'association_distance_px', 'review_reasons', 'reviewed', 'counted'].every((c) => c in obj[0]));
  const byCls = obj.reduce((m, r) => ((m[r.class] = (m[r.class] || 0) + 1), m), {});
  const sl = parseCsv(fs.readFileSync(files.slide, 'utf8'))[0];
  check('slide.csv: both counting conventions as unambiguous columns', ['red_only', 'green_only', 'yellow', 'uncertain', 'red_positive_incl_yellow', 'green_positive_incl_yellow', 'yellow_overlap', 'total_objects_exclusive'].every((c) => c in sl), Object.keys(sl).slice(0, 14).join(','));
  const c = after.real && (await jget(`/api/analyses/${rid}/counts`)).slide;
  const counted = obj.filter((r) => r.counted === 'true');
  const cc = counted.reduce((m, r) => ((m[r.class] = (m[r.class] || 0) + 1), m), {});
  check('slide.csv numbers == API counts == recount of objects.csv(counted=true)', +sl.red_only === c.red_only && +sl.green_only === c.green_only && +sl.yellow === c.yellow && +sl.red_only === (cc.red || 0) && +sl.green_only === (cc.green || 0) && +sl.yellow === (cc.yellow || 0) && +sl.red_positive_incl_yellow === +sl.red_only + +sl.yellow && +sl.green_positive_incl_yellow === +sl.green_only + +sl.yellow, `red_only=${sl.red_only} green_only=${sl.green_only} yellow=${sl.yellow} uncertain=${sl.uncertain} red+=${sl.red_positive_incl_yellow} green+=${sl.green_positive_incl_yellow} total_excl=${sl.total_objects_exclusive}; all rows by class ${JSON.stringify(byCls)}`);
  check('slide.csv: status FINAL, no concentration values when inputs not provided', sl.status === 'FINAL' && Object.keys(sl).filter((k) => k.startsWith('est_conc')).every((k) => sl[k] === '') && /^inputs incomplete/.test(sl.concentration_status), `concentration_status="${sl.concentration_status}"`);
  const rg = parseCsv(fs.readFileSync(files.regions, 'utf8'));
  check('regions.csv: counting + exclusion rows, approval recorded, FINAL', rg.length === 2 && rg.some((r) => r.kind === 'counting' && r.approval_status === 'approved' && r.inspected === 'true') && rg.every((r) => r.status === 'FINAL'), JSON.stringify(rg.map((r) => [r.kind, r.area_px2, r.approval_status])));
  const prov = JSON.parse(fs.readFileSync(files.provenance, 'utf8'));
  const sh = prov.image.sha256;
  const shv = typeof sh === 'string' ? [sh] : Object.values(sh);
  check('provenance: image sha256 equals sha256 of the original file', shv.includes(realHashBefore), `provenance=${JSON.stringify(sh)} file=${realHashBefore}`);
  check('original bytes unchanged (source file and stored copy)', sha256(REAL_TIF) === realHashBefore);
  const stored = fs.readdirSync(path.join(DATA, 'originals', rid)).map((f) => sha256(path.join(DATA, 'originals', rid, f)));
  check('stored original copy in data dir has identical hash', stored.includes(realHashBefore), `${stored.length} stored file(s)`);
  check('provenance: status FINAL, channel mapping with roles confirmed, label_hint kept as hint', prov.status === 'FINAL' && prov.image.channel_mapping.map((c) => c.role).join() === 'red,green' && prov.image.channel_mapping.every((c) => c.role_confirmed) && prov.image.channel_mapping[0].label_hint === '635 [W1]');
  check('provenance: package versions incl. big-fish, detection params, schema/app version, regions', Object.keys(prov.package_versions_at_detection).some((k) => /fish/i.test(k)) && prov.detection_parameters_used && prov.schema_version && prov.app_version && prov.regions.length === 2, JSON.stringify(prov.package_versions_at_detection));
  check('provenance: no concentration inputs invented (slide fields null)', prov.slide.sample_volume_ul == null && prov.slide.concentration_factor == null);
  const zip = execFileSync(PY, ['-I', '-c', 'import zipfile,sys;z=zipfile.ZipFile(sys.argv[1]);print("\\n".join(z.namelist()));print("----");print(z.read([n for n in z.namelist() if "README" in n][0]).decode()[:200])', files.bundle]).toString();
  check('bundle zip lists csv/json files + README marked FINAL', /objects/.test(zip) && /provenance/.test(zip) && /README/i.test(zip) && /FINAL/.test(zip.split('----')[1] || ''), zip.split('\n').slice(0, 8).join(' | '));
}

// ================================================================== STEP 9: alignment on the real image (post restart, finalized view)
{
  await openAnalysis(rid);
  await tid('labels-mode').selectOption('off');
  const objs = live(after.real);
  const iso = objs.filter((o) => objs.every((q) => q === o || dist(o, q) > 30));
  // spread: pick near the 4 region corners + centre among isolated objects
  const cr = after.real.regions.find((r) => r.kind === 'counting').polygon;
  const cx = (Math.min(...cr.map((p) => p[0])) + Math.max(...cr.map((p) => p[0]))) / 2, cy = (Math.min(...cr.map((p) => p[1])) + Math.max(...cr.map((p) => p[1]))) / 2;
  const ex = after.real.regions.find((r) => r.kind === 'exclusion').polygon;
  const exMid = ex.reduce((m, p) => [m[0] + p[0] / ex.length, m[1] + p[1] / ex.length], [0, 0]);
  const anchors = [[Math.min(...cr.map((p) => p[0])), Math.min(...cr.map((p) => p[1]))], [Math.max(...cr.map((p) => p[0])), Math.min(...cr.map((p) => p[1]))], [Math.min(...cr.map((p) => p[0])), Math.max(...cr.map((p) => p[1]))], [Math.max(...cr.map((p) => p[0])), Math.max(...cr.map((p) => p[1]))], [cx, cy], exMid];
  const chosen = [];
  for (const a of anchors) {
    const c = iso.filter((o) => !chosen.includes(o)).sort((p, q) => dist(p, { x: a[0], y: a[1] }) - dist(q, { x: a[0], y: a[1] }))[0];
    if (c) chosen.push(c);
  }
  const rowsReal = await alignmentRun('real', chosen, [6, 12], 'align-real');
  globalThis.ALIGN.real = rowsReal;
  await tid('tab-review').click().catch(() => {});
  // failure-mode screenshots on the real image
  await page.keyboard.press('f');
  await page.keyboard.press('3');
  await sleep(600);
  await shot('real-fit-composite-overlay');
  await page.keyboard.press('h');
  await sleep(300);
  await shot('real-fit-composite-clean');
  await page.keyboard.press('h');
  for (const [k, n] of [['1', 'red'], ['2', 'green'], ['4', 'processed-red'], ['5', 'processed-green']]) {
    await page.keyboard.press(k);
    await sleep(900);
    await waitImg();
    await shot(`real-fit-${n}`);
  }
  await page.keyboard.press('3');
  await waitImg();
  for (const [i, o] of chosen.slice(0, 4).entries()) {
    await zoomTo(o.x, o.y, 10);
    await sleep(300);
    await shot(`real-spot-${i}-${o.cls}`);
  }
  const bad = [];
  const alignSummary = {};
  for (const [name, rows] of Object.entries(globalThis.ALIGN)) {
    const ok = rows.filter((r) => r.ringModel);
    const miss = rows.filter((r) => !r.ringModel);
    const mx = (k) => Math.max(...ok.map((r) => rowErr(r, k)));
    alignSummary[name] = { n: ok.length, missing: miss.length, scales: [...new Set(ok.map((r) => +r.scale.toFixed(1)))], maxRingVsModel: mx('ringModel'), maxRingVsModelScreenPx: Math.max(...ok.map((r) => rowErr(r, 'ringModel') * r.scale)), maxRingVsImage: mx('ringImg'), maxImageVsModel: mx('imgModel') };
    for (const r of ok) if (rowErr(r, 'ringModel') * r.scale > 1) bad.push(`${name} ${r.id} ringModel ${rowErr(r, 'ringModel').toFixed(3)}`);
    fs.writeFileSync(path.join(OUT, `alignment-${name}.json`), JSON.stringify(rows, null, 1));
    console.log(`ALIGNMENT ${name}:`, JSON.stringify(alignSummary[name]));
    for (const r of rows) console.log(`   ${r.id} ${r.cls ?? ''} (${r.x?.toFixed?.(1)},${r.y?.toFixed?.(1)}) zoom ${r.scale?.toFixed?.(1)}: ` + (r.ringModel ? `ring-vs-model ${rowErr(r, 'ringModel').toFixed(3)} px, ring-vs-blob ${rowErr(r, 'ringImg').toFixed(3)} px, blob-vs-model ${rowErr(r, 'imgModel').toFixed(3)} px` : r.note));
  }
  const P = alignSummary.pair, R = alignSummary.real;
  check('ALIGNMENT pair (corners/edges, zoom 6 and 12): overlay ring centre vs its model coordinate <= 1 screen px (measurement quantisation)', P.n >= 16 && P.missing === 0 && P.maxRingVsModelScreenPx <= 1.0, JSON.stringify(P));
  check('ALIGNMENT pair: overlay ring centre vs bright-blob centroid under it <= 0.75 image px (includes detector sub-pixel centre error)', P.maxRingVsImage <= 0.75, `max ${P.maxRingVsImage.toFixed(3)} px`);
  check('ALIGNMENT real image: overlay ring centre vs its model coordinate <= 1 screen px', R.n >= 8 && R.maxRingVsModelScreenPx <= 1.0, JSON.stringify(R));
  check('ALIGNMENT real image: ring centre vs blob centroid (informational; includes detector centre vs blob centroid)', R.n > 0 && Number.isFinite(R.maxRingVsImage), `max ${R.maxRingVsImage.toFixed(3)} px; blob-vs-model max ${R.maxImageVsModel.toFixed(3)} px`);
  fs.writeFileSync(path.join(OUT, 'alignment-summary.json'), JSON.stringify(alignSummary, null, 1));
}

// ================================================================== wrap up
check('no uncaught page errors / console errors', pageErrors.length === 0, pageErrors.slice(0, 4).join(' | '));
console.log('HTTP >=400 responses seen by the browser:', failedReqs.length ? failedReqs.join('; ') : 'none');
await browser.close();
await stopServer();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed; data dir ${DATA}`);
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify({ results, failedReqs, DATA }, null, 2));
process.exit(failed.length ? 1 : 0);
