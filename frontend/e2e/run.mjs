// End-to-end checks in a REAL Chromium against the mock API (frontend/mock/server.mjs).
//   npm run build && npm run e2e
// Screenshots go to e2e-artifacts/ (git-ignored).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { server, SPOTS, SIGMA } from '../mock/server.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, '../e2e-artifacts');
fs.mkdirSync(OUT, { recursive: true });

const CHROME =
  process.env.CHROMIUM_PATH ||
  ['/opt/pw-browsers/chromium', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].find((p) => fs.existsSync(p));

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const BASE = `http://127.0.0.1:${server.address().port}`;

const results = [];
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${info ? '  — ' + info : ''}`);
};

const browser = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox'] });
const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 }, acceptDownloads: true });
const page = await ctx.newPage();
page.setDefaultTimeout(8000);
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && pageErrors.push(m.text()));

const reset = async () => {
  await fetch(`${BASE}/api/__mock/reset`, { method: 'POST' });
};
const tid = (id) => page.locator(`[data-testid="${id}"]`);
const setChecked = async (id, want = true) => {
  const loc = tid(id);
  if ((await loc.isChecked()) !== want) await loc.click();
  await page.waitForFunction(([i, w]) => document.querySelector(`[data-testid="${i}"]`).checked === w, [id, want]);
};
const shot = (name) => page.screenshot({ path: path.join(OUT, name + '.png') });

async function openAnalysis(id) {
  await page.goto(BASE + '/');
  await tid(`analysis-${id}`).click();
  await page.waitForSelector('[data-testid=viewer]');
  await page.waitForFunction(() => document.querySelector('[data-testid=viewer]')?.dataset.imgReady === '1');
  await page.waitForTimeout(100);
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

/** centroid computed from SCREENSHOT PIXELS inside a clip, inside the browser. */
async function centroid(clip, mode) {
  const buf = await page.screenshot({ clip });
  return page.evaluate(
    async ({ b64, mode, clip }) => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height;
      const g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      let sw = 0,
        sx = 0,
        sy = 0,
        x0 = 1e9,
        x1 = -1,
        y0 = 1e9,
        y1 = -1;
      for (let y = 0; y < c.height; y++)
        for (let x = 0; x < c.width; x++) {
          const i = (y * c.width + x) * 4;
          const r = d[i],
            gg = d[i + 1],
            b = d[i + 2];
          if (mode === 'img') {
            const w = Math.max(0, r + gg - 60);
            sw += w;
            sx += w * (x + 0.5);
            sy += w * (y + 0.5);
          } else if (Math.min(r, gg, b) >= 150) {
            // ring: bounding-box centre of the white outline (robust to the review marker on the ring)
            sw += 1;
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x + 1);
            y0 = Math.min(y0, y);
            y1 = Math.max(y1, y + 1);
          }
        }
      if (mode === 'ring') return sw ? { x: clip.x + (x0 + x1) / 2, y: clip.y + (y0 + y1) / 2, n: sw } : null;
      return sw ? { x: clip.x + sx / sw, y: clip.y + sy / sw, n: sw } : null;
    },
    { b64: buf.toString('base64'), mode, clip },
  );
}

async function setHidden(want) {
  const cur = (await tid('viewer').getAttribute('data-hidden')) === '1';
  if (cur !== want) {
    await page.keyboard.press('h');
    await page.waitForFunction((w) => document.querySelector('[data-testid=viewer]').dataset.hidden === (w ? '1' : '0'), want);
  }
}

// ============================================================= 1. alignment
async function alignmentSuite() {
  await reset();
  await openAnalysis('demo');
  await tid('labels-mode').selectOption('off');
  // select raw composite (spots visible), overlays on, nothing selected
  const targets = SPOTS.filter((s) => s.cls !== 'uncertain'); // red circle, green diamond, yellow square, ...
  const states = [];
  const box = await tid('viewer').boundingBox();
  const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

  const measure = async (label) => {
    let done = 0;
    const v = await viewState();
    for (const s of targets) {
      const p = await toScreen(s.x, s.y);
      const R = Math.min(120, Math.max(26, Math.round(4 * SIGMA * v.s)));
      if (p.x - R < v.left + 4 || p.y - R < v.top + 4 || p.x + R > v.left + v.w - 4 || p.y + R > v.top + v.h - 4) continue;
      const clip = { x: Math.round(p.x - R), y: Math.round(p.y - R), width: 2 * R, height: 2 * R };
      await setHidden(true);
      const spot = await centroid(clip, 'img');
      await setHidden(false);
      const ring = await centroid(clip, 'ring');
      if (!spot || !ring) {
        check(`alignment ${label} spot ${s.key}`, false, `no pixels (spot=${!!spot}, ring=${!!ring})`);
        continue;
      }
      const dx = ring.x - spot.x,
        dy = ring.y - spot.y;
      const mx = spot.x - p.x,
        my = spot.y - p.y;
      // tolerance: 0.75 screen px (antialiasing); also model must agree with the image pixels
      const ok = Math.abs(dx) < 0.75 && Math.abs(dy) < 0.75 && Math.abs(mx) < 0.75 && Math.abs(my) < 0.75;
      check(
        `alignment [${label}] spot#${s.key} (${s.x},${s.y}) ${s.cls}`,
        ok,
        `zoom ${v.s.toFixed(2)}; ring-vs-image-centroid d=(${dx.toFixed(2)},${dy.toFixed(2)})px; image-vs-model d=(${mx.toFixed(2)},${my.toFixed(2)})px`,
      );
      done++;
    }
    states.push([label, done]);
    await page.mouse.move(centre.x, centre.y);
    await shot(`align-${label.replace(/\W+/g, '_')}`);
  };

  await measure('fit');
  // wheel zoom about cursor at the red spot
  let p = await toScreen(100, 100);
  await page.mouse.move(p.x, p.y);
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -120);
  await page.waitForTimeout(150);
  await measure('zoom-in at red spot');
  // pan by dragging empty canvas
  await page.mouse.move(box.x + 60, box.y + 60);
  await page.mouse.down();
  await page.mouse.move(box.x + 140, box.y + 110, { steps: 5 });
  await page.mouse.up();
  await measure('panned');
  // zoom at another point
  p = await toScreen(300, 250);
  await page.mouse.move(Math.max(box.x + 20, Math.min(box.x + box.width - 20, p.x)), Math.max(box.y + 20, Math.min(box.y + box.height - 20, p.y)));
  for (let i = 0; i < 4; i++) await page.mouse.wheel(0, 100);
  await page.waitForTimeout(150);
  await measure('zoom-out after pan');
  // high zoom on the yellow square (pixelated)
  await page.keyboard.press('f');
  p = await toScreen(300, 250);
  await page.mouse.move(p.x, p.y);
  for (let i = 0; i < 16; i++) await page.mouse.wheel(0, -120);
  await page.waitForTimeout(200);
  await measure('high zoom yellow');
  p = await toScreen(200, 150);
  // fit then zoom on green diamond
  await page.keyboard.press('f');
  p = await toScreen(200, 150);
  await page.mouse.move(p.x, p.y);
  for (let i = 0; i < 12; i++) await page.mouse.wheel(0, -120);
  await page.waitForTimeout(200);
  await measure('high zoom green');
  check('alignment exercised at >= 6 spot/zoom states', states.reduce((a, [, n]) => a + n, 0) >= 6, JSON.stringify(states));
  await shot('align-final-closeup');
}

// ============================================================= 2. coordinates / zoom about cursor
async function coordSuite() {
  await reset();
  await openAnalysis('demo');
  const box = await tid('viewer').boundingBox();
  // Whole CSS pixels: the wheel anchor and the pointer sample then land on the same point. At scale ~2.4 a
  // sub-pixel offset between the two would show up as ~0.2 image px of drift (seen in CI, not locally).
  const at = { x: Math.round(box.x + 300), y: Math.round(box.y + 200) };
  await page.mouse.move(at.x, at.y);
  const readout = async () => {
    const t = await tid('coord-readout').innerText();
    const m = t.match(/x\s+([-\d.]+)\s+y\s+([-\d.]+)/);
    return m ? { x: +m[1], y: +m[2] } : null;
  };
  const before = await readout();
  await page.mouse.wheel(0, -300);
  // leave and return to the anchor so the readout is re-sampled at the same screen point under the new view
  await page.mouse.move(at.x + 3, at.y);
  await page.mouse.move(at.x, at.y);
  await page.waitForTimeout(100);
  const after = await readout();
  const s = await viewState();
  // one readout unit (the readout shows 0.1 px); the anchor is exact, so anything larger is a real regression
  const ok = before && after && Math.abs(before.x - after.x) <= 0.1 && Math.abs(before.y - after.y) <= 0.1;
  check('zoom about cursor keeps image point fixed', ok, `before=${JSON.stringify(before)} after=${JSON.stringify(after)} scale=${s.s.toFixed(2)}`);
  // readout equals model
  const exp = { x: (at.x - s.left - s.tx) / s.s, y: (at.y - s.top - s.ty) / s.s };
  check('coordinate readout matches transform', Math.abs(exp.x - after.x) < 0.15 && Math.abs(exp.y - after.y) < 0.15, `exp=${exp.x.toFixed(2)},${exp.y.toFixed(2)}`);
  // fit button
  await tid('fit').click();
  const f = await viewState();
  check('fit button recentres (scale changed back)', f.s < s.s);
  // contrast sliders only change request params
  const reqs = [];
  const h = (r) => r.url().includes('/image/') && reqs.push(r.url());
  page.on('request', h);
  await tid('slider-hi').fill('500');
  await page.waitForTimeout(600);
  page.off('request', h);
  check('contrast slider re-requests image with lo/hi params', reqs.some((u) => /hi=\d+/.test(u)), reqs.at(-1) ?? 'none');
  check('display-only label visible', (await tid('display-controls').innerText()).includes('Display only – does not affect detection'));
  await tid('auto-contrast').click();
}

// ============================================================= 3. regions
async function regionSuite() {
  await reset();
  await openAnalysis('demo');
  await tid('tab-regions').click();
  await tid('tool-draw-count-rect').click();
  const a = await toScreen(220, 200);
  const b = await toScreen(380, 300);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(300);
  const cards = await page.locator('[data-testid^="region-reg-"]').count();
  check('draw rectangle creates a counting region', cards >= 2, `region cards: ${cards}`);
  const analysis = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  const nr = analysis.regions.find((r) => r.name === 'Region 2');
  check('new region persisted via PUT /regions with expected geometry', !!nr && Math.abs(nr.polygon[0][0] - 220) < 1.5 && Math.abs(nr.polygon[2][1] - 300) < 1.5, JSON.stringify(nr?.polygon));
  await shot('region-drawn');

  // polygon tool
  await tid('tool-draw-excl-poly').click();
  const pts = [await toScreen(420, 200), await toScreen(500, 200), await toScreen(500, 260), await toScreen(440, 270)];
  for (const p of pts) await page.mouse.click(p.x, p.y);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(300);
  const a2 = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  const ex = a2.regions.filter((r) => r.kind === 'exclusion').find((r) => r.polygon.length === 4 && r.name.startsWith('Exclusion'));
  check('polygon exclusion drawn with Enter to finish', !!ex, ex ? `${ex.name} ${JSON.stringify(ex.polygon)}` : 'none');

  // approval refused with reasons
  await tid('tab-regions').click();
  await page.locator('[data-testid=approve-reg-1]').click();
  await page.waitForSelector('[data-testid=region-error-reg-1]');
  const msg = await tid('region-error-reg-1').innerText();
  check('approval refusal reason shown', /inspected/.test(msg) || /uncertain/.test(msg), msg);
  await setChecked('inspected-reg-1');
  await page.waitForTimeout(250);
  await page.locator('[data-testid=approve-reg-1]').click();
  await page.waitForTimeout(250);
  const msg2 = await tid('region-error-reg-1').innerText();
  check('approval still refused while uncertain objects unresolved', /uncertain/.test(msg2), msg2);
  await shot('region-approval-refused');
  // rename
  await page.locator('[data-testid=region-reg-1] .rname').dblclick();
  await tid('region-name-input').fill('Left lobe');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(300);
  const a3 = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  check('region rename saved', a3.regions.find((r) => r.id === 'reg-1')?.name === 'Left lobe');
  // vertex edit: drag a vertex of selected region
  await page.locator('[data-testid=region-reg-1] .rhead').click();
  const v0 = await toScreen(20, 20);
  await page.mouse.move(v0.x, v0.y);
  await page.mouse.down();
  await page.mouse.move(v0.x + 20, v0.y + 14, { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(300);
  const a4 = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  const poly = a4.regions.find((r) => r.id === 'reg-1').polygon;
  check('region vertex dragged', poly[0][0] > 20.5 && poly[0][1] > 20.5, JSON.stringify(poly[0]));
  // ROI suggestions are not auto-applied
  check('embedded ROI suggestion not auto-applied', !a4.regions.some((r) => r.name === 'embedded-1'));
  await page.locator('[data-testid=roi-suggestions] button', { hasText: 'Adopt as counting' }).first().click();
  await page.waitForTimeout(300);
  const a5 = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  check('adopting ROI suggestion creates region on explicit click', a5.regions.some((r) => r.name === 'embedded-1'));
  // delete region
  page.once('dialog', (d) => d.accept());
  await page.locator('[data-testid^="delete-region-"]').last().click();
  await page.waitForTimeout(300);
  const a6 = await (await fetch(`${BASE}/api/analyses/demo`)).json();
  check('region delete', a6.regions.length === a5.regions.length - 1);
}

// ============================================================= 4. editing + hotkeys
async function editSuite() {
  await reset();
  await openAnalysis('demo');
  const count = async () => (await (await fetch(`${BASE}/api/analyses/demo`)).json()).objects.filter((o) => !o.deleted);
  const n0 = (await count()).length;
  // add mode
  await page.keyboard.press('a');
  await tid('addcls-green').click();
  const p = await toScreen(250, 200);
  await page.mouse.click(p.x, p.y);
  await page.waitForTimeout(250);
  let objs = await count();
  const added = objs.find((o) => o.origin === 'manual');
  check('add mode: click adds spot at clicked image coordinate', objs.length === n0 + 1 && added && Math.abs(added.x - 250) < 0.6 && Math.abs(added.y - 200) < 0.6 && added.cls === 'green', JSON.stringify(added && [added.x, added.y, added.cls]));
  await page.keyboard.press('v');
  // select it, then reclassify via hotkeys
  await page.mouse.click(p.x, p.y);
  await page.waitForSelector('[data-testid=inspector]');
  await page.keyboard.press('y');
  await page.waitForTimeout(250);
  objs = await count();
  check('hotkey Y reclassifies selected to yellow', objs.find((o) => o.id === added.id).cls === 'yellow');
  await page.keyboard.press('r');
  await page.waitForTimeout(200);
  check('hotkey R reclassifies to red', (await count()).find((o) => o.id === added.id).cls === 'red');
  await shot('inspector');
  // move by dragging
  const to = await toScreen(280, 215);
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(300);
  const moved = (await count()).find((o) => o.id === added.id);
  check('drag moves spot via API', Math.abs(moved.x - 280) < 0.8 && Math.abs(moved.y - 215) < 0.8, `${moved.x},${moved.y}; origin=${moved.origin}`);
  // delete with Del
  await page.keyboard.press('Delete');
  await page.waitForTimeout(250);
  check('Delete key deletes selected spot', (await count()).length === n0);
  // undo / redo
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(300);
  check('Ctrl+Z undoes delete', (await count()).length === n0 + 1);
  await page.keyboard.press('Control+Shift+Z');
  await page.waitForTimeout(300);
  check('Ctrl+Shift+Z redoes delete', (await count()).length === n0);
  // merge
  const a = await toScreen(100, 100);
  const b = await toScreen(60, 320);
  await page.mouse.click(a.x, a.y);
  await page.keyboard.down('Shift');
  await page.mouse.click(b.x, b.y);
  await page.keyboard.up('Shift');
  await page.keyboard.press('m');
  await page.waitForTimeout(300);
  objs = await count();
  check('M merges two selected spots', objs.length === n0 - 1 && objs.some((o) => o.history.some((h) => h.startsWith('merged'))));
  // split
  const mid = objs.find((o) => o.history.some((h) => h.startsWith('merged')));
  const ms = await toScreen(mid.x, mid.y);
  await page.mouse.click(ms.x, ms.y);
  await page.keyboard.press('s');
  const s1 = await toScreen(mid.x - 20, mid.y);
  const s2 = await toScreen(mid.x + 20, mid.y);
  await page.mouse.click(s1.x, s1.y);
  await page.mouse.click(s2.x, s2.y);
  await page.waitForTimeout(300);
  check('S then two clicks splits a spot', (await count()).length === n0);
  // hide overlays hotkey
  await page.keyboard.press('h');
  check('H hides all overlays (objects layer removed)', (await page.locator('[data-testid=objects-layer]').count()) === 0);
  await shot('overlays-hidden');
  await page.keyboard.press('h');
  check('H shows overlays again', (await page.locator('[data-testid=objects-layer]').count()) === 1);
  // class toggle
  await tid('vis-red').uncheck({ force: true }).catch(() => undefined);
  // help dialog
  await page.keyboard.press('?');
  check('? opens shortcut help', await tid('help-dialog').isVisible());
  await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  check('Esc closes help', (await tid('help-dialog').count()) === 0);
  // failure -> toast
  await fetch(`${BASE}/api/__mock/fail-next-action`, { method: 'POST', body: JSON.stringify({ message: 'injected action failure' }) });
  const q = await toScreen(200, 150);
  await page.mouse.click(q.x, q.y);
  await page.keyboard.press('u');
  await page.waitForSelector('[data-testid=toast]');
  check('action error shown in toast', /injected action failure/.test(await tid('toast').first().innerText()));
  await shot('toast-error');
}

// ============================================================= 5. review queue
async function reviewSuite() {
  await reset();
  await openAnalysis('demo');
  const items = await page.locator('[data-testid^="queue-o-"]').count();
  check('review queue lists flagged items with reasons', items === 4, `items=${items}`);
  check('queue item shows reason text', (await page.locator('[data-testid^="queue-o-"]').first().innerText()).includes('ambiguous match'));
  await page.mouse.move(700, 500);
  await page.keyboard.press('j');
  await page.waitForSelector('[data-testid=inspector]');
  const s1 = await viewState();
  check('J selects first queue item and zooms', s1.s >= 9.9, `scale=${s1.s.toFixed(1)}`);
  const first = await tid('inspector').innerText();
  check('inspector shows evidence and reasons, no confidence %', /SNR/.test(first) && /ambiguous/i.test(first) && !/%/.test(first), first.slice(0, 120).replace(/\n/g, ' | '));
  await shot('review-zoom');
  await page.keyboard.press('Enter'); // confirm + advance
  await page.waitForTimeout(500);
  const objs = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).objects;
  check('Enter confirms (reviewed) and advances', objs.find((o) => o.id === 'o-0004').reviewed === true && (await page.locator('[data-testid^="queue-o-"]').count()) === 3);
  await page.keyboard.press('k');
  await page.waitForTimeout(150);
  check('K navigates back', (await tid('inspector').count()) === 1);
}

// ============================================================= 6. mapping, composite, csv, upload
async function mappingSuite() {
  await reset();
  await openAnalysis('unmapped').catch(() => undefined);
  await page.waitForSelector('[data-testid=mapping-dialog]');
  const hint = await tid('mapping-dialog').innerText();
  check('mapping dialog shows label_hint as hint', hint.includes('635 [W1]') && /hint/i.test(hint));
  check('no role preselected', (await page.locator('[data-testid=mapping-dialog] input[type=radio]:checked').count()) === 0);
  check('confirm disabled until explicit choice', await tid('confirm-mapping').isDisabled());
  await shot('mapping-dialog');
  await tid('role-0-red').check();
  await tid('role-1-green').check();
  check('confirm enabled after choosing', await tid('confirm-mapping').isEnabled());
  await tid('confirm-mapping').click();
  await page.waitForTimeout(400);
  const a = await (await fetch(`${BASE}/api/analyses/unmapped`)).json();
  check('mapping saved, status ready', a.status === 'ready' && a.image.channels.every((c) => c.role_confirmed));

  await page.goto(BASE + '/');
  await tid('analysis-composite').click();
  await page.waitForSelector('[data-testid=banner-composite]');
  check('composite-only banner text', (await tid('banner-composite').innerText()).includes('Raw channels unavailable; counts are approximate'));
  await shot('composite-banner');

  // csv panel
  await tid('analysis-demo').click();
  await page.locator('.csv-title').first().click();
  const blanks = await page.locator('.csv-tbl td.cell-blank').count();
  const zeros = await page.locator('.csv-tbl td.cell-val', { hasText: /^0$/ }).count();
  check('CSV shows blank cells distinct from 0', blanks === 2 && zeros === 1, `blank=${blanks} zero=${zeros}`);
  check('CSV colour not preselected', (await tid('csv-colour').inputValue()) === '');
  await tid('csv-colour').selectOption('green');
  await page.locator('.csv-detail button', { hasText: 'Save' }).click();
  await page.waitForTimeout(300);
  const csv = await (await fetch(`${BASE}/api/csv-imports/csv-1`)).json();
  check('CSV colour mapping saved only after explicit choice', csv.color_mapping === 'green');
  await shot('csv-panel');

  // upload
  const dir = path.join(OUT, 'fixtures');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'a.tif'), 'II*\0fake');
  fs.writeFileSync(path.join(dir, 'b.tif'), 'II*\0fake');
  fs.writeFileSync(path.join(dir, 'c.png'), 'x');
  await tid('file-input').setInputFiles([path.join(dir, 'a.tif'), path.join(dir, 'b.tif')]);
  await page.getByRole('button', { name: 'Create analysis' }).click();
  await page.waitForSelector('[data-testid=mapping-dialog]');
  check('upload of 2 TIFFs creates analysis needing mapping', true);
  await tid('role-0-ignore').check();
  await tid('role-1-ignore').check();
  check('ignoring both channels is rejected in UI', await tid('confirm-mapping').isDisabled());
  await page.keyboard.press('Escape');
}

// ============================================================= 7. detection + finalize
async function detectSuite() {
  await reset();
  await openAnalysis('demo');
  await tid('tab-detect').click();
  check('dense decomposition off by default', (await tid('red-dense').isChecked()) === false);
  await tid('red-dense').check();
  check('dense decomposition warning shown when enabled', (await tid('detect-tab').innerText()).includes('Dense decomposition can split or merge'));
  await tid('red-dense').uncheck();
  check('pixel unit note when pixel size unknown', (await tid('detect-tab').innerText()).includes('Pixel size unknown'));
  check('re-run warning present', /Re-running keeps your manual edits/.test(await tid('detect-tab').innerText()));
  // preview
  await tid('preview-btn').click();
  const a = await toScreen(80, 80);
  const b = await toScreen(330, 280);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 5 });
  await page.mouse.up();
  await page.waitForSelector('[data-testid=preview-info]');
  check('preview overlay shown (temporary)', (await page.locator('[data-testid=preview-layer] g').count()) >= 1, await tid('preview-info').innerText());
  await shot('preview');
  // job with progress + cancel
  await tid('run-detection').click();
  await page.waitForSelector('[data-testid=job]');
  const p1 = +(await page.locator('[role=progressbar]').getAttribute('aria-valuenow'));
  await page.waitForTimeout(900);
  const p2 = +(await page.locator('[role=progressbar]').getAttribute('aria-valuenow'));
  check('job progress bar advances', p2 > p1 || p2 > 0, `${p1} -> ${p2}`);
  await tid('cancel-job').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=job]')?.dataset.status === 'cancelled');
  check('job cancel', true);
  // failure
  await page.locator('[data-testid=red-radius]').fill('30');
  await tid('run-detection').click();
  await page.waitForSelector('[data-testid=job][data-status=failed]', { timeout: 8000 });
  check('failed job shows error', /Spot radius too large/.test(await tid('job').innerText()));
  await shot('job-failed');
  await page.locator('[data-testid=red-radius]').fill('4');
  await tid('run-detection').click();
  await page.waitForSelector('[data-testid=job][data-status=done]', { timeout: 8000 });
  check('job completes', true);

  // totals
  await tid('tab-totals').click();
  const t1 = await tid('counts-slide').innerText();
  check('totals: exclusive columns', /red-only/.test(t1) && /green-only/.test(t1) && /yellow/.test(t1) && /uncertain/.test(t1));
  await tid('mode-channel_inclusive').click();
  await page.waitForTimeout(400);
  const t2 = await tid('counts-slide').innerText();
  check('totals: channel-inclusive columns clearly labelled', /red\+/.test(t2) && /yellow overlap/.test(t2) && /Channel-inclusive/.test(t2));
  check('DRAFT badge visible', (await tid('draft-badge').first().innerText()) === 'DRAFT');
  await shot('totals-inclusive');
  await tid('mode-exclusive').click();

  // finalize flow
  await tid('tab-finish').click();
  check('finalize disabled with blockers listed', (await tid('finalize').isDisabled()) && (await page.locator('[data-testid=checklist] li.bad').count()) >= 2);
  await shot('finish-blocked');
  // resolve: reclassify uncertain -> yellow; inspect + approve
  const q = await toScreen(400, 80);
  await page.mouse.click(q.x, q.y);
  await page.keyboard.press('y');
  await page.waitForTimeout(250);
  await tid('tab-regions').click();
  await setChecked('inspected-reg-1');
  await page.waitForTimeout(250);
  await page.locator('[data-testid=accept-reg-1]').click();
  await page.waitForTimeout(250);
  await page.locator('[data-testid=approve-reg-1]').click();
  await page.waitForTimeout(300);
  check('region approved badge', (await tid('region-status-reg-1').innerText()) === 'approved');
  await tid('tab-finish').click();
  check('finalize enabled when no blockers', await tid('finalize').isEnabled());
  // export link has DRAFT label before final
  check('export labelled DRAFT before finalizing', (await tid('export-objects').innerText()).includes('DRAFT'));
  await tid('finalize').click();
  await page.waitForSelector('[data-testid=reopen]');
  check('FINAL badge after finalizing', (await tid('draft-badge').first().innerText()) === 'FINAL');
  check('export labelled FINAL', (await tid('export-objects').innerText()).includes('FINAL'));
  const [dl] = await Promise.all([page.waitForEvent('download'), tid('export-objects').click()]);
  check('export download filename carries status', /FINAL/.test(dl.suggestedFilename()), dl.suggestedFilename());
  await shot('finalized');
  await tid('reopen').click();
  await page.waitForTimeout(400);
  check('reopen returns to review (DRAFT)', (await tid('draft-badge').first().innerText()) === 'DRAFT');
  // slide meta
  check('slide meta empty by default', (await tid('slide-volume').inputValue()) === '' && (await tid('slide-factor').inputValue()) === '');
  check('concentration gating note shown', /Lab export not yet possible/.test(await tid('slide-form').innerText()));
  await tid('slide-volume').fill('50');
  await tid('slide-save').click();
  await page.waitForTimeout(300);
  const s = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).slide;
  check('slide meta saved with null for unset fields', s.sample_volume_ul === 50 && s.concentration_factor === null);
  check('old total-area field removed', (await tid('slide-total-area').count()) === 0);
}

// ============================================================= grid: click to select, Randomise (persisted spec)
async function gridPickSuite() {
  await reset();
  await openAnalysis('demo');
  await tid('tab-regions').click();
  await tid('grid-toggle').click();
  const setNum = async (k, v) => tid(`grid-${k}`).fill(String(v));
  await setNum('rows', 2); await setNum('cols', 3); await setNum('cellW', 100); await setNum('cellH', 100); await setNum('originX', 20); await setNum('originY', 20);
  page.once('dialog', (d) => d.accept());
  await tid('grid-apply').click();
  await page.waitForSelector('[data-testid=grid-cells]');
  const st = () => page.evaluate(() => window.__polony.getState());
  check('Apply switches to the grid-pick tool', (await page.evaluate(() => window.__polony.getState().tool)) === 'grid-pick' && (await tid('grid-pick-help').count()) === 1);
  const onCells = () => page.evaluate(() => [...document.querySelectorAll('[data-testid=grid-cells] g[data-on="1"]')].map((g) => +g.dataset.field));
  check('stored grid drawn, all 6 fields counted', (await page.locator('[data-testid=grid-cells] g[data-field]').count()) === 6 && (await onCells()).length === 6);
  let srv = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).slide.grid;
  check('grid spec persisted on the server', srv && srv.rows === 2 && srv.cols === 3 && srv.cell_w === 100 && srv.seed === null, JSON.stringify(srv));
  // N = none, then click two cells
  await page.mouse.move(5, 5);
  await page.keyboard.press('n');
  await page.waitForFunction(() => document.querySelectorAll('[data-testid=grid-cells] g[data-on="1"]').length === 0);
  check('N switches every field off', (await onCells()).length === 0);
  const click = async (x, y) => {
    const p = await toScreen(x, y);
    const before = (await onCells()).length;
    await page.mouse.click(p.x, p.y);
    await page.waitForFunction((n) => document.querySelectorAll('[data-testid=grid-cells] g[data-on="1"]').length !== n, before);
  };
  await click(170, 70); // field 2 (row 0, col 1)
  await click(170, 170); // field 5 (row 1, col 1)
  check('clicking two cells selects fields 2 and 5', JSON.stringify((await onCells()).sort()) === '[2,5]', JSON.stringify(await onCells()));
  const p2 = await toScreen(170, 70);
  await page.mouse.move(p2.x, p2.y);
  await page.waitForSelector('[data-testid=grid-cells] g[data-hover="1"]');
  check('hovered cell highlighted', (await page.locator('[data-testid=grid-cells] g[data-hover="1"]').getAttribute('data-field')) === '2');
  // reload: grid and selection unchanged, the spec is loaded into the form
  await openAnalysis('demo');
  await tid('tab-regions').click();
  await page.waitForSelector('[data-testid=grid-cells]');
  check('after reload fields 2 and 5 still selected', JSON.stringify((await onCells()).sort()) === '[2,5]');
  await tid('grid-toggle').click();
  check('stored spec loaded into the generator form', (await tid('grid-rows').inputValue()) === '2' && (await tid('grid-cellW').inputValue()) === '100' && (await tid('grid-originX').inputValue()) === '20');
  // Randomise: same size and count, inside the image, selection kept by field number, seed recorded
  await tid('grid-randomise').click();
  await page.waitForFunction(() => window.__polony.getState().current.slide.grid?.seed != null);
  srv = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).slide;
  const g = srv.grid;
  const regs = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).regions.filter((r) => r.field_index != null);
  const polys = [...regs.map((r) => r.polygon), ...srv.skipped_fields.map((x) => x.polygon)];
  const inside = polys.every((p) => p.every(([x, y]) => x >= -0.5 && y >= -0.5 && x <= 511.5 && y <= 383.5));
  const sized = polys.every((p) => p[1][0] - p[0][0] === 100 && p[2][1] - p[1][1] === 100);
  check('Randomise keeps size/count, stays inside, records the seed', g.rows === 2 && g.cols === 3 && polys.length === 6 && inside && sized && Number.isInteger(g.seed), JSON.stringify(g));
  check('Randomise keeps the selected field numbers', JSON.stringify(regs.map((r) => r.field_index).sort()) === '[2,5]' && (await tid('grid-stored').innerText()).includes(`seed ${g.seed}`));
  check('grid-pick active after Randomise', (await st()).tool === 'grid-pick');
  // I = invert
  await page.mouse.move(5, 5);
  await page.keyboard.press('i');
  await page
    .waitForFunction(() => [...document.querySelectorAll('[data-testid=grid-cells] g[data-on="1"]')].map((g) => g.dataset.field).sort().join() === '1,3,4,6')
    .catch(() => undefined);
  check('I inverts the selection', JSON.stringify((await onCells()).sort()) === '[1,3,4,6]', JSON.stringify(await onCells()));
  // Esc leaves the tool; A then means add mode again
  await page.keyboard.press('Escape');
  check('Esc leaves grid-pick', (await st()).tool === 'select');
  await shot('grid-pick');
  await reset();
}

// ============================================================= lab export
async function labSuite() {
  await reset();
  await openAnalysis('demo');
  await tid('tab-regions').click();
  check('blank-vs-zero note shown', /No region = BLANK cell, region with 0 objects = 0/.test(await tid('field-note').innerText()));
  await tid('grid-toggle').click();
  check('numbering-order caveat shown', /confirm with your lab convention/.test(await tid('grid-order-note').innerText()));
  check('defaults prefilled 4x7', (await tid('grid-rows').inputValue()) === '4' && (await tid('grid-cols').inputValue()) === '7');
  // image is small: use an explicit 2x3 grid inside the demo region
  const setNum = async (k, v) => tid(`grid-${k}`).fill(String(v));
  await setNum('rows', 2); await setNum('cols', 3); await setNum('cellW', 100); await setNum('cellH', 100); await setNum('originX', 20); await setNum('originY', 20);
  await tid('grid-preview-btn').click();
  await page.waitForSelector('[data-testid=grid-preview]');
  check('grid preview drawn on image (6 cells)', (await page.locator('[data-testid=grid-preview] polygon').count()) === 12);
  check('exclusion warn not shown for clear grid', (await tid('grid-exclusion-warn').count()) === 0);
  await setNum('originX', 0); await setNum('originY', 0);
  check('partial exclusion warned', (await tid('grid-exclusion-warn').count()) === 1 && /partially covered/.test(await tid('grid-exclusion-warn').innerText()));
  await setNum('originX', 20); await setNum('originY', 20);
  await setNum('firstIndex', 23);
  check('field numbers > 24 refuse apply', (await tid('grid-apply').isDisabled()) && /exceed 24/.test(await tid('grid-issues').innerText()));
  await setNum('firstIndex', 1);
  await tid('grid-mode').selectOption('add');
  page.once('dialog', (d) => d.accept());
  await tid('grid-apply').click();
  await page.waitForTimeout(500);
  let regs = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).regions;
  const fl = regs.filter((r) => r.field_index != null);
  check('6 Field-N regions created with field_index', fl.length === 6 && fl[0].name === 'Field-1' && fl[5].field_index === 6, JSON.stringify(fl.map((r) => r.name)));
  check('preview cleared after apply', (await tid('grid-preview').count()) === 0);
  // edit one field number to a duplicate -> flagged
  const first = fl[0].id;
  await tid(`field-index-${first}`).fill('2');
  await page.waitForTimeout(400);
  check('duplicate field number flagged', (await tid(`field-dup-${first}`).count()) === 1 && /duplicate/.test(await tid('field-issues').innerText()));
  await tid(`field-index-${first}`).fill('1');
  await page.waitForTimeout(400);
  // delete a field -> blank
  page.once('dialog', (d) => d.accept());
  await tid(`delete-region-${fl[5].id}`).click();
  await page.waitForTimeout(400);
  check('deleted field gone (blank in export)', (await tid('field-count').innerText()).startsWith('5 lab fields'));
  // re-run grid in replace mode
  await tid('grid-mode').selectOption('replace');
  page.once('dialog', (d) => { if (!/Replace 5/.test(d.message())) throw new Error('bad confirm: ' + d.message()); d.accept(); });
  await tid('grid-apply').click();
  await page.waitForTimeout(500);
  regs = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).regions;
  check('replace mode kept 6 fields', regs.filter((r) => r.field_index != null).length === 6);

  // metadata form
  await tid('tab-finish').click();
  await tid('slide-id').fill('AI_A');
  await tid('slide-sampling-date').fill('15/09/2026');
  await tid('slide-polony-date').fill('30/09/2026');
  await tid('slide-qc').selectOption('0');
  await tid('chip-vol-2.5').click();
  await page.waitForTimeout(150);
  check('volume chip fills field only', (await tid('slide-volume').inputValue()) === '2.5');
  await tid('chip-vol-2.5056').click();
  await page.waitForTimeout(150);
  check('2.5056 chip fills', (await tid('slide-volume').inputValue()) === '2.5056');
  await tid('chip-areas').click();
  await page.waitForTimeout(150);
  check('area chip fills 2.9 / 0.1', (await tid('slide-field-area').inputValue()) === '2.9' && (await tid('slide-grid-area').inputValue()) === '0.1');
  await tid('slide-factor').fill('1');
  await tid('series-add').click();
  await tid('series-name-0').fill('T7A');
  check('series without count type blocks save', (await tid('slide-save').isDisabled()) && /count type/.test(await tid('series-errors').innerText()));
  check('never-inferred text', /never inferred from names or colours/.test(await tid('series-note').innerText()));
  await tid('series-type-0').selectOption('red_only');
  check('formula text shown', /mean\(field counts\) × \(field area \/ grid area\) ÷ sample volume × 1000/.test(await tid('formula-note').innerText()));
  check('confirmation renamed', /field geometry and series mapping match the lab convention/.test(await tid('formula-note').innerText()));
  await tid('slide-save').click();
  await page.waitForTimeout(400);
  let sl = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).slide;
  check('series saved with explicit count type', sl.lab_series.length === 1 && sl.lab_series[0].count_type === 'red_only' && sl.field_area === 2.9 && sl.qc === 0 && sl.sampling_date === '15/09/2026');

  // batch panel: draft disabled
  await tid('lab-batch-toggle').click();
  check('draft not selectable in batch', await tid('lab-sel-demo').isDisabled());
  // make the demo analysis finalizable through the API
  await fetch(`${BASE}/api/__mock/force-final/demo`, { method: 'POST' });
  await page.reload();
  await page.waitForSelector('[data-testid=lab-batch-toggle]');
  await tid('lab-batch-toggle').click();
  check('finalized analysis selectable', await tid('lab-sel-demo').isEnabled());
  await tid('lab-sel-demo').check();
  await tid('lab-series').fill('T7A');
  // blockers: confirmation missing
  await tid('lab-download').click();
  await page.waitForSelector('[data-testid=lab-blockers]');
  check('server blockers listed verbatim', /missing confirmation of field geometry and series mapping/.test(await tid('lab-blockers').innerText()));
  await shot('lab-blockers');
  // confirm, then single export buttons + batch download
  await fetch(`${BASE}/api/__mock/confirm-lab/demo`, { method: 'POST' });
  await openAnalysis('demo');
  await tid('tab-finish').click();
  check('per-series lab export button on finalized', (await tid('lab-export-T7A-xlsx').count()) === 1 && /T7-style XLSX/.test(await tid('lab-export-T7A-xlsx').innerText()));
  const [d1] = await Promise.all([page.waitForEvent('download'), tid('lab-export-T7A-csv').click()]);
  check('single lab CSV downloads', /T7A.*\.csv$/.test(d1.suggestedFilename()), d1.suggestedFilename());
  await tid('lab-batch-toggle').click();
  await tid('lab-sel-demo').check();
  await tid('lab-series').fill('T7A');
  const [d2] = await Promise.all([page.waitForEvent('download'), tid('lab-download').click()]);
  check('batch lab export downloads', /T7A_mock\.xlsx$/.test(d2.suggestedFilename()), d2.suggestedFilename());
}

// ============================================================= 8. themes
async function themeSuite() {
  await reset();
  await openAnalysis('demo');
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(200);
  await shot('dark-theme');
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  check('dark theme applies', bg !== 'rgb(244, 245, 247)', bg);
  await page.emulateMedia({ colorScheme: 'light' });
}


// ============================================================= 9. researcher reference
async function referenceSuite() {
  await reset();
  await openAnalysis('refslide').catch(() => undefined);
  await page.waitForSelector('[data-testid=mapping-dialog]');
  const md = await tid('mapping-dialog').innerText();
  check('mapping: suggested role text shown', /Suggested from validated GenePix stacks – confirm/.test(md));
  check('mapping: suggested option highlighted but NOT selected', (await page.locator('[data-testid=mapping-dialog] [data-suggested="1"]').count()) === 2 && (await page.locator('[data-testid=mapping-dialog] input[type=radio]:checked').count()) === 0);
  check('mapping: confirm still needs explicit selection', await tid('confirm-mapping').isDisabled());
  await tid('role-0-red').check();
  await tid('role-1-green').check();
  await tid('confirm-mapping').click();
  await page.waitForSelector('[data-testid=mapping-dialog]', { state: 'detached' });

  // detect presets
  await tid('tab-detect').click();
  check('preset note text', /18–33% fewer polonies than the researcher/.test(await tid('preset-note').innerText()) && /Not yet validated beyond these slides/.test(await tid('preset-note').innerText()));
  await tid('preset-sensitive').click();
  const scales = await page.locator('[data-testid=detect-tab] input[type=number]').evaluateAll((els) => els.map((e) => e.value));
  check('sensitive preset sets auto scale 0.5 on both channels', scales.filter((v) => v === '0.5').length === 2, scales.join(','));
  await tid('preset-default').click();
  const scales2 = await page.locator('[data-testid=detect-tab] input[type=number]').evaluateAll((els) => els.map((e) => e.value));
  check('default preset resets auto scale to 1', scales2.filter((v) => v === '1').length >= 2, scales2.join(','));

  // reference section
  await tid('tab-regions').click();
  await tid('reference-section').waitFor();
  const planesTxt = await tid('reference-planes').innerText();
  check('planes listed with point counts, role only as suggested', /page 0: 4 points/.test(planesTxt) && /page 1: 2 points/.test(planesTxt) && /suggested channel role: red/.test(planesTxt) && !/\bred plane|\bgreen plane/i.test(planesTxt), planesTxt.replace(/\n/g, ' | '));
  check('reference points hidden by default', (await tid('reference-points').count()) === 0);
  await setChecked('ref-show', true);
  check('reference points drawn (all planes)', (await tid('reference-points').getAttribute('data-count')) === '6');
  await tid('ref-plane-view').selectOption('0');
  check('plane filter for display', (await tid('reference-points').getAttribute('data-count')) === '4');
  await shot('reference-points');

  // import dialog + live preview
  await tid('ref-import-toggle').click();
  check('ambiguity explanation shown', /Fields with no researcher points are ambiguous: not counted or counted as zero\. Choose\./.test(await tid('ref-ambiguity').innerText()));
  await tid('ref-plane').selectOption('0');
  await page.waitForSelector('[data-testid=ref-grid-preview]');
  const inc = () => page.locator('[data-testid=ref-grid-preview] g[data-included="1"]').count();
  check('preview: counted-only shows 3 included cells (plane 0)', (await inc()) === 3, String(await inc()));
  await tid('ref-mode').selectOption('all');
  check('preview: all mode includes 28 cells', (await inc()) === 28);
  await tid('ref-mode').selectOption('counted');
  const firstPoly = () => page.locator('[data-testid=ref-grid-preview] g[data-included="1"] polygon').first().getAttribute('points');
  const p0 = await firstPoly();
  await tid('ref-dx').fill('7');
  await tid('ref-dy').fill('3');
  await page.waitForTimeout(150);
  const p1 = await firstPoly();
  const x0 = +p0.split(' ')[0].split(',')[0], x1 = +p1.split(' ')[0].split(',')[0];
  check('preview follows grid shift live', x1 - x0 === 7, `${p0} -> ${p1}`);
  await shot('reference-import-preview');
  await tid('ref-import-apply').click();
  await page.waitForSelector('[data-testid=ref-import]', { state: 'detached' });
  let an = await (await fetch(`${BASE}/api/analyses/refslide`)).json();
  const f2 = an.regions.find((r) => r.name === 'Field-2');
  check('imported 3 counted fields with shift', an.regions.filter((r) => r.field_index != null).length === 3 && f2 && f2.polygon[0][0] === 87 && f2.polygon[0][1] === 23, JSON.stringify(f2?.polygon[0]));
  check('preview cleared after import', (await tid('ref-grid-preview').count()) === 0);
  await tid('ref-import-toggle').click();
  check('re-import needs replace confirmation checkbox', await tid('ref-import-apply').isDisabled());
  await tid('ref-replace').check();
  page.once('dialog', (d) => d.accept());
  await tid('ref-import-apply').click();
  await page.waitForSelector('[data-testid=ref-import]', { state: 'detached' });
  check('replace import works', (await (await fetch(`${BASE}/api/analyses/refslide`)).json()).regions.filter((r) => r.field_index != null).length === 3);

  // auto chamber
  check('rim default 40', (await tid('chamber-rim').inputValue()) === '40');
  await tid('chamber-run').click();
  await page.waitForSelector('[data-testid=chamber-preview]');
  check('chamber preview drawn with suggested exclusion', (await page.locator('[data-testid=chamber-preview] polygon').count()) === 4 && /1 suggested exclusion/.test(await tid('chamber-info').innerText()));
  await shot('chamber-preview');
  await tid('chamber-apply').click();
  await page.waitForSelector('[data-testid=chamber-preview]', { state: 'detached' });
  an = await (await fetch(`${BASE}/api/analyses/refslide`)).json();
  check('chamber applied as counting region + exclusion', an.regions.some((r) => r.name === 'Chamber' && r.kind === 'counting') && an.regions.some((r) => r.kind === 'exclusion'));
  await tid('chamber-rim').fill('400');
  await tid('chamber-run').click();
  await tid('chamber-error').waitFor();
  check('auto-chamber error shown', /rim_px too large/.test(await tid('chamber-error').innerText()));

  // totals: compare + convenience counts
  await tid('tab-totals').click();
  await tid('cmp-table').waitFor();
  const note = await tid('cmp-note').innerText();
  check('compare: agreement-not-accuracy sentence', /Reference = the researcher's embedded marks for this plane; agreement is not the same as accuracy\./.test(note));
  check('compare: rows + summary', (await tid('cmp-table').locator('tbody tr').count()) === 3 && /Pearson r/.test(await tid('cmp-summary').innerText()) && /MAE/.test(await tid('cmp-summary').innerText()));
  check('compare: no accuracy percentage', !/accuracy\s*[:=]?\s*\d|%/.test(await tid('compare-section').innerText().then((t) => t.replace(/not the same as accuracy/, ''))));
  await tid('cmp-type').selectOption('green_positive');
  await tid('cmp-plane').selectOption('1');
  await page.waitForTimeout(300);
  check('compare: selectors re-query', (await tid('cmp-table').locator('tbody tr').count()) === 3);
  check('convenience counts labelled', /excl\. debris-like/.test(await tid('convenience-counts').innerText()) && /flagged objects are not removed/.test(await tid('convenience-counts').innerText()));
  await shot('compare');
  await reset();
}

// ============================================================= 8. two-engine UI
async function engineSuite() {
  await reset();
  // --- Detect tab: selector, model prep, notes
  await openAnalysis('demo');
  await tid('tab-detect').click();
  await tid('engine-section').waitFor();
  check('engine default is Big-FISH', (await tid('engine-bigfish').getAttribute('class')).includes('on'));
  const notes = await tid('engine-notes').innerText();
  check('engine notes: CPU time, GPU, unchanged threshold', /30 s per page/.test(notes) && /GPU is used automatically/.test(notes) && /default threshold is unchanged/.test(notes), notes.slice(0, 80));
  check('no Spotiflow fields while Big-FISH only', (await tid('spotiflow-model').count()) === 0 && (await tid('consensus-tol').count()) === 0);
  await tid('engine-hybrid').click();
  await tid('spotiflow-model').waitFor();
  check('hybrid shows model select + consensus tolerance', (await tid('consensus-tol').inputValue()) === '4' && (await tid('spotiflow-model').inputValue()) === 'general');
  check('consensus tolerance help text', /paired one-to-one/.test(await tid('engine-section').innerText()));
  check('downscale defaults to ÷1 with the exploratory note', (await tid('spotiflow-downscale').inputValue()) === '1' && /precision 0\.73 and recall 0\.77/.test(await tid('spotiflow-downscale-note').innerText()));
  await tid('spotiflow-preset-large').click();
  check('Large blobs preset sets ÷4', (await tid('spotiflow-downscale').inputValue()) === '4');
  await tid('spotiflow-downscale').selectOption('1');
  check('prob_thresh empty by default', (await tid('spotiflow-prob-thresh').inputValue()) === '');
  await tid('spotiflow-model').selectOption('hybiss');
  await tid('prepare-model').click();
  await page.waitForSelector('[data-testid=prepare-job][data-status=running], [data-testid=prepare-job][data-status=done]');
  await page.waitForSelector('[data-testid=prepare-job][data-status=done]');
  check('prepare model reports progress then ready', /ready/.test(await tid('spotiflow-model').innerText()), await tid('spotiflow-model').innerText());
  // run hybrid with a model whose run fails => Big-FISH result stays valid
  await tid('spotiflow-model').selectOption('smfish_3d');
  await tid('run-detection').click();
  await page.waitForSelector('[data-testid=job][data-status=done]', { timeout: 10000 });
  await tid('engine-run-error').first().waitFor();
  const et = await tid('engine-runs').innerText();
  check('failed Spotiflow run: error + Big-FISH still valid', /checksum mismatch/.test(et) && /Big-FISH result is still valid/.test(et));
  check('failed run row marked', (await page.locator('[data-testid=engine-runs] tr[data-status=failed]').count()) === 2);
  await shot('engine-run-failed');
  // successful hybrid run
  await tid('spotiflow-model').selectOption('general');
  await tid('run-detection').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=job]')?.dataset.status === 'done' && document.querySelectorAll('[data-testid=engine-runs] tr[data-status=ok]').length === 4, null, { timeout: 12000 });
  const rt = await tid('engine-runs').innerText();
  check('engine runs table: engine, model+version, checksum, page, device, time', /Spotiflow/.test(rt) && /general v1\.0/.test(rt) && /c0ffee12/.test(rt) && /cpu/.test(rt) && /31 s/.test(rt) && /Big-FISH/.test(rt), rt.replace(/\s+/g, ' ').slice(0, 160));
  await shot('engine-runs');
  // unavailable Spotiflow
  await fetch(`${BASE}/api/__mock/spotiflow`, { method: 'POST', body: JSON.stringify({ available: false }) });
  await openAnalysis('demo');
  await tid('tab-detect').click();
  await tid('spotiflow-unavailable').waitFor();
  check('Spotiflow unavailable: reason + optional note, selector disabled', /not installed/.test(await tid('spotiflow-unavailable').innerText()) && /Spotiflow is optional; Big-FISH keeps working/.test(await tid('spotiflow-unavailable').innerText()) && (await tid('engine-hybrid').isDisabled()) && (await tid('engine-bigfish').isEnabled()));
  await reset();

  // --- hybrid analysis: summary, filters, queue order, inspector
  await openAnalysis('hybrid');
  await tid('summary-total').waitFor();
  const tot = await tid('summary-total').innerText();
  check('summary total line', /Both engines 4 · Big-FISH only 3 · Spotiflow only 1 · Human added 0 · Confirmed 0 · Rejected 0 · Unresolved 8/.test(tot), tot);
  const p0 = await tid('summary-page-0').innerText();
  check('summary is per page with confirmed role (not merged)', /Page 0/.test(p0) && /confirmed role: red/.test(p0) && (await tid('summary-page-1').count()) === 1, p0.replace(/\n/g, ' '));
  const firstQ = await page.locator('[data-testid^="queue-o-"]').first().innerText();
  check('queue: disagreement first (server order)', /Big-FISH only|Spotiflow only/.test(firstQ), firstQ.replace(/\n/g, ' '));
  await tid('engine-filter-select').selectOption('spotiflow_only');
  check('filter Spotiflow only', (await page.locator('[data-testid^="queue-o-"]').count()) === 1 && (await tid('queue-o-0002').count()) === 1);
  await tid('engine-filter-select').selectOption('agreement');
  check('filter Agreement lists unqueued agreements', (await page.locator('[data-testid^="queue-o-"]').count()) === 3);
  await tid('engine-filter-select').selectOption('disagreements');
  check('filter Disagreements only', (await page.locator('[data-testid^="queue-o-"]').count()) === 3);
  await tid('engine-filter-select').selectOption('human_added');
  check('filter empty state', (await tid('filter-empty').count()) === 1);
  await tid('engine-filter-select').selectOption('all');
  await tid('queue-o-0002').click();
  await tid('engine-provenance').waitFor();
  const ins = await tid('inspector').innerText();
  check('inspector: model score + LoG response labels', /model score/.test(ins) && /LoG response \(noise units\)/.test(ins), ins.slice(0, 200).replace(/\n/g, ' | '));
  check('inspector: no percent / confidence wording', !/%|confiden/i.test(ins));
  check('inspector: Spotiflow-only has no Big-FISH coordinates', /Spotiflow only/.test(ins) && /–/.test(await tid('prov-green').innerText()));
  await tid('engine-filter-select').selectOption('agreement');
  await tid('queue-o-0003').click();
  const ag = await tid('engine-provenance').innerText();
  check('agreement: both coordinates and match distance', /matched, 1\.44 px apart/.test(ag) && /101\.20, 99\.20|301\.20, 249\.20/.test(ag), ag.replace(/\s+/g, ' ').slice(0, 200));
  check('agreement info popover', (await tid('agreement-info').count()) >= 1);
  await tid('engine-filter-select').selectOption('all');

  // --- overlay layers
  const before = (await (await fetch(`${BASE}/api/__mock/raw-requests`)).json()).length;
  check('no raw-detections fetched until toggled', before === 0, String(before));
  await tid('fit').click();
  await tid('engine-layers').locator('summary.btn').click();
  check('legend lists patterns + wording', /long-dash/.test(await tid('layer-legend').innerText()) && /dotted/.test(await tid('layer-legend').innerText()) && /double ring/.test(await tid('layer-legend').innerText()));
  await tid('layer-raw-bigfish').check();
  await page.waitForFunction(() => document.querySelector('[data-testid=raw-bigfish]')?.dataset.count === '7');
  check('raw Big-FISH: valid only by default', (await tid('raw-bigfish').getAttribute('data-count')) === '7', await tid('raw-bigfish').getAttribute('data-count'));
  let reqs = await (await fetch(`${BASE}/api/__mock/raw-requests`)).json();
  check('raw fetch is per engine+page, capped, valid=true, lazy', reqs.length === 2 && reqs.every((u) => /engine=bigfish/.test(u) && /valid=true/.test(u) && /limit=5000/.test(u)), reqs.join(' '));
  await tid('layer-raw-spotiflow').check();
  await page.waitForFunction(() => document.querySelector('[data-testid=raw-spotiflow]')?.dataset.count === '5');
  check('raw Spotiflow layer', true);
  await tid('layer-raw-excluded').check();
  await page.waitForFunction(() => document.querySelector('[data-testid=raw-bigfish]')?.dataset.count === '9' && document.querySelector('[data-testid=raw-spotiflow]')?.dataset.count === '7');
  check('show excluded adds faint invalid points', (await page.locator('[data-testid=raw-bigfish] g[data-valid="0"]').count()) === 2);
  check('raw shapes: hollow circle vs triangle', (await page.locator('[data-testid=raw-bigfish] circle').count()) > 0 && (await page.locator('[data-testid=raw-spotiflow] path').count()) > 0);
  await shot('engine-layers');
  await page.evaluate(() => document.activeElement && document.activeElement.blur());
  await setHidden(true);
  check('H hides raw layers too', (await tid('raw-bigfish').count()) === 0);
  await setHidden(false);
  await tid('layer-raw-bigfish').uncheck();
  await tid('layer-raw-spotiflow').uncheck();
  check('raw layers removable', (await tid('raw-bigfish').count()) === 0);
  const kinds = await page.evaluate(() => [...document.querySelectorAll('[data-consensus]')].map((g) => g.dataset.consensus));
  check('consensus rings for each status', ['agreement', 'bigfish_only', 'spotiflow_only'].every((k) => kinds.includes(k)), kinds.join(','));
  const dashes = await page.evaluate(() => {
    const d = {};
    for (const g of document.querySelectorAll('[data-consensus]')) d[g.dataset.consensus] = g.querySelector('circle:last-of-type')?.getAttribute('stroke-dasharray') ?? 'solid';
    return d;
  });
  check('status rings differ by pattern, not only colour', new Set(Object.values(dashes)).size >= 3, JSON.stringify(dashes));
  await tid('layer-consensus').uncheck();
  check('consensus styling toggle', (await page.locator('[data-consensus]').count()) === 0);
  await tid('layer-consensus').check();

  // --- reject / approve mapped to delete / confirm, rejected layer
  await page.locator('[data-testid=engine-layers] summary.btn').click();
  await tid('queue-o-0002').click();
  check('Approve / Reject buttons', (await tid('confirm').innerText()).startsWith('Approve') && (await tid('inspector-delete').innerText()).startsWith('Reject'));
  await tid('inspector-delete').click();
  await page.waitForFunction(() => !document.querySelector('[data-testid=queue-o-0002]'));
  let an = await (await fetch(`${BASE}/api/analyses/hybrid`)).json();
  check('Reject = existing delete action', an.objects.find((o) => o.id === 'o-0002').deleted === true);
  await page.waitForFunction(() => /Rejected 1/.test(document.querySelector('[data-testid=summary-total]')?.innerText ?? ''));
  check('summary counts rejected', true);
  check('rejected hidden by default', (await tid('rejected-layer').count()) === 0);
  await tid('engine-layers').locator('summary.btn').click();
  await tid('layer-rejected').check();
  await tid('rejected-layer').waitFor();
  check('Show rejected draws the deleted candidate', (await tid('rejected-layer').getAttribute('data-count')) === '1');
  await shot('engine-rejected');
  await tid('layer-rejected').uncheck();
  await page.locator('[data-testid=engine-layers] summary.btn').click();
  await tid('queue-o-0004').click();
  await tid('confirm').click();
  await page.waitForFunction(() => /Confirmed 2/.test(document.querySelector('[data-testid=summary-total]')?.innerText ?? ''));
  check('Approve = confirm; summary Confirmed', true);

  // --- QC sample
  await page.locator('[data-testid=qc-sample] > summary').click();
  check('QC explains agreement is not ground truth', /Agreement is not ground truth/.test(await tid('qc-sample').innerText()) && /agreement ≠ ground truth/.test(await tid('qc-sample').innerText()));
  check('QC default sample size is 40', (await tid('qc-n').inputValue()) === '40');
  check('QC shows no bound before review', /No QC sample drawn yet/.test(await tid('qc-bound').innerText()));
  await tid('qc-n').fill('3');
  await tid('qc-seed').fill('7');
  await tid('qc-run').click();
  await page.waitForFunction(() => document.querySelectorAll('[data-testid^="queue-o-"]').length > 0 && /QC sample/.test(document.querySelector('[data-testid=review-tab]').innerText));
  reqs = await (await fetch(`${BASE}/api/__mock/raw-requests`)).json();
  check('QC sample posts a fixed n + seed (no fraction), queue gains QC items', reqs.some((u) => u.startsWith('qc') && /"n":3/.test(u) && /"seed":7/.test(u) && !/fraction/.test(u)));
  await page.waitForFunction(() => /none reviewed yet/.test(document.querySelector('[data-testid=qc-bound]')?.innerText ?? ''));
  check('QC reports the sample size before review (no bound yet)', /QC sample of \d+ agreements: none reviewed yet/.test(await tid('qc-bound').innerText()));
  await shot('engine-review');
  await reset();
}

// ============================================================= 9. right-panel layout + low-zoom region labels
async function layoutSuite() {
  await reset();
  for (const [w, h] of [[1280, 800], [1600, 1000], [1920, 1080]]) {
    await page.setViewportSize({ width: w, height: h });
    await openAnalysis('hybrid');
    for (const t of ['totals', 'regions', 'review', 'detect', 'finish']) {
      await tid(`tab-${t}`).click();
      const o = await page.evaluate(() => {
        const r = document.querySelector('aside.right').getBoundingClientRect();
        const f = document.querySelector('[data-testid=tab-finish]').getBoundingClientRect();
        return { right: r.right, vw: innerWidth, finishRight: f.right, docOverflow: document.documentElement.scrollWidth > innerWidth };
      });
      check(`right panel fits at ${w}px, ${t} tab`, o.right <= o.vw + 0.5 && o.finishRight <= o.vw + 0.5 && !o.docOverflow, JSON.stringify(o));
    }
  }
  await page.setViewportSize({ width: 1500, height: 900 });
  await openAnalysis('hybrid');
  await page.mouse.move(700, 450);
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, 400);
  await page.waitForTimeout(200);
  const vs = await viewState();
  const lbl = await page.evaluate(() => [...document.querySelectorAll('[data-testid=overlay-svg] text')].map((t) => t.textContent).filter((t) => /Region|unreviewed|^\d+$/.test(t)));
  check('low zoom: region label is just the number', vs.s < 0.4 ? lbl.every((t) => !/unreviewed/.test(t)) : true, `scale=${vs.s.toFixed(2)} labels=${lbl.join('|')}`);
  await reset();
}

// ============================================================= 10. experiments: import, grid for all, unattended run, review, one CSV
async function experimentSuite() {
  await reset();
  await page.goto(BASE + '/');
  const api = async (p, init) => (await fetch(`${BASE}/api${p}`, init)).json();
  const st = () => page.evaluate(() => window.__polony.getState());
  await tid('exp-new').click();
  await tid('exp-name').fill('Night run');
  await tid('exp-notes').fill('T7 phage');
  await tid('exp-create').click();
  await page.waitForSelector('[data-testid=experiment-page]');
  const [exp0] = await api('/experiments');
  check('New experiment creates it with the engine defaults and opens its page', exp0?.name === 'Night run' && exp0.default_detection.engine === 'bigfish' && exp0.default_detection.preview_window === null && (await st()).mainView === 'experiment', JSON.stringify(exp0?.default_detection));
  const eid = exp0.id;
  check('left rail lists the experiment with an "open" chip', (await tid(`exp-chip-${eid}`).innerText()) === 'open');

  // import 3 TIFFs (one named to fail in the engine) + a non-TIFF that is refused
  const buf = Buffer.from('II*\0fake');
  await tid('exp-file-input').setInputFiles([
    { name: 'slide_01.tif', mimeType: 'image/tiff', buffer: buf },
    { name: 'slide_02_fail.tif', mimeType: 'image/tiff', buffer: buf },
    { name: 'slide_03.tif', mimeType: 'image/tiff', buffer: buf },
    { name: 'notes.txt', mimeType: 'text/plain', buffer: buf },
  ]);
  await page.waitForFunction(() => document.querySelectorAll('[data-testid^="exp-row-"]').length === 3);
  check('import: one row per slide, in file order', JSON.stringify(await page.$$eval('[data-testid^="exp-row-"]', (r) => r.map((x) => x.dataset.name))) === '["slide_01","slide_02_fail","slide_03"]');
  check('import: the refused file is listed with its error', /notes\.txt: 'notes\.txt': only \.tif\/\.tiff files are accepted/.test(await tid('exp-import-errors').innerText()));
  const v0 = await api(`/experiments/${eid}`);
  const [s1, s2, s3] = v0.slides.map((s) => s.id);

  // the researcher confirms the mapping of the first slide and builds its grid (Regions tab, covered by gridPickSuite)
  await tid(`exp-open-${s1}`).click();
  await page.waitForSelector('[data-testid=mapping-dialog]');
  await tid('role-0-red').check();
  await tid('role-1-green').check();
  await tid('confirm-mapping').click();
  await page.waitForFunction(() => !document.querySelector('[data-testid=mapping-dialog]'));
  const spec = { rows: 2, cols: 3, cell_w: 100, cell_h: 100, origin_x: 20, origin_y: 20, order: 'row-major', corner: 'tl', first_index: 1, seed: null, shift_dx: 0, shift_dy: 0 };
  const J = { 'Content-Type': 'application/json' };
  await api(`/analyses/${s1}/grid`, { method: 'PUT', headers: J, body: JSON.stringify({ grid: spec }) });
  await api(`/analyses/${s1}/grid-selection`, { method: 'PUT', headers: J, body: JSON.stringify({ field_index: 6, counted: false }) });
  check('the analyses list shows the experiment of a slide', (await tid(`analysis-exp-${s1}`).innerText()).includes('Night run'));
  await tid('back-to-experiment').click();
  await page.waitForSelector('[data-testid=experiment-page]');

  // copy the confirmed mapping to the other slides (one explicit confirmation)
  page.once('dialog', (d) => d.accept());
  await tid('exp-copy-mapping').click();
  await page.waitForFunction(() => !document.querySelector('[data-testid=exp-mapping-note]'));
  check('mapping copied to the 2 other slides', (await api(`/experiments/${eid}`)).slides.every((s) => s.mapping_confirmed));

  // grid for all: template = slide 1 (2 x 3, field 6 skipped)
  await page.waitForFunction(() => /skipped: 6/.test(document.querySelector('[data-testid=exp-grid-spec]')?.innerText ?? ''));
  check('grid box shows the template slide grid', /2 × 3 fields of 100 × 100 px at \(20, 20\)/.test(await tid('exp-grid-spec').innerText()));
  await tid('exp-grid-apply-missing').click();
  await page.waitForFunction(() => /2 ok, 0 refused/.test(document.querySelector('[data-testid=exp-grid-results]')?.innerText ?? ''));
  const a2 = await api(`/analyses/${s2}`), a3 = await api(`/analyses/${s3}`);
  const same = (a) => a.slide.grid && a.slide.grid.rows === 2 && a.slide.grid.cols === 3 && a.slide.grid.cell_w === 100 && a.slide.grid.origin_x === 20 && a.slide.grid.origin_y === 20;
  const skippedOf = (a) => JSON.stringify(a.slide.skipped_fields.map((x) => x.field_index));
  check('Apply to all: same spec on every slide without a grid', same(a2) && same(a3));
  check('Apply to all: the same skipped field (6) on every slide', skippedOf(a2) === '[6]' && skippedOf(a3) === '[6]' && a2.regions.filter((r) => r.field_index != null).length === 5);
  const e1 = await api(`/experiments/${eid}`);
  check('the template grid is stored as the experiment default_grid', e1.default_grid?.rows === 2 && JSON.stringify(e1.default_grid.skipped_fields) === '[6]' && e1.default_grid.template_id === s1, JSON.stringify(e1.default_grid));

  // apply to all (overwrite) asks first and names the slides with a grid
  let msg = '';
  page.once('dialog', (d) => ((msg = d.message()), d.dismiss()));
  await tid('exp-grid-apply-all').click();
  await page.waitForTimeout(200);
  check('Apply to all slides confirms, naming slides that already have a grid', /^2 slides already have a grid; overwrite\?/.test(msg), msg);

  // randomise on all: fresh seed per slide, shown per slide
  page.once('dialog', (d) => d.accept());
  await tid('exp-grid-randomise').click();
  await page.waitForFunction((ids) => ids.every((id) => document.querySelector(`[data-testid=exp-grid-${id}]`)?.dataset.seed), [s1, s2, s3]);
  const seeds = await page.$$eval('[data-testid^="exp-grid-"][data-seed]', (els) => els.map((e) => e.dataset.seed).filter(Boolean));
  const srvSeeds = await Promise.all([s1, s2, s3].map(async (id) => String((await api(`/analyses/${id}`)).slide.grid.seed)));
  check('Randomise on all: one fresh seed per slide, shown in the table', seeds.length === 3 && new Set(seeds).size === 3 && JSON.stringify(seeds) === JSON.stringify(srvSeeds), `${seeds} vs ${srvSeeds}`);
  check('Randomise keeps the skipped field', skippedOf(await api(`/analyses/${s3}`)) === '[6]');
  check('seed in the row tooltip', (await tid(`exp-grid-${s2}`).getAttribute('title')) === `seed ${srvSeeds[1]}`);

  // run and leave: progress, then done_with_failures with the failed row showing its error
  await tid('exp-run-all').click();
  await tid('exp-progress').waitFor();
  const sawRunning = await page.waitForFunction(() => document.querySelector('[data-testid=exp-progress]')?.dataset.status === 'running', null, { timeout: 3000 }).then(() => true).catch(() => false);
  check('Run: progress bar while the batch runs, Cancel offered', sawRunning && (await tid('exp-cancel').count()) === 1);
  await page.waitForFunction(() => document.querySelector('[data-testid=exp-progress]')?.dataset.status === 'done_with_failures', null, { timeout: 15000 });
  check('run ends done_with_failures (terminal)', (await tid('exp-batch-status').innerText()) === 'done with failures' && (await tid('exp-cancel').count()) === 0);
  check('progress label: 3 / 3 finished, 2 done, 1 failed', (await tid('exp-progress-label').innerText()) === '3 / 3 finished · 2 done · 1 failed', await tid('exp-progress-label').innerText());
  check('the failed row is red and shows its error', (await tid(`exp-row-${s2}`).getAttribute('data-status')) === 'failed' && /RuntimeError: mock engine failure on 'slide_02_fail'/.test(await tid(`exp-err-${s2}`).innerText()));
  check('done rows show their counts', (await tid(`exp-row-${s1}`).getAttribute('data-status')) === 'done' && /\d+ \/ \d+ \/ \d+/.test(await tid(`exp-row-${s1}`).innerText()));
  await page.waitForFunction((id) => document.querySelector(`[data-testid=exp-chip-${id}]`)?.innerText === 'done with failures', eid);
  check('left-rail chip shows done with failures', true);
  await shot('experiment-done-with-failures');

  // Retry the failed row: a run of that slide only
  await tid(`exp-retry-${s2}`).click();
  await page.waitForFunction(() => document.querySelector('[data-testid=exp-progress]')?.dataset.status === 'running', null, { timeout: 3000 }).catch(() => undefined);
  await page.waitForFunction(() => !['running', 'queued'].includes(document.querySelector('[data-testid=exp-progress]')?.dataset.status), null, { timeout: 10000 });
  const r2 = await api(`/experiments/${eid}/run`);
  check('Retry runs only the failed slide', r2.batch.detail.slides.length === 1 && r2.batch.detail.slides[0].analysis_id === s2 && r2.summary.total === 1, JSON.stringify(r2.summary));
  check('after Retry the row shows the new failure', /mock engine failure/.test(await tid(`exp-err-${s2}`).innerText()));

  // Next unreviewed: ] on the experiment page opens the first unapproved slide, ] on a slide the next one
  await tid('exp-run').click({ position: { x: 4, y: 4 } });
  await page.keyboard.press(']');
  await page.waitForSelector('[data-testid=viewer]');
  check('] on the experiment page opens the first unreviewed slide', (await st()).current.id === s1 && (await st()).mainView === 'analysis');
  await page.mouse.move(5, 5);
  await page.keyboard.press(']');
  await page.waitForFunction((id) => window.__polony.getState().current?.id === id && !window.__polony.getState().loadingCurrent, s2);
  check('] on a slide opens the next unreviewed slide', true);
  await tid('back-to-experiment').click();
  await tid('exp-next-unreviewed').click();
  await page.waitForFunction((id) => window.__polony.getState().current?.id === id && window.__polony.getState().mainView === 'analysis', s1);
  check('Next unreviewed button opens the first unreviewed slide', true);
  await tid('back-to-experiment').click();

  // finalize refuses with the unreviewed slides; one CSV for the experiment
  await tid('exp-finalize').click();
  await tid('exp-finalize-blocked').waitFor();
  check('Finalize experiment refuses, listing the unreviewed slides', /3 slide\(s\) are not fully approved/.test(await tid('exp-finalize-blocked').innerText()) && /slide_02_fail/.test(await tid('exp-finalize-blocked').innerText()));
  const href = await tid('exp-csv').getAttribute('href');
  check('Download CSV link points at the experiment CSV', href === `/api/experiments/${eid}/export.csv`, href);
  check('Provenance link present', (await tid('exp-provenance').getAttribute('href')) === `/api/experiments/${eid}/provenance.json`);
  const [dl] = await Promise.all([page.waitForEvent('download'), tid('exp-csv').click()]);
  const csvText = fs.readFileSync(await dl.path(), 'utf8');
  check('experiment CSV: one row per slide per colour, failed slide left out', dl.suggestedFilename() === 'Night_run_experiment.csv' && csvText.split('\r\n').filter((l) => /^slide_0[13],/.test(l)).length === 6 && !/slide_02_fail/.test(csvText), dl.suggestedFilename());
  check('Lab export disabled until a slide is finalized', await tid('exp-lab').isDisabled());

  // reorder: ▼ on the first slide
  await tid(`exp-down-${s1}`).click();
  await page.waitForFunction(() => document.querySelector('[data-testid^="exp-row-"]')?.dataset.name === 'slide_02_fail');
  check('▼ moves a slide down (PUT /order)', JSON.stringify((await api(`/experiments/${eid}`)).slides.map((s) => s.id)) === JSON.stringify([s2, s1, s3]));

  // delete keeps the slides as loose analyses
  await tid('exp-delete').click();
  await tid('exp-delete-keep').click();
  await page.waitForFunction(() => !document.querySelector('[data-testid=experiment-page]'));
  const left = await api('/analyses');
  check('Delete (keep slides): experiment gone, slides kept as loose analyses', (await api('/experiments')).length === 0 && [s1, s2, s3].every((id) => left.some((a) => a.id === id && a.experiment_id === null)));
  await reset();
}

const suites = [alignmentSuite, coordSuite, regionSuite, editSuite, reviewSuite, mappingSuite, detectSuite, labSuite, gridPickSuite, themeSuite, referenceSuite, engineSuite, layoutSuite, experimentSuite];
// E2E_ONLY=experimentSuite,gridPickSuite runs a subset while developing
const only = (process.env.E2E_ONLY || "").split(",").filter(Boolean);
for (const s of suites.filter((x) => !only.length || only.includes(x.name))) {
  try {
    await s();
  } catch (e) {
    check(`suite ${s.name} completed`, false, String(e).split('\n').slice(0,6).join(' ⏎ '));
    await shot(`FAILED-${s.name}`).catch(() => undefined);
  }
}
check('no uncaught page errors', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));
await browser.close();
server.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 2));
process.exit(failed.length ? 1 : 0);
