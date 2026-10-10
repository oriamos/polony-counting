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
        sy = 0;
      for (let y = 0; y < c.height; y++)
        for (let x = 0; x < c.width; x++) {
          const i = (y * c.width + x) * 4;
          const r = d[i],
            gg = d[i + 1],
            b = d[i + 2];
          const w = mode === 'img' ? Math.max(0, r + gg - 60) : Math.max(0, Math.min(r, gg, b) - 90);
          sw += w;
          sx += w * (x + 0.5);
          sy += w * (y + 0.5);
        }
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
  const at = { x: box.x + 300, y: box.y + 200 };
  await page.mouse.move(at.x, at.y);
  const readout = async () => {
    const t = await tid('coord-readout').innerText();
    const m = t.match(/x\s+([-\d.]+)\s+y\s+([-\d.]+)/);
    return m ? { x: +m[1], y: +m[2] } : null;
  };
  const before = await readout();
  await page.mouse.wheel(0, -300);
  await page.mouse.move(at.x + 0.01, at.y); // refresh readout
  await page.waitForTimeout(100);
  const after = await readout();
  const s = await viewState();
  const ok = before && after && Math.abs(before.x - after.x) < 0.2 && Math.abs(before.y - after.y) < 0.2;
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
  await page.mouse.click(b.x, b.y, { modifiers: ['Shift'] });
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
  check('dense decomposition off by default', (await page.locator('.params input[type=checkbox]').first().isChecked()) === false);
  await page.locator('.params input[type=checkbox]').first().check();
  check('dense decomposition warning shown when enabled', (await tid('detect-tab').innerText()).includes('Dense decomposition can split or merge'));
  await page.locator('.params input[type=checkbox]').first().uncheck();
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
  check('concentration gating note shown', /Concentration will not be computed/.test(await tid('slide-form').innerText()));
  await tid('slide-volume').fill('50');
  await tid('slide-save').click();
  await page.waitForTimeout(300);
  const s = (await (await fetch(`${BASE}/api/analyses/demo`)).json()).slide;
  check('slide meta saved with null for unset fields', s.sample_volume_ul === 50 && s.concentration_factor === null);
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

const suites = [alignmentSuite, coordSuite, regionSuite, editSuite, reviewSuite, mappingSuite, detectSuite, themeSuite];
for (const s of suites) {
  try {
    await s();
  } catch (e) {
    check(`suite ${s.name} completed`, false, String(e).split('\n')[0]);
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
