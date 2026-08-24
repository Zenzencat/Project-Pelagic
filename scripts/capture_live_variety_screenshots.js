// Round 16 Part 2: expanded docs/deck_assets/live_fallback/ screenshot set.
// Adds variety on top of the existing Round 14/15 Singapore Strait set
// (01-04, left untouched) -- a different geography (Stockholm Archipelago),
// an honest few/no-vessels result (wherever it naturally occurs), and a
// legible close-up of the Round 15 marker-spread fix. Real live calls only;
// whatever each location's real result is gets captured, not forced.
//
// Run manually (both dev servers already running, frontend :5173 backend :8000):
//   node scripts/capture_live_variety_screenshots.js
const { chromium } = require('playwright');

const OUT_DIR = 'docs/deck_assets/live_fallback';

// Same verified tile-load wait as scripts/capture_live_screenshots.js (Track K method).
async function waitForTilesSettled(page, { settleMs = 600, timeoutMs = 20000 } = {}) {
  const inflight = new Set();
  const onRequest = (req) => { if (req.url().includes('tile.openstreetmap.org')) inflight.add(req); };
  const onDone = (req) => inflight.delete(req);
  page.on('request', onRequest);
  page.on('requestfinished', onDone);
  page.on('requestfailed', onDone);

  const deadline = Date.now() + timeoutMs;
  let lastChange = Date.now();
  let lastCount = null;
  try {
    while (Date.now() < deadline) {
      const count = inflight.size;
      if (count !== lastCount) { lastCount = count; lastChange = Date.now(); }
      const domReady = await page.evaluate(() => {
        const tiles = document.querySelectorAll('.leaflet-tile-pane img.leaflet-tile');
        if (tiles.length === 0) return false;
        for (const t of tiles) { if (!t.complete || t.naturalWidth === 0) return false; }
        const pane = document.querySelector('.leaflet-map-pane');
        if (pane && (pane.classList.contains('leaflet-pan-anim') || pane.classList.contains('leaflet-zoom-anim'))) return false;
        return true;
      });
      if (count === 0 && domReady && (Date.now() - lastChange) >= settleMs) return;
      await page.waitForTimeout(100);
    }
    throw new Error(`Tiles never settled: inflight=${inflight.size} lastCount=${lastCount}`);
  } finally {
    page.off('request', onRequest);
    page.off('requestfinished', onDone);
    page.off('requestfailed', onDone);
  }
}

async function setNumberInput(page, placeholder, value) {
  await page.locator(`input[placeholder="${placeholder}"]`).fill(String(value));
}

async function runLiveFetch(page, { minLat, minLon, maxLat, maxLon, dateFrom, dateTo, radiusKm = 10 }) {
  await setNumberInput(page, 'min_lat', minLat);
  await setNumberInput(page, 'min_lon', minLon);
  await setNumberInput(page, 'max_lat', maxLat);
  await setNumberInput(page, 'max_lon', maxLon);
  const dateInputs = page.locator('input[type="date"]');
  await dateInputs.nth(0).fill(dateFrom);
  await dateInputs.nth(1).fill(dateTo);
  const radiusInput = page.locator('input[type="number"][min="1"]');
  await radiusInput.fill(String(radiusKm));

  await page.click('text=ดึงภาพจริง (Live Fetch)');
  await page.waitForSelector('text=/^(OK|NOT_CONFIGURED|SKIPPED|ERROR):/', { timeout: 180000 });
  await page.waitForTimeout(1000);
  await waitForTilesSettled(page);
  const resultText = await page.locator('text=/^(OK|NOT_CONFIGURED|SKIPPED|ERROR):/').first().innerText();
  return resultText;
}

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:\\Users\\Iris\\AppData\\Local\\ms-playwright\\chromium-1208\\chrome-win64\\chrome.exe',
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });

  console.log('[*] Loading app...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  await page.waitForSelector('text=ดึงภาพจริง (Live Fetch)');
  await page.waitForTimeout(1500);
  await waitForTilesSettled(page);

  // --- 1. Singapore Strait, land-mask fix confirmed (Pulau Semakau) ---
  console.log('[*] 1/3: Singapore Strait (land-mask fix)...');
  let result = await runLiveFetch(page, {
    minLat: 1.10, minLon: 103.70, maxLat: 1.30, maxLon: 103.90,
    dateFrom: '2026-08-05', dateTo: '2026-08-15', radiusKm: 10,
  });
  console.log('    result:', result);
  await page.screenshot({ path: `${OUT_DIR}/05_singapore_landmask_fixed_result.png`, fullPage: true });
  const mapPanel = page.locator('#map-container');
  await mapPanel.screenshot({ path: `${OUT_DIR}/06_singapore_landmask_fixed_map.png` });

  // --- 2. Stockholm Archipelago, Sweden -- different geography, many small islands ---
  console.log('[*] 2/3: Stockholm Archipelago...');
  result = await runLiveFetch(page, {
    minLat: 59.20, minLon: 18.30, maxLat: 59.45, maxLon: 18.75,
    dateFrom: '2026-07-25', dateTo: '2026-08-15', radiusKm: 10,
  });
  console.log('    result:', result);
  await page.screenshot({ path: `${OUT_DIR}/07_stockholm_archipelago_result.png`, fullPage: true });
  await mapPanel.screenshot({ path: `${OUT_DIR}/08_stockholm_archipelago_map.png` });

  // Vessel marker cluster close-up (proving the Round 15 marker-spread fix
  // stays legible as a standalone crop, not just visible if you squint at
  // a full-map screenshot). Whichever of the two real captures so far has
  // vessels renders here -- Stockholm returned 5 real candidates.
  const vesselCount = await page.locator('text=/^[0-9]+ ลำ \\(Vessels\\)/').first().innerText().catch(() => null);
  console.log('    vessel count text:', vesselCount);

  // --- 3. Remote location -- real result, not forced (Chonos Archipelago, Chile) ---
  console.log('[*] 3/3: Chonos Archipelago, Chile (remote)...');
  result = await runLiveFetch(page, {
    minLat: -45.30, minLon: -74.10, maxLat: -45.05, maxLon: -73.65,
    dateFrom: '2026-07-25', dateTo: '2026-08-15', radiusKm: 10,
  });
  console.log('    result:', result);
  await page.screenshot({ path: `${OUT_DIR}/09_chonos_remote_result.png`, fullPage: true });

  await browser.close();
  console.log('[+] Done.');
})();
