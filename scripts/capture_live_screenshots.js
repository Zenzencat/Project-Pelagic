// One-off capture script for docs/deck_assets/live_fallback/ (Round 14
// verification screenshots; Round 15 added the tile-load wait below after
// Track K found demo_fallback screenshots needed one for the same reason).
// Not part of the app runtime -- run manually:
//   npx playwright chromium (already cached in this environment)
//   node scripts/capture_live_screenshots.js
// Requires both dev servers already running (frontend :5173, backend :8000).
const { chromium } = require('playwright');

const OUT_DIR = 'docs/deck_assets/live_fallback';

// Same verified "tiles actually finished loading" wait built for Track K
// (docs/deck_assets/MANIFEST.md's demo_fallback section) -- a fixed sleep
// isn't a real guarantee, and screenshots 2/3 here show the map, so the
// same tile-mid-load risk applies. Scoped to OSM tile requests specifically
// (not a page-wide networkidle), checks every <img class="leaflet-tile">
// is complete with a real decoded image (not just Leaflet's own
// leaflet-tile-loaded class, which it adds on the error path too), and
// waits out any in-progress pan/zoom animation from fitBounds.
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
        for (const t of tiles) {
          if (!t.complete || t.naturalWidth === 0) return false;
        }
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

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:\\Users\\Iris\\AppData\\Local\\ms-playwright\\chromium-1208\\chrome-win64\\chrome.exe',
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

  console.log('[*] Loading app...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  await page.waitForSelector('text=ดึงภาพจริง (Live Fetch)');
  await page.waitForTimeout(1500); // let demo scene cards + default detail panel settle
  await waitForTilesSettled(page);

  console.log('[*] Screenshot 1: Live Fetch panel BEFORE any fetch (initial state)');
  await page.screenshot({ path: `${OUT_DIR}/01_live_fetch_panel_before.png`, fullPage: true });

  console.log('[*] Clicking Live Fetch (real CDSE + GFW call, this takes ~25-40s)...');
  await page.click('text=ดึงภาพจริง (Live Fetch)');

  // Wait for the real result banner (OK/NOT_CONFIGURED/SKIPPED/ERROR) to appear --
  // polls the actual DOM, not a fixed sleep, so this reflects the TRUE outcome
  // whatever it is (never staged).
  await page.waitForSelector('text=/^(OK|NOT_CONFIGURED|SKIPPED|ERROR):/', { timeout: 180000 });
  await page.waitForTimeout(1000); // let the map re-fit/re-render after state update
  await waitForTilesSettled(page); // fitBounds pans/zooms to the new detection -- wait for those tiles too

  const resultText = await page.locator('text=/^(OK|NOT_CONFIGURED|SKIPPED|ERROR):/').first().innerText();
  console.log(`[*] Real result: ${resultText}`);

  console.log('[*] Screenshot 2: real live-fetch result (full sidebar, incl. acquisition timestamp if OK)');
  await page.screenshot({ path: `${OUT_DIR}/02_live_fetch_result.png`, fullPage: true });

  console.log('[*] Screenshot 3: map view (vessel markers if any + GFW attribution near AIS layer control)');
  const mapPanel = page.locator('#map-container');
  await mapPanel.screenshot({ path: `${OUT_DIR}/03_map_and_gfw_attribution.png` });

  console.log('[*] Screenshot 4: sidebar detail panel close-up (vessel list / honest empty state + GFW credit)');
  // Scroll the vessel list itself into view first -- the sidebar is a
  // scrollable column taller than the viewport, and an element screenshot
  // only captures what's actually laid out/visible, not off-screen overflow.
  const vesselHeading = page.locator('text=เรือใกล้เคียงที่เป็นไปได้').first();
  await vesselHeading.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const sidebar = page.locator('#sidebar');
  await sidebar.screenshot({ path: `${OUT_DIR}/04_vessel_attribution_panel.png` });

  await browser.close();
  console.log('[+] Done. True result was:', resultText);
})();
