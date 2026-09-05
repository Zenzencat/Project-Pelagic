import { WindEvidence, ObservationEvidence } from './SupplementaryEvidence';
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, CircleMarker, Polyline, useMap } from 'react-leaflet';
import { ShieldAlert, Layers, Anchor, Loader2, Info } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:8000';

// The 4 verified demo scenes for the live presentation, in the fixed order
// they should be presented -- not the raw DB history, which also contains
// stale/dummy seed rows and would let a presenter accidentally click into
// an unverified scene mid-demo.
//
// No no_oil scene: every no_oil holdout scene turned out to be either
// genuinely land (no_oil_00002/3/4/6/7/8/9), a live false positive on real
// water (no_oil_00000/1), or a broken/corrupted file (no_oil_00005) -- see
// docs/deck_assets/MANIFEST.md for the full investigation. There's no
// "real water, correctly suppressed" scene available in this holdout set,
// so the live demo is oil-detection scenes only.
const DEMO_SCENES = [
  { id: 'oil_00000', title: 'ฉากที่ 1' },
  { id: 'oil_00001', title: 'ฉากที่ 2' },
  { id: 'oil_00003', title: 'ฉากที่ 3' },
  { id: 'oil_00004', title: 'ฉากที่ 4' },
];

// Real ground area per the GeoJSON polygon's own lat/lon extent -- same
// lat-corrected km/deg methodology as src/compare_checkpoints.py
// (pixel_area_km2), applied to the actual detected polygon instead of a
// per-pixel constant. Replaces the old hardcoded 14.5/8.5 km2 mock badge.
const KM_PER_DEG_LAT = 111.32;

function calcSlickAreaKm2(det) {
  if (!det || !det.geojson_mask || !det.geojson_mask.coordinates || !det.bbox) return 0;
  const centerLat = (det.bbox[0] + det.bbox[2]) / 2;
  const kmPerDegLon = KM_PER_DEG_LAT * Math.cos((centerLat * Math.PI) / 180);
  let totalKm2 = 0;
  for (const ring of det.geojson_mask.coordinates) {
    let sum = 0;
    for (let i = 0; i < ring.length - 1; i++) {
      const [lon1, lat1] = ring[i];
      const [lon2, lat2] = ring[i + 1];
      const x1 = lon1 * kmPerDegLon, y1 = lat1 * KM_PER_DEG_LAT;
      const x2 = lon2 * kmPerDegLon, y2 = lat2 * KM_PER_DEG_LAT;
      sum += x1 * y2 - x2 * y1;
    }
    totalKm2 += Math.abs(sum) / 2;
  }
  return totalKm2;
}

// Honest reason text for the vessel_attribution_status the API now returns
// instead of the old hardcoded mock_vessels (src/api/main.py /
// src/analysis/gfw_client.py). Every status here is a real outcome of an
// actual attempt (or a real reason one couldn't be attempted) -- never a
// placeholder standing in for missing data.
function vesselStatusMessage(det) {
  switch (det?.vessel_attribution_status) {
    case 'ok':
      return null; // real vessels found -- the list below speaks for itself
    case 'empty':
      return `ไม่พบเรือในระยะ ${det.vessel_search_radius_km ?? '?'} กม. จากข้อมูล AIS จริง (Global Fishing Watch)`;
    case 'skipped_no_credentials':
      return 'ยังไม่ได้ตั้งค่า GFW API token (GFW_TOKEN) — ไม่สามารถระบุเรือใกล้เคียงได้';
    case 'skipped_no_timestamp':
      return 'ฉากนี้ไม่มีข้อมูลเวลาถ่ายภาพจริง จึงไม่สามารถค้นหาเรือ AIS ที่ตรงเวลาได้ (ใช้ได้เฉพาะฉากที่ดึงแบบ Live)';
    case 'error':
      return 'เกิดข้อผิดพลาดขณะค้นหาเรือจาก Global Fishing Watch API';
    default:
      return 'ยังไม่มีข้อมูลการระบุเรือใกล้เคียงสำหรับฉากนี้';
  }
}

// GFW's Report API reports each vessel's position as its ~0.01deg grid-cell
// center, not a precise ping (src/analysis/gfw_client.py) -- so a distance
// smaller than that cell's real size (position_resolution_m) isn't a
// meaningfully precise number, it just means "somewhere in this AIS grid
// cell." Showing e.g. "0.0 กม." for that case implied false precision;
// this shows the honest resolution instead.
function formatVesselDistance(v) {
  if (v.position_resolution_m && v.distance_meters < v.position_resolution_m) {
    return `< ${(v.position_resolution_m / 1000).toFixed(1)} กม. (ระยะกริด AIS)`;
  }
  return `${(v.distance_meters / 1000).toFixed(1)} กม.`;
}

// Multiple distinct vessels can legitimately share the exact same reported
// AIS position -- see formatVesselDistance above, same root cause. Drawn
// as-is, co-located markers render exactly on top of each other on the
// map (only the topmost is visible), which looked like "only 2 of 5
// vessels shown." This spreads co-located vessels into a small ring around
// their shared reported position purely for on-screen visibility -- the
// real lat/lon/distance used everywhere else (list, tooltip text) are
// never changed, only where the marker is drawn, and the tooltip discloses
// the spread so it doesn't read as a more precise position than it is.
function spreadColocatedVessels(vessels) {
  const groups = new Map();
  for (const v of vessels) {
    const key = `${v.latitude.toFixed(6)},${v.longitude.toFixed(6)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(v);
  }
  const out = [];
  for (const group of groups.values()) {
    if (group.length === 1) {
      out.push({ ...group[0], displayLat: group[0].latitude, displayLon: group[0].longitude, colocatedCount: 1 });
      continue;
    }
    // Spread radius is a fraction of the real AIS grid-cell size, so the
    // ring stays visually "at this cell" rather than implying a precise
    // separate position for each vessel. Kept well under half the real
    // grid step (position_resolution_m is the cell's diagonal, so the step
    // between adjacent cell centers is roughly resM / sqrt(2)) -- at 0.35
    // two adjacent grid cells' rings could reach far enough to overlap and
    // hide a marker again (found by re-inspecting a real capture: two real
    // groups one grid cell apart produced two points close enough to render
    // as one dot), so this stays comfortably below that collision radius.
    const resM = group[0].position_resolution_m || 1200;
    const spreadDegLat = (resM * 0.15) / 111320;
    group.forEach((v, i) => {
      const theta = (2 * Math.PI * i) / group.length;
      const lonScale = Math.cos((v.latitude * Math.PI) / 180) || 1;
      out.push({
        ...v,
        displayLat: v.latitude + spreadDegLat * Math.sin(theta),
        displayLon: v.longitude + (spreadDegLat / lonScale) * Math.cos(theta),
        colocatedCount: group.length,
      });
    });
  }
  return out;
}

// Land-sea masking (src/analysis/landmask.py) carves land -- including
// every small island in an archipelago -- out of the raw prediction, which
// shatters what would otherwise be one or a few large water regions into
// many small disjoint fragments (real, separate polygons, not duplicates:
// confirmed directly by running shapely's unary_union on a real 692-
// fragment live-fetch result over the Stockholm Archipelago -- it did not
// reduce the count at all, since there's nothing overlapping to merge).
// At a fixed 3px stroke, a dense cluster of these small fragments packed
// close together reads as a solid yellow mass rather than distinct
// boundaries -- easy to mistake for a second, unlabeled layer. Thinning
// the stroke as fragment count grows keeps each boundary legible instead.
// Thresholds set from real measured fragment counts across this project's
// scenes: the 4 cached demo scenes range 9-228, live-fetch scenes (which
// always carry land-mask fragmentation) range 138-692.
function getSlickStrokeWeight(nFragments) {
  if (nFragments <= 50) return 3;
  if (nFragments <= 150) return 2;
  if (nFragments <= 400) return 1.5;
  return 1;
}

// Zero-confidence detections have no real contour -- treat them as "no oil"
// rather than rendering/measuring them as detections.
const hasRealDetection = (det) => !!det && det.confidence_score > 0.001;

// Real bounding box to frame the camera on: the detected polygon's own
// coordinate extent, not the (much larger) satellite scene footprint. A
// fixed zoom centered on the scene made every contour icon-sized regardless
// of its real area -- measured at 33x41px out of a 1180x1000 map viewport
// for oil_00000's 162 km2 polygon at the old zoom=8. Falls back to the
// scene footprint for no-oil detections, which have no real polygon to fit.
function getFitBounds(det) {
  if (!det) return null;
  if (hasRealDetection(det) && det.geojson_mask && det.geojson_mask.coordinates) {
    let minLat = Infinity, minLon = Infinity, maxLat = -Infinity, maxLon = -Infinity;
    for (const ring of det.geojson_mask.coordinates) {
      for (const [lon, lat] of ring) {
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
      }
    }
    if (Number.isFinite(minLat) && Number.isFinite(minLon) && Number.isFinite(maxLat) && Number.isFinite(maxLon)) {
      return [[minLat, minLon], [maxLat, maxLon]];
    }
  }
  if (det.bbox) {
    return [[det.bbox[0], det.bbox[1]], [det.bbox[2], det.bbox[3]]];
  }
  return null;
}

// Map controller to fit the camera to the selected detection's real bounding
// box whenever the selection changes (not on every re-render -- keyed on
// det.id so toggling a layer checkbox etc. doesn't re-trigger the fit/pan).
function MapFitter({ det }) {
  const map = useMap();
  useEffect(() => {
    const bounds = getFitBounds(det);
    if (bounds) {
      map.fitBounds(bounds, { padding: [80, 80], maxZoom: 14 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [det?.id, map]);
  return null;
}

export default function App() {
  const [detections, setDetections] = useState([]);
  const [selectedDet, setSelectedDet] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  // Map Layer Controls
  const [showFootprint, setShowFootprint] = useState(true);
  const [showSlick, setShowSlick] = useState(true);
  const [showAIS, setShowAIS] = useState(true);
  
  // Notification Toast
  const [toastMessage, setToastMessage] = useState(null);

  // Live Fetch panel state -- additive "live" mode alongside the 4 verified
  // demo scenes (POST /api/live/fetch, src/api/main.py). Defaults are a
  // real, verified-covered bbox/window (Singapore Strait, confirmed to have
  // real Sentinel-1 GRD coverage) so the panel works out of the box rather
  // than requiring the presenter to already know good coordinates.
  const [liveBbox, setLiveBbox] = useState({ minLat: '1.10', minLon: '103.70', maxLat: '1.30', maxLon: '103.90' });
  const [liveDateFrom, setLiveDateFrom] = useState('2026-08-05');
  const [liveDateTo, setLiveDateTo] = useState('2026-08-15');
  const [liveRadiusKm, setLiveRadiusKm] = useState('10');
  const [includeOptical, setIncludeOptical] = useState(false);
  const [opticalWindow, setOpticalWindow] = useState(10);
  const [maxCloud, setMaxCloud] = useState(20);
  const [includeTemporal, setIncludeTemporal] = useState(false);
  const [temporalWindow, setTemporalWindow] = useState(30);
  const [includeEra5, setIncludeEra5] = useState(false);
  const [liveFetching, setLiveFetching] = useState(false);
  const [liveResult, setLiveResult] = useState(null); // {status, detail}

  // 1. Fetch detections history list on startup
  const fetchDetections = async (autoSelectId = null) => {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/api/detections`);
      if (!res.ok) throw new Error('Failed to fetch detections history.');
      const data = await res.json();
      setDetections(data);
      
      if (data.length > 0) {
        // Auto-select a specific detection if given (e.g. newly created), else
        // default to the first scene in the fixed demo lineup (not just the
        // most recently touched DB row) so a fresh page load always starts
        // the presentation from the same, predictable scene.
        const firstDemo = DEMO_SCENES.map(s => s.id)
          .map(id => data.find(d => d.scene_id === id))
          .find(Boolean);
        const targetId = autoSelectId || (firstDemo ? firstDemo.id : data[0].id);
        fetchDetectionDetails(targetId);
      } else {
        setLoading(false);
      }
    } catch (err) {
      console.error(err);
      setError('ไม่สามารถเชื่อมต่อกับ FastAPI Backend ได้ (API server offline)');
      setLoading(false);
    }
  };

  // 2. Fetch specific detection details (including AIS vessels)
  const fetchDetectionDetails = async (id) => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/detections/${id}`);
      if (!res.ok) throw new Error('Failed to fetch detection details.');
      const data = await res.json();
      setSelectedDet(data);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch details for selected detection.');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetections();
  }, []);

  // 3. Select one of the 4 verified demo scenes. Switches instantly to the
  // cached detection if it's already in the DB (the normal demo path, since
  // all 4 are pre-seeded); only falls back to a live U-Net inference call if
  // a scene is somehow missing (e.g. a freshly reset database).
  const handleSelectScene = async (sceneId) => {
    const existing = detections.find(d => d.scene_id === sceneId);
    if (existing) {
      fetchDetectionDetails(existing.id);
      return;
    }

    try {
      setPredicting(true);
      setError(null);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene_id: sceneId })
      });

      if (res.status === 503) {
        const data = await res.json();
        throw new Error(data.detail || 'Model is not loaded on server.');
      }

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Error running model inference.');
      }

      const newDet = await res.json();
      showToast(`วิเคราะห์ฉาก ${sceneId} สำเร็จ!`);
      await fetchDetections(newDet.id);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // 4. Live Fetch -- real CDSE search + Sentinel Hub Process API fetch +
  // inference + real GFW attribution (POST /api/live/fetch). Renders its own
  // status envelope (OK/NOT_CONFIGURED/SKIPPED/ERROR) rather than routing
  // through the generic `error` banner above, since NOT_CONFIGURED/SKIPPED
  // are expected, honest outcomes here, not failures.
  const handleLiveFetch = async () => {
    setLiveFetching(true);
    setLiveResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/live/fetch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          min_lat: parseFloat(liveBbox.minLat),
          min_lon: parseFloat(liveBbox.minLon),
          max_lat: parseFloat(liveBbox.maxLat),
          max_lon: parseFloat(liveBbox.maxLon),
          date_from: liveDateFrom,
          date_to: liveDateTo,
          radius_km: parseFloat(liveRadiusKm) || 10,
          include_era5: includeEra5,
          include_temporal: includeTemporal,
          include_optical: includeOptical,
          optical_window_days: Number(opticalWindow),
          optical_max_cloud_pct: Number(maxCloud),
          temporal_window_days: Number(temporalWindow),
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(Array.isArray(body.detail) ? body.detail.map(e => e.msg).join('; ') : body.detail || 'Live fetch failed.');
      setLiveResult({ status: body.status, detail: body.detail });

      if (body.status === 'OK' && body.detection) {
        showToast('ดึงภาพดาวเทียมจริงและวิเคราะห์สำเร็จ! (Live Fetch complete)');
        await fetchDetections(body.detection.id);
      }
    } catch (err) {
      console.error(err);
      setLiveResult({ status: 'ERROR', detail: err.message || 'Live fetch request failed.' });
    } finally {
      setLiveFetching(false);
    }
  };

  // Center coordinate helper
  const getMapCenter = () => {
    if (selectedDet) {
      // Find bounding box center
      const lat = (selectedDet.bbox[0] + selectedDet.bbox[2]) / 2;
      const lon = (selectedDet.bbox[1] + selectedDet.bbox[3]) / 2;
      return [lat, lon];
    }
    return [9.200, 101.500]; // Default GoT center
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      
      {/* 1. Inference Loading Overlay */}
      {predicting && (
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(9, 9, 11, 0.85)', zIndex: 9999,
          display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: '16px'
        }}>
          <Loader2 className="animate-spin" size={48} color="#06b6d4" />
          <h2 style={{ fontWeight: 600 }}>กำลังประมวลผล U-Net Model (Segmentation)...</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>กำลังคำนวณแบ่งส่วนพิกเซลคราบน้ำมัน ดึงขอบพิกัด และเขียนฐานข้อมูล</p>
        </div>
      )}

      {/* 2. Success Toast */}
      {toastMessage && (
        <div style={{
          position: 'absolute', bottom: '20px', right: '20px', zIndex: 10000,
          backgroundColor: 'var(--success)', color: '#fff', padding: '12px 24px',
          borderRadius: '6px', fontWeight: 600, fontSize: '0.9rem',
          boxShadow: '0 4px 15px rgba(0,0,0,0.4)', transition: 'all 0.3s'
        }}>
          {toastMessage}
        </div>
      )}

      {/* 3. Left Sidebar */}
      <div id="sidebar" style={{ width: '420px', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--border-color)', backgroundColor: 'var(--bg-card)', height: '100%', zIndex: 10 }}>
        
        {/* Brand Header */}
        <div style={{ padding: '20px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="glow-dot" style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--primary)' }}></div>
            <div>
              <h1 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#fff' }}>PROJECT PELAGIC</h1>
              <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>SAR Maritime Monitoring PoC</p>
            </div>
          </div>
          <div style={{ fontSize: '0.65rem', backgroundColor: 'rgba(255,255,255,0.08)', padding: '4px 8px', borderRadius: '4px', color: 'var(--text-muted)', fontWeight: 500 }}>
            SWU AI ENG
          </div>
        </div>

        {/* Error panel */}
        {error && (
          <div style={{ margin: '16px', padding: '12px', borderRadius: '6px', border: '1px solid rgba(239, 68, 68, 0.3)', backgroundColor: 'rgba(239, 68, 68, 0.08)', color: 'var(--danger)', fontSize: '0.8rem', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
            <ShieldAlert size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <p style={{ fontWeight: 600 }}>ข้อผิดพลาด (Error):</p>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Live Fetch panel -- additive "live" mode alongside the 4 verified
            demo scenes below (never modifies that flow). Real CDSE search +
            fetch + inference + real GFW attribution via POST
            /api/live/fetch; see src/data/cdse_fetch.py and
            src/analysis/gfw_client.py. Renders the NOT_CONFIGURED/SKIPPED/
            ERROR envelope as an inline banner instead of the generic error
            panel above, since those are expected outcomes here (missing
            credentials, no product for the chosen bbox/date), not crashes. */}
        <div style={{ margin: '16px', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'rgba(6, 182, 212, 0.03)' }}>
          <h3 style={{ fontSize: '0.8rem', fontWeight: 600, color: '#fff', marginBottom: '10px' }}>
            ดึงภาพดาวเทียมจริง (Live Fetch — Sentinel-1 จริงผ่าน CDSE)
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '6px' }}>
            <input type="number" step="0.01" placeholder="min_lat" value={liveBbox.minLat}
              onChange={e => setLiveBbox({ ...liveBbox, minLat: e.target.value })}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
            <input type="number" step="0.01" placeholder="min_lon" value={liveBbox.minLon}
              onChange={e => setLiveBbox({ ...liveBbox, minLon: e.target.value })}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
            <input type="number" step="0.01" placeholder="max_lat" value={liveBbox.maxLat}
              onChange={e => setLiveBbox({ ...liveBbox, maxLat: e.target.value })}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
            <input type="number" step="0.01" placeholder="max_lon" value={liveBbox.maxLon}
              onChange={e => setLiveBbox({ ...liveBbox, maxLon: e.target.value })}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '6px' }}>
            <input type="date" value={liveDateFrom} onChange={e => setLiveDateFrom(e.target.value)}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
            <input type="date" value={liveDateTo} onChange={e => setLiveDateTo(e.target.value)}
              style={{ fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
          </div>
          <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '10px' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>รัศมีค้นหาเรือ (km):</span>
            <input type="number" step="1" min="1" value={liveRadiusKm} onChange={e => setLiveRadiusKm(e.target.value)}
              style={{ width: '60px', fontSize: '0.72rem', padding: '5px 6px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: '#fff' }} />
          </div>
          <details style={{ margin: '8px 0', fontSize: '0.78rem' }}>
          <summary style={{ cursor: 'pointer' }}>Supplementary evidence (optional)</summary>
          <label style={{ display: 'block', fontSize: '0.78rem', margin: '8px 0' }}>
            <input type="checkbox" checked={includeEra5} onChange={e => setIncludeEra5(e.target.checked)} />
            {' '}ERA5 wind evidence (optional, up to 60 seconds)
          </label>
          <label style={{ display: 'block', fontSize: '0.78rem', margin: '8px 0' }}>
            <input type="checkbox" checked={includeTemporal} onChange={e => setIncludeTemporal(e.target.checked)} />
            {' '}Compare another Sentinel-1 pass
          </label>
          {includeTemporal && <label style={{ display: 'block', fontSize: '0.78rem', marginBottom: 8 }}>
            Search ± days: <input type="number" min="1" max="90" value={temporalWindow}
              onChange={e => setTemporalWindow(e.target.value)} style={{ width: 60 }} />
          </label>}
          <label style={{ display: 'block', fontSize: '0.78rem', margin: '8px 0' }}>
            <input type="checkbox" checked={includeOptical} onChange={e => setIncludeOptical(e.target.checked)} />
            {' '}Sentinel-2 optical supplement
          </label>
          {includeOptical && <div style={{ fontSize: '0.78rem', marginBottom: 8 }}>
            <label>Search ± days: <input type="number" min="1" max="30" value={opticalWindow}
              onChange={e => setOpticalWindow(e.target.value)} style={{ width: 50 }} /></label>
            {' '}<label>Max cloud %: <input type="number" min="0" max="100" value={maxCloud}
              onChange={e => setMaxCloud(e.target.value)} style={{ width: 50 }} /></label>
          </div>}
          </details>
          <button onClick={handleLiveFetch} disabled={liveFetching}
            style={{
              width: '100%', padding: '8px', borderRadius: '6px', border: 'none',
              backgroundColor: liveFetching ? 'rgba(6, 182, 212, 0.3)' : 'var(--primary)',
              color: '#fff', fontWeight: 600, fontSize: '0.78rem', cursor: liveFetching ? 'default' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
            }}>
            {liveFetching && <Loader2 size={14} className="animate-spin" />}
            {liveFetching ? 'กำลังดึงและวิเคราะห์ภาพจริง...' : 'ดึงภาพจริง (Live Fetch)'}
          </button>

          {liveResult && (
            <div style={{
              marginTop: '10px', padding: '8px', borderRadius: '4px', fontSize: '0.72rem', lineHeight: 1.4,
              border: '1px solid ' + (liveResult.status === 'OK' ? 'rgba(16, 185, 129, 0.3)' : liveResult.status === 'ERROR' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(245, 158, 11, 0.3)'),
              backgroundColor: liveResult.status === 'OK' ? 'rgba(16, 185, 129, 0.08)' : liveResult.status === 'ERROR' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(245, 158, 11, 0.08)',
              color: liveResult.status === 'OK' ? 'var(--success)' : liveResult.status === 'ERROR' ? 'var(--danger)' : 'var(--warning)'
            }}>
              <strong>{liveResult.status}</strong>{': '}{liveResult.detail}
            </div>
          )}
        </div>

        {/* Demo Scene Selector -- fixed to the 4 verified oil-detection scenes only.
            Scene list + details panel share one scrollable region so any
            leftover vertical space (a 4-card list is shorter than most
            viewports) settles below the details panel instead of opening a
            gap between the last card and the panel's top border. */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '16px 20px 8px 20px', fontWeight: 600 }}>
          เลือกฉากตัวอย่างสำหรับสาธิต (Demo Scenes)
        </div>

        <div style={{ padding: '0 20px' }}>
          {DEMO_SCENES.map(scene => {
            const det = detections.find(d => d.scene_id === scene.id);
            const isActive = selectedDet && selectedDet.scene_id === scene.id;
            const areaKm2 = det ? calcSlickAreaKm2(det) : 0;

            return (
              <div
                key={scene.id}
                onClick={() => handleSelectScene(scene.id)}
                style={{
                  backgroundColor: isActive ? 'rgba(6, 182, 212, 0.05)' : 'rgba(255,255,255,0.01)',
                  border: '1px solid',
                  borderColor: isActive ? 'var(--primary)' : 'var(--border-color)',
                  borderRadius: '8px',
                  padding: '12px 14px',
                  marginBottom: '10px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff' }}>
                    {scene.title}
                  </span>
                  {!det ? (
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                      backgroundColor: 'rgba(255,255,255,0.08)', color: 'var(--text-muted)'
                    }}>
                      รอวิเคราะห์
                    </span>
                  ) : (
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                      backgroundColor: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)'
                    }}>
                      {(det.confidence_score * 100).toFixed(1)}% Conf
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'monospace', fontSize: '0.72rem' }}>
                    {scene.id}
                  </span>
                  <span style={{ fontWeight: 500 }}>
                    {det ? `ขนาดคราบ: ${areaKm2.toFixed(1)} ตร.กม.` : '—'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Details Panel */}
        {selectedDet && (
          <div style={{ padding: '20px', borderTop: '1px solid var(--border-color)', backgroundColor: 'rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Info size={14} color="var(--primary)" /> รายละเอียดคราบนํ้ามัน (Slick Details)
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>ขนาดคราบน้ำมัน:</span>
                <span style={{ fontWeight: 600, color: 'var(--primary)' }}>
                  {calcSlickAreaKm2(selectedDet).toFixed(1)} ตร.กม.
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>พิกัดศูนย์กลาง:</span>
                <span style={{ fontWeight: 500, fontFamily: 'monospace' }}>
                  {((selectedDet.bbox[0] + selectedDet.bbox[2])/2).toFixed(4)}°N, {((selectedDet.bbox[1] + selectedDet.bbox[3])/2).toFixed(4)}°E
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>พิกัดดาวเทียม (Footprint):</span>
                <span style={{ fontWeight: 500, fontSize: '0.75rem', fontFamily: 'monospace' }}>
                  {selectedDet.bbox[0].toFixed(2)}N, {selectedDet.bbox[1].toFixed(2)}E
                </span>
              </div>
              {/* Real acquisition datetime -- only ever present for source
                  == 'live' detections (POST /api/live/fetch), straight from
                  CDSE's own catalog, never fabricated/defaulted. Cached
                  holdout/synthetic scenes have no real timestamp to show
                  (see docs/status.md) so this row simply doesn't render for
                  them, rather than showing a blank/fake value. */}
              {selectedDet.source === 'live' && selectedDet.acquisition_start_utc && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>เวลาถ่ายภาพจริง (Acquisition, UTC):</span>
                  <span style={{ fontWeight: 500, fontSize: '0.72rem', fontFamily: 'monospace' }}>
                    {selectedDet.acquisition_start_utc}
                  </span>
                </div>
              )}
              {selectedDet.source === 'live' && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>แหล่งที่มา (Source):</span>
                  <span style={{ fontWeight: 600, fontSize: '0.72rem', color: 'var(--primary)' }}>
                    LIVE — Sentinel-1 จริงผ่าน CDSE
                  </span>
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>จำนวนเรือบริเวณใกล้เคียง (AIS):</span>
                <span style={{ fontWeight: 500, color: 'var(--warning)' }}>
                  {selectedDet.nearby_vessels ? selectedDet.nearby_vessels.length : 0} ลำ (Vessels)
                </span>
              </div>
              
              {selectedDet.source === 'live' && <WindEvidence evidence={selectedDet.supplementary?.era5} />}

              {/* Vessels List inside details -- real GFW AIS candidates only
                  (src/analysis/gfw_client.py), never the old mock_vessels.
                  "candidate" wording matches the source data's own posture:
                  nearby-in-space-and-time AIS presence, not a confirmed
                  source of the slick. */}
              {selectedDet.nearby_vessels && selectedDet.nearby_vessels.length > 0 && (
                <div style={{ marginTop: '8px', padding: '8px', borderRadius: '4px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)' }}>
                  <p style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>เรือใกล้เคียงที่เป็นไปได้ (Candidate Nearby Vessels — AIS จริงจาก GFW)</p>
                  {selectedDet.nearby_vessels.map(v => (
                    <div key={v.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Anchor size={10} color="var(--warning)" /> {v.vessel_name || `MMSI ${v.mmsi}`}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>ห่าง {formatVesselDistance(v)}</span>
                    </div>
                  ))}
                  {/* Required GFW attribution (API Terms of Use, Section 3) --
                      must be visible wherever GFW-derived vessel data is
                      shown, not buried in a footer. Placed directly under
                      the vessel list it attributes, not just once globally. */}
                  <p style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px solid var(--border-color)', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    <a href="https://globalfishingwatch.org" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', textDecoration: 'underline' }}>
                      Powered by Global Fishing Watch.
                    </a>
                  </p>
                </div>
              )}

              {/* Honest empty-attribution state -- replaces what used to be
                  a silent "0 ลำ" with no explanation. See
                  vessel_attribution_status in src/api/database.py. */}
              {vesselStatusMessage(selectedDet) && (
                <div style={{ marginTop: '8px', padding: '8px', borderRadius: '4px', border: '1px dashed var(--border-color)', color: 'var(--text-muted)', fontSize: '0.72rem', lineHeight: 1.4 }}>
                  {vesselStatusMessage(selectedDet)}
                </div>
              )}
            </div>
          </div>
        )}
        </div>
      </div>

      {/* 4. Right Map Panel */}
      <div id="map-container" style={{ flex: 1, position: 'relative', height: '100%' }}>
        
        {selectedDet?.source === 'live' && selectedDet.supplementary?.original && (
          <ObservationEvidence evidence={selectedDet.supplementary} />
        )}
        {/* Leaflet MapContainer */}
        <MapContainer 
          center={getMapCenter()} 
          zoom={8} 
          style={{ width: '100%', height: '100%' }}
          zoomControl={true}
        >
          {/* Basemap Layer (Standard OpenStreetMap) */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          <MapFitter det={selectedDet} />

          {/* Render selected detection overlays */}
          {selectedDet && (
            <>
              {/* A. Bounding Box Footprint */}
              {showFootprint && (
                <Polygon 
                  positions={[
                    [selectedDet.bbox[0], selectedDet.bbox[1]],
                    [selectedDet.bbox[0], selectedDet.bbox[3]],
                    [selectedDet.bbox[2], selectedDet.bbox[3]],
                    [selectedDet.bbox[2], selectedDet.bbox[1]]
                  ]}
                  pathOptions={{
                    color: '#4285f4',
                    weight: 1.5,
                    fillColor: '#4285f4',
                    fillOpacity: 0.03,
                    dashArray: '5, 5'
                  }}
                />
              )}

              {/* B. Oil Slick Contours -- bold, high-contrast outline so the
                  detected boundary itself reads clearly on a projector, not
                  just a filled blob. Suppressed for zero-confidence
                  detections, which are the API's fallback placeholder
                  square rather than a real contour (see hasRealDetection).
                  Stroke weight scales down with fragment count
                  (getSlickStrokeWeight above) so a heavily land-fragmented
                  scene's many small boundaries don't visually merge into a
                  solid mass at a fixed bold weight. */}
              {showSlick && hasRealDetection(selectedDet) && selectedDet.geojson_mask && selectedDet.geojson_mask.coordinates && (() => {
                const strokeWeight = getSlickStrokeWeight(selectedDet.geojson_mask.coordinates.length);
                return selectedDet.geojson_mask.coordinates.map((poly, pIdx) => {
                  // GeoJSON holds [lon, lat], Leaflet needs [lat, lon]
                  const leafPositions = poly.map(pt => [pt[1], pt[0]]);
                  return (
                    <Polygon
                      key={pIdx}
                      positions={leafPositions}
                      pathOptions={{
                        color: '#facc15',
                        weight: strokeWeight,
                        opacity: 1,
                        lineJoin: 'round',
                        fillColor: '#d946ef',
                        fillOpacity: 0.35
                      }}
                    />
                  );
                });
              })()}

              {/* C. AIS Vessel Markers & Lines -- spreadColocatedVessels
                  handles vessels that share an identical reported AIS
                  position (see its own comment above): without it, those
                  markers render exactly on top of each other and only the
                  topmost is visible/clickable. */}
              {showAIS && selectedDet.nearby_vessels && spreadColocatedVessels(selectedDet.nearby_vessels).map(v => {
                const slickCenter = getMapCenter();
                return (
                  <React.Fragment key={v.id}>
                    {/* Dashed line to centroid */}
                    <Polyline
                      positions={[[v.displayLat, v.displayLon], slickCenter]}
                      pathOptions={{
                        color: 'rgba(245, 158, 11, 0.4)',
                        weight: 1.5,
                        dashArray: '3, 6'
                      }}
                    />
                    {/* Vessel point marker */}
                    <CircleMarker
                      center={[v.displayLat, v.displayLon]}
                      radius={6}
                      pathOptions={{
                        fillColor: '#f59e0b',
                        color: '#fff',
                        weight: 1.5,
                        fillOpacity: 1.0
                      }}
                    >
                      {/* Standard tooltip */}
                      <div className="custom-ship-tooltip">
                        <b>เรือ: {v.vessel_name} (MMSI: {v.mmsi})</b><br />
                        ห่าง: {formatVesselDistance(v)}
                        {v.colocatedCount > 1 && (
                          <><br /><span style={{ opacity: 0.8 }}>ตำแหน่งโดยประมาณ — {v.colocatedCount} ลำ ใช้กริด AIS เดียวกัน</span></>
                        )}
                      </div>
                    </CircleMarker>
                  </React.Fragment>
                );
              })}
            </>
          )}
        </MapContainer>

        {/* Floating Layer Control Card */}
        <div className="glass-panel" style={{
          position: 'absolute', top: '20px', right: '20px', zIndex: 1000,
          borderRadius: '8px', padding: '16px', width: '280px'
        }}>
          <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={14} color="var(--primary)" /> ควบคุมชั้นข้อมูลแผนที่
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.8rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
              <span style={{ color: 'var(--text-muted)' }}>ขอบเขตภาพดาวเทียม (Footprint)</span>
              <input 
                type="checkbox" 
                checked={showFootprint} 
                onChange={(e) => setShowFootprint(e.target.checked)}
                style={{ accentColor: 'var(--primary)' }}
              />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
              <span style={{ color: 'var(--text-muted)' }}>คราบน้ำมัน (Oil Slick Mask)</span>
              <input 
                type="checkbox" 
                checked={showSlick} 
                onChange={(e) => setShowSlick(e.target.checked)}
                style={{ accentColor: 'var(--primary)' }}
              />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
              <span style={{ color: 'var(--text-muted)' }}>พิกัดข้อมูลเรือ (AIS Overlay)</span>
              <input 
                type="checkbox" 
                checked={showAIS} 
                onChange={(e) => setShowAIS(e.target.checked)}
                style={{ accentColor: 'var(--primary)' }}
              />
            </label>
          </div>

          {/* Required GFW attribution (API Terms of Use, Section 3), also
              shown right on the map panel itself (not just the sidebar
              details panel) since this is "wherever vessel/attribution
              markers are rendered on the map" -- visible whenever the
              currently-selected detection actually has real GFW vessel
              markers on screen. */}
          {showAIS && selectedDet && selectedDet.nearby_vessels && selectedDet.nearby_vessels.length > 0 && (
            <p style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-color)', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              <a href="https://globalfishingwatch.org" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', textDecoration: 'underline' }}>
                Powered by Global Fishing Watch.
              </a>
            </p>
          )}
        </div>

      </div>

    </div>
  );
}
