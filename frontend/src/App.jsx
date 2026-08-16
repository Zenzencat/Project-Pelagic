import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, CircleMarker, Polyline, useMap } from 'react-leaflet';
import { ShieldAlert, Layers, Anchor, Loader2, Info, CheckCircle2 } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:8000';

// The 5 verified demo scenes for the live presentation, in the fixed order
// they should be presented -- not the raw DB history, which also contains
// stale/dummy seed rows and would let a presenter accidentally click into
// an unverified scene mid-demo.
const DEMO_SCENES = [
  { id: 'oil_00000', title: 'ฉากที่ 1', category: 'oil' },
  { id: 'oil_00001', title: 'ฉากที่ 2', category: 'oil' },
  { id: 'oil_00003', title: 'ฉากที่ 3', category: 'oil' },
  { id: 'oil_00004', title: 'ฉากที่ 4', category: 'oil' },
  // Reverted from no_oil_00007 (and no_oil_00001 before that): reverse-geocoded
  // every no_oil holdout scene's real coordinate and found "genuinely open
  // water" and "live v2 model correctly outputs ~0% confidence" never overlap
  // in this 10-scene set -- the only two confirmed-water scenes (no_oil_00000,
  // no_oil_00001) both trigger live false positives (~67%, ~66%), and every
  // scene the model correctly suppresses is on land. no_oil_00004 is a
  // confirmed non-maritime (land) scene the model correctly abstains on --
  // labeled as such below rather than mislabeled "clean water".
  { id: 'no_oil_00004', title: 'ฉากที่ 5', category: 'no_oil' },
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

// Zero-confidence detections are the API's fallback placeholder square
// (src/api/main.py draws a tiny mock polygon when the model finds no slick
// pixels at all), not a real contour -- treat them as "no oil" rather than
// rendering/measuring that placeholder as if it were a detection.
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
    return [[minLat, minLon], [maxLat, maxLon]];
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

  // 3. Select one of the 5 verified demo scenes. Switches instantly to the
  // cached detection if it's already in the DB (the normal demo path, since
  // all 5 are pre-seeded); only falls back to a live U-Net inference call if
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

        {/* Demo Scene Selector -- fixed to the 5 verified scenes only */}
        <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '16px 20px 8px 20px', fontWeight: 600 }}>
          เลือกฉากตัวอย่างสำหรับสาธิต (Demo Scenes)
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0 20px' }}>
          {DEMO_SCENES.map(scene => {
            const det = detections.find(d => d.scene_id === scene.id);
            const isActive = selectedDet && selectedDet.scene_id === scene.id;
            const isNoOil = scene.category === 'no_oil';
            const detected = hasRealDetection(det);
            const areaKm2 = detected ? calcSlickAreaKm2(det) : 0;

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
                  ) : detected ? (
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                      backgroundColor: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)'
                    }}>
                      {(det.confidence_score * 100).toFixed(1)}% Conf
                    </span>
                  ) : (
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                      display: 'flex', alignItems: 'center', gap: '3px',
                      backgroundColor: 'rgba(161, 161, 170, 0.15)', color: 'var(--text-muted)', border: '1px solid var(--border-color)'
                    }}>
                      <CheckCircle2 size={11} /> ไม่พบคราบ
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'monospace', fontSize: '0.72rem' }}>
                    {scene.id}
                  </span>
                  <span style={{ fontWeight: 500 }}>
                    {isNoOil
                      ? 'พื้นที่ที่ไม่ใช่ทะเล'
                      : detected
                        ? `ขนาดคราบ: ${areaKm2.toFixed(1)} ตร.กม.`
                        : '—'}
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
                {hasRealDetection(selectedDet) ? (
                  <span style={{ fontWeight: 600, color: 'var(--primary)' }}>
                    {calcSlickAreaKm2(selectedDet).toFixed(1)} ตร.กม.
                  </span>
                ) : (
                  <span style={{ fontWeight: 600, color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={13} /> ไม่พบคราบน้ำมัน — พื้นที่ที่ไม่ใช่ทะเล (Non-maritime, model correctly abstains)
                  </span>
                )}
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
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>จำนวนเรือบริเวณใกล้เคียง (AIS):</span>
                <span style={{ fontWeight: 500, color: 'var(--warning)' }}>
                  {selectedDet.nearby_vessels ? selectedDet.nearby_vessels.length : 0} ลำ (Vessels)
                </span>
              </div>
              
              {/* Vessels List inside details */}
              {selectedDet.nearby_vessels && selectedDet.nearby_vessels.length > 0 && (
                <div style={{ marginTop: '8px', padding: '8px', borderRadius: '4px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)' }}>
                  <p style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>รายชื่อเรือเดินทะเล</p>
                  {selectedDet.nearby_vessels.map(v => (
                    <div key={v.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Anchor size={10} color="var(--warning)" /> {v.vessel_name}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>ห่าง {(v.distance_meters / 1000).toFixed(1)} กม.</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 4. Right Map Panel */}
      <div id="map-container" style={{ flex: 1, position: 'relative', height: '100%' }}>
        
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
                  square rather than a real contour (see hasRealDetection). */}
              {showSlick && hasRealDetection(selectedDet) && selectedDet.geojson_mask && selectedDet.geojson_mask.coordinates && (
                selectedDet.geojson_mask.coordinates.map((poly, pIdx) => {
                  // GeoJSON holds [lon, lat], Leaflet needs [lat, lon]
                  const leafPositions = poly.map(pt => [pt[1], pt[0]]);
                  return (
                    <Polygon
                      key={pIdx}
                      positions={leafPositions}
                      pathOptions={{
                        color: '#facc15',
                        weight: 3,
                        opacity: 1,
                        lineJoin: 'round',
                        fillColor: '#d946ef',
                        fillOpacity: 0.35
                      }}
                    />
                  );
                })
              )}

              {/* C. AIS Vessel Markers & Lines */}
              {showAIS && selectedDet.nearby_vessels && selectedDet.nearby_vessels.map(v => {
                const slickCenter = getMapCenter();
                return (
                  <React.Fragment key={v.id}>
                    {/* Dashed line to centroid */}
                    <Polyline 
                      positions={[[v.latitude, v.longitude], slickCenter]}
                      pathOptions={{
                        color: 'rgba(245, 158, 11, 0.4)',
                        weight: 1.5,
                        dashArray: '3, 6'
                      }}
                    />
                    {/* Vessel point marker */}
                    <CircleMarker 
                      center={[v.latitude, v.longitude]}
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
                        ห่าง: {(v.distance_meters / 1000).toFixed(1)} กม. (km)
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
        </div>

      </div>

    </div>
  );
}
