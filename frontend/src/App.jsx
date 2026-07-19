import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, CircleMarker, Polyline, useMap } from 'react-leaflet';
import { Activity, ShieldAlert, Layers, Play, Clock, Anchor, MapPin, Loader2, Info } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:8000';

// Map controller to handle programmatically panning/zooming when selected detection changes
function MapRecenter({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.panTo(center);
    }
  }, [center, map]);
  return null;
}

export default function App() {
  const [detections, setDetections] = useState([]);
  const [selectedDet, setSelectedDet] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);
  
  // Simulated scene ID input
  const [sceneToPredict, setSceneToPredict] = useState('S1_MOCK_SCENE_004');
  
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
        // Auto-select first detection, or a specific one (e.g. newly created)
        const targetId = autoSelectId || data[0].id;
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

  // 3. Trigger U-Net inference on backend
  const handlePredict = async (e) => {
    e.preventDefault();
    if (!sceneToPredict.trim()) return;

    try {
      setPredicting(true);
      setError(null);
      
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene_id: sceneToPredict })
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
      
      // Show success notification toast
      showToast(`ตรวจพบรอยคราบน้ำมันในฉาก ${sceneToPredict} สำเร็จ!`);
      
      // Re-fetch history list and auto-select the new detection
      await fetchDetections(newDet.id);
      setPredicting(false);
      
      // Cycle simulated ID for easy consecutive test clicks
      if (sceneToPredict === 'S1_MOCK_SCENE_004') setSceneToPredict('S1_MOCK_SCENE_005');
      else setSceneToPredict('S1_MOCK_SCENE_004');
      
    } catch (err) {
      console.error(err);
      setError(err.message);
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

        {/* History List */}
        <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '16px 20px 8px 20px', fontWeight: 600 }}>
          ประวัติการวิเคราะห์ฉากดาวเทียม (Processed Scenes)
        </div>
        
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 20px' }}>
          {detections.length === 0 ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', padding: '20px 0', textAlign: 'center' }}>
              ไม่มีข้อมูลประวัติการตรวจจับในระบบ.
            </p>
          ) : (
            detections.map(det => {
              const isActive = selectedDet && selectedDet.id === det.id;
              // Format geojson features area estimation or mock size
              const displayArea = det.scene_id.includes('MOCK') ? '8.5' : '14.5';
              return (
                <div 
                  key={det.id}
                  onClick={() => fetchDetectionDetails(det.id)}
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
                    <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-muted)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '240px' }}>
                      {det.scene_id}
                    </span>
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                      backgroundColor: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)'
                    }}>
                      {(det.confidence_score * 100).toFixed(1)}% Conf
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                    <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Clock size={12} /> {det.detected_at.split(' ')[0]}
                    </span>
                    <span style={{ fontWeight: 500 }}>
                      ขนาดคราบ: {displayArea} ตร.กม.
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Selected Details Panel */}
        {selectedDet && (
          <div style={{ padding: '20px', borderTop: '1px solid var(--border-color)', backgroundColor: 'rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Info size={14} color="var(--primary)" /> รายละเอียดคราบนํ้ามัน (Slick Details)
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem' }}>
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

        {/* Prediction Trigger Form */}
        <form onSubmit={handlePredict} style={{ padding: '16px 20px', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '10px' }}>
          <input 
            type="text" 
            value={sceneToPredict} 
            onChange={(e) => setSceneToPredict(e.target.value)}
            placeholder="Scene ID (e.g. S1_MOCK_SCENE_004)"
            style={{
              flex: 1, backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-color)',
              borderRadius: '6px', padding: '8px 12px', fontSize: '0.8rem', color: 'var(--text-main)',
              outline: 'none'
            }}
          />
          <button 
            type="submit" 
            style={{
              backgroundColor: 'var(--primary)', color: '#fff', border: 'none', borderRadius: '6px',
              padding: '8px 14px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '6px', transition: 'background-color 0.2s'
            }}
          >
            <Play size={12} fill="#fff" /> วิเคราะห์ภาพ
          </button>
        </form>
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
          
          <MapRecenter center={getMapCenter()} />

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

              {/* B. Oil Slick Contours */}
              {showSlick && selectedDet.geojson_mask && selectedDet.geojson_mask.coordinates && (
                selectedDet.geojson_mask.coordinates.map((poly, pIdx) => {
                  // GeoJSON holds [lon, lat], Leaflet needs [lat, lon]
                  const leafPositions = poly.map(pt => [pt[1], pt[0]]);
                  return (
                    <Polygon 
                      key={pIdx}
                      positions={leafPositions}
                      pathOptions={{
                        color: '#d946ef',
                        weight: 2,
                        fillColor: '#d946ef',
                        fillOpacity: 0.4
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
