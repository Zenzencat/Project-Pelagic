import React, { useEffect, useState } from 'react';

function PreviewImage({ src, alt, apiBase, style }) {
  const hasSource = typeof src === 'string' && src.length > 0;
  const [unavailable, setUnavailable] = useState(!hasSource);
  useEffect(() => setUnavailable(!hasSource), [hasSource, src]);
  if (!hasSource || unavailable) return <p role="status">Preview unavailable.</p>;
  const resolvedSrc = src.startsWith('data:') || /^https?:\/\//.test(src)
    ? src
    : `${apiBase}${src.startsWith('/') ? src : `/${src}`}`;
  return <img src={resolvedSrc} alt={alt} onError={() => setUnavailable(true)} style={style} />;
}

export function WindEvidence({ evidence }) {
  const available = evidence?.status === 'available' && Number.isFinite(evidence.wind_speed_ms);
  return (
    <section aria-label="ERA5 wind evidence" aria-live="polite" style={{ borderTop: '1px solid var(--border-color)', paddingTop: 10, fontSize: '0.78rem', lineHeight: 1.5 }}>
      <strong>ERA5 wind at acquisition</strong>
      {available ? <>
        <p>{evidence.wind_speed_ms.toFixed(2)} m/s</p>
        <p>Record (UTC): {evidence.valid_time_utc}</p>
        <p>Grid cell: {evidence.grid_lat}°, {evidence.grid_lon}°</p>
      </> : <p>{evidence?.status || 'skipped'}: {evidence?.reason || 'Wind evidence has not been requested for this observation.'}</p>}
      <p style={{ color: 'var(--text-muted)' }}>{evidence?.resolution_note || 'ERA5 is hourly, on a coarse 0.25° grid (~28 km north-south). It cannot resolve wind at SAR pixel scale.'}</p>
      <p style={{ color: 'var(--text-muted)' }}>Supplementary evidence for review; no oil classification is inferred.</p>
    </section>
  );
}

function SarObservation({ observation, label, apiBase }) {
  return <figure style={{ margin: 0, minWidth: 0 }}>
    <figcaption>
      <strong>{label}</strong>
      <p>Acquired (UTC): {observation.acquisition_start_utc}</p>
      <p>{observation.predicted_pixel_count.toLocaleString()} predicted pixels · mean confidence {(observation.confidence_score * 100).toFixed(1)}%</p>
    </figcaption>
    <PreviewImage src={observation.overlay_preview} apiBase={apiBase}
      alt={`${label}, acquired ${observation.acquisition_start_utc}, VV SAR with segmentation overlay`}
      style={{ width: '100%', maxHeight: 280, objectFit: 'contain', background: '#000' }} />
    <p>{observation.preview_note}</p>
    {observation.preview_error && <p role="status">{observation.preview_error}</p>}
    <details><summary>Observation source</summary><p style={{ overflowWrap: 'anywhere' }}>{observation.product.name}</p></details>
  </figure>;
}

export function ObservationEvidence({ evidence, apiBase = '' }) {
  const temporal = evidence.temporal;
  const optical = evidence.optical;
  return <details style={{ position: 'absolute', bottom: 16, left: 16, right: 16, zIndex: 1000,
    background: 'var(--bg-dark)', color: 'var(--text-main)', border: '1px solid var(--border-color)',
    borderRadius: 6, padding: 12, maxHeight: '60%', overflow: 'auto', fontSize: '0.8rem', lineHeight: 1.5 }}>
    <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Observation evidence — SAR and optical</summary>
    <p aria-live="polite">Comparison: {temporal?.status || 'skipped'} — {temporal?.reason || 'Not requested.'}</p>
    <p aria-live="polite">Optical: {optical?.status || 'skipped'} — {optical?.reason || 'Not requested.'}</p>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: 16 }}>
      <SarObservation observation={evidence.original} apiBase={apiBase} label="Original Sentinel-1" />
      {temporal?.status === 'available' && temporal.observations.map(observation =>
        <SarObservation key={observation.scene_id} observation={observation} apiBase={apiBase} label="Comparison Sentinel-1" />)}
      {optical?.status === 'available' && <figure style={{ margin: 0, minWidth: 0 }}>
        <figcaption><strong>Sentinel-2 L2A true color</strong>
          <p>Acquired (UTC): {optical.observation.acquisition_start_utc}</p>
          <p>Catalog cloud cover: {optical.observation.cloud_cover_pct}% (whole tile)</p>
        </figcaption>
        <PreviewImage src={optical.observation.rgb_preview} apiBase={apiBase}
          alt={`Sentinel-2 true color acquired ${optical.observation.acquisition_start_utc}, same geographic region as SAR`}
          style={{ width: '100%', maxHeight: 280, objectFit: 'contain', background: '#000' }} />
        {optical.observation.preview_error && <p role="status">{optical.observation.preview_error}</p>}
        <p>Cloud filtering does not guarantee this crop is cloud-free. No cloud removal or fusion applied.</p>
        <details><summary>Observation source</summary><p style={{ overflowWrap: 'anywhere' }}>{optical.observation.product.name}</p></details>
      </figure>}
    </div>
    <p>These observations are evidence for human review. Dates and imaging conditions differ; no persistence or oil verdict is inferred.</p>
  </details>;
}
