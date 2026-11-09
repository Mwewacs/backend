import { Map, NavigationControl, AttributionControl } from './vendor/maplibre-gl.mjs';
const message = document.getElementById('message');
const countryBounds = [[21.8, -18.2], [33.8, -8.1]];
const emptyCollection = { type: 'FeatureCollection', features: [] };
let map;
let geometry;
let currentState = { districts: [], mode: 'districts', selectedName: null, resetNumber: 0 };
let focusedName = null;
let resetNumber = 0;

function sendEvent(event) {
  const payload = { source: 'cholera-map', ...event };
  if (window.ReactNativeWebView) {
    window.ReactNativeWebView.postMessage(JSON.stringify(payload));
    return;
  }
  window.parent.postMessage(payload, '*');
}

function showStatus(value) {
  message.textContent = value;
  message.hidden = !value;
  sendEvent({ type: 'status', message: value });
}

function receiveState(event) {
  if (event.source && event.source !== window.parent) return;
  let payload;
  try {
    payload = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
  } catch {
    return;
  }
  if (payload?.source !== 'cholera-app' || payload.type !== 'state' || !Array.isArray(payload.state?.districts)) return;
  currentState = payload.state;
  renderForecasts();
}

window.addEventListener('message', receiveState);
document.addEventListener('message', receiveState);
sendEvent({ type: 'ready' });

function renderForecasts() {
  if (!geometry || !map?.getSource('districts')) return;
  const forecasts = new globalThis.Map(currentState.districts.map(district => [district.name, district]));
  const features = geometry.features.map(feature => {
    const forecast = forecasts.get(feature.properties.name);
    return { ...feature, properties: { ...feature.properties, color: forecast?.visible ? forecast.color : '#E9EFF3', text_color: forecast?.visible ? forecast.textColor : '#527491', category: forecast?.visible ? forecast.category : null, label: forecast?.visible ? forecast.label : '', score: forecast?.visible ? forecast.score : null, selected: feature.properties.name === currentState.selectedName, available: Boolean(forecast?.visible) } };
  });
  map.getSource('districts').setData({ type: 'FeatureCollection', features });
  map.getSource('centres').setData({ type: 'FeatureCollection', features: features.filter(feature => feature.properties.available).map(feature => ({ type: 'Feature', properties: feature.properties, geometry: { type: 'Point', coordinates: feature.properties.center } })) });
  const heatmap = currentState.mode === 'heatmap';
  map.setLayoutProperty('forecast-heatmap', 'visibility', heatmap ? 'visible' : 'none');
  const colors = currentState.heatmapColors;
  if (colors?.length === 4) map.setPaintProperty('forecast-heatmap', 'heatmap-color', ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(0,0,0,0)', 0.12, colors[0], 0.35, colors[1], 0.65, colors[2], 1, colors[3]]);
  map.setPaintProperty('district-fill', 'fill-opacity', heatmap ? 0.06 : 0.76);
  if (currentState.resetNumber !== resetNumber) {
    resetNumber = currentState.resetNumber;
    focusedName = currentState.selectedName;
    map.fitBounds(countryBounds, { padding: 20, duration: 450 });
    return;
  }
  if (!currentState.selectedName) {
    focusedName = null;
    return;
  }
  if (currentState.selectedName === focusedName) return;
  focusedName = currentState.selectedName;
  const selected = features.find(feature => feature.properties.name === focusedName);
  if (!selected) return;
  const bounds = selected.geometry.coordinates.flat(2).reduce((result, point) => [Math.min(result[0], point[0]), Math.min(result[1], point[1]), Math.max(result[2], point[0]), Math.max(result[3], point[1])], [180, 90, -180, -90]);
  map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: 56, maxZoom: 8, duration: 450 });
}

async function initializeMap() {
  try {
    const response = await fetch('districts.geojson');
    if (!response.ok) throw new Error('District boundaries are unavailable.');
    geometry = await response.json();
    map = new Map({ container: 'map', style: { version: 8, sources: {}, layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#F3F9FD' } }] }, bounds: countryBounds, fitBoundsOptions: { padding: 20 }, minZoom: 3, maxZoom: 12, maxBounds: [[18, -22], [38, -5]], attributionControl: false, dragRotate: false, pitchWithRotate: false });
    map.touchZoomRotate.disableRotation();
    map.addControl(new NavigationControl({ showCompass: false }), 'bottom-right');
    map.addControl(new AttributionControl({ compact: true, customAttribution: '<a href="https://www.geoboundaries.org/api/current/gbOpen/ZMB/ADM2/" target="_blank" rel="noopener">Districts: GRID3 / OSG · CC BY 4.0</a>' }), 'bottom-left');
    map.on('style.load', () => {
      const label = map.getStyle().layers.find(layer => layer.type === 'symbol')?.id;
      map.addSource('districts', { type: 'geojson', data: emptyCollection });
      map.addSource('centres', { type: 'geojson', data: emptyCollection });
      map.addLayer({ id: 'district-fill', type: 'fill', source: 'districts', paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.76 } }, label);
      map.addLayer({ id: 'forecast-heatmap', type: 'heatmap', source: 'centres', layout: { visibility: 'none' }, paint: { 'heatmap-weight': ['get', 'score'], 'heatmap-intensity': 1, 'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 4, 18, 8, 42], 'heatmap-opacity': 0.8, 'heatmap-color': 'rgba(0,0,0,0)' } }, label);
      map.addLayer({ id: 'district-line', type: 'line', source: 'districts', paint: { 'line-color': ['case', ['get', 'selected'], '#397EBA', '#93AEBE'], 'line-width': ['case', ['get', 'selected'], 3, 0.65], 'line-opacity': 0.9 } }, label);
      if (map.getStyle().glyphs) map.addLayer({ id: 'district-label', type: 'symbol', source: 'centres', minzoom: 6, layout: { 'text-field': ['concat', ['get', 'name'], '\n', ['get', 'label']], 'text-font': ['Noto Sans Regular'], 'text-size': 11, 'text-padding': 8 }, paint: { 'text-color': ['get', 'text_color'], 'text-halo-color': '#FFFFFF', 'text-halo-width': 1.5 } });
      renderForecasts();
    });
    map.on('click', event => {
      if (!map.getLayer('district-fill')) return;
      const district = map.queryRenderedFeatures(event.point, { layers: ['district-fill'] })[0];
      if (!district) return;
      sendEvent({ type: 'select', name: district.properties.name });
    });
    map.on('mousemove', event => {
      if (!map.getLayer('district-fill')) return;
      map.getCanvas().style.cursor = map.queryRenderedFeatures(event.point, { layers: ['district-fill'] }).length ? 'pointer' : '';
    });
    map.on('error', event => {
      if (!event.error?.message?.includes('WebGL')) return;
      showStatus('This device cannot draw the map. Use the area list.');
    });
    showStatus('');
    try {
      const response = await fetch('https://tiles.openfreemap.org/styles/positron', { signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw new Error('Basemap unavailable');
      const style = await response.json();
      map.setStyle(style);
    } catch {
      showStatus('Street map unavailable. District forecasts are still shown.');
    }
  } catch {
    showStatus('The map could not load. Use the area list or refresh.');
  }
}

void initializeMap();
