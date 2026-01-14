import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { Camera, MapPin, Activity } from 'lucide-react'
import clsx from 'clsx'

// Fix default marker icon issue with webpack/vite
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Custom camera marker icon
const createCameraIcon = (status, hasAlert = false) => {
  const color = hasAlert ? '#ef4444' : status === 'active' ? '#0ea5e9' : '#64748b'
  const pulseClass = status === 'active' ? 'animate-ping' : ''

  return L.divIcon({
    className: 'custom-camera-marker',
    html: `
      <div class="relative flex items-center justify-center">
        <div class="absolute w-10 h-10 rounded-full ${hasAlert ? 'bg-red-500/30 animate-ping' : ''}" style="animation-duration: 1.5s;"></div>
        <div class="relative w-10 h-10 rounded-full flex items-center justify-center shadow-lg border-2 border-white/90" style="background-color: ${color};">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        ${status === 'active' ? `<span class="absolute -top-0.5 -right-0.5 h-3 w-3 bg-green-500 rounded-full border-2 border-white"></span>` : ''}
      </div>
    `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20],
  })
}

// Component to fit map bounds
function FitBounds({ cameras }) {
  const map = useMap()

  useEffect(() => {
    if (cameras.length > 0) {
      const bounds = cameras.map(cam => [cam.location.lat, cam.location.lng])
      map.fitBounds(bounds, { padding: [50, 50] })
    }
  }, [cameras, map])

  return null
}

export function CameraMap({
  cameras,
  onCameraSelect,
  center = [40.6824, 14.7681],
  zoom = 13,
  selectedCamera = null,
  alerts = {}
}) {
  return (
    <div className="h-full w-full rounded-2xl overflow-hidden border border-surface-800 shadow-lg">
      <MapContainer
        center={center}
        zoom={zoom}
        className="h-full w-full"
        style={{ background: '#0f172a' }}
        zoomControl={false}
      >
        {/* Dark theme tiles */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Auto-fit to camera bounds */}
        {cameras.length > 0 && <FitBounds cameras={cameras} />}

        {/* Camera markers */}
        {cameras.map((camera) => {
          const hasAlert = alerts[camera.id]?.stray_index >= 0.7

          return (
            <Marker
              key={camera.id}
              position={[camera.location.lat, camera.location.lng]}
              icon={createCameraIcon(camera.status, hasAlert)}
              eventHandlers={{
                click: () => onCameraSelect?.(camera),
              }}
            >
              <Popup className="camera-popup">
                <div className="p-1 min-w-[180px]">
                  <div className="flex items-center gap-2 mb-2">
                    <div className={clsx(
                      'p-1.5 rounded-lg',
                      camera.status === 'active' ? 'bg-brand-500/20' : 'bg-surface-700'
                    )}>
                      <Camera className={clsx(
                        'h-4 w-4',
                        camera.status === 'active' ? 'text-brand-400' : 'text-surface-400'
                      )} />
                    </div>
                    <div>
                      <h4 className="font-semibold text-surface-100 text-sm">{camera.name}</h4>
                      <p className="text-xs text-surface-400">{camera.zone}</p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-2 border-t border-surface-700">
                    <div className="flex items-center gap-1.5">
                      <span className={clsx(
                        'h-2 w-2 rounded-full',
                        camera.status === 'active' ? 'bg-success animate-pulse' : 'bg-danger'
                      )} />
                      <span className="text-xs text-surface-300">
                        {camera.status === 'active' ? 'Online' : 'Offline'}
                      </span>
                    </div>

                    <button
                      onClick={() => onCameraSelect?.(camera)}
                      className="text-xs text-brand-400 hover:text-brand-300 font-medium"
                    >
                      View Feed
                    </button>
                  </div>

                  {hasAlert && (
                    <div className="mt-2 px-2 py-1.5 bg-danger/20 rounded-lg text-danger text-xs font-medium flex items-center gap-1">
                      <Activity className="h-3 w-3" />
                      <span>Alert attivo</span>
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          )
        })}
      </MapContainer>
    </div>
  )
}

export default CameraMap
