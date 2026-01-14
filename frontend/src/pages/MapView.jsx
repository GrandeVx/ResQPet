import { useState, useEffect } from 'react'
import { Map, Camera, Video, X, MapPin, Activity } from 'lucide-react'
import { CameraMap } from '../components/map/MapView'
import clsx from 'clsx'

export default function MapViewPage({ socket }) {
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState(null)
  const [loading, setLoading] = useState(true)
  const [alerts, setAlerts] = useState({})

  useEffect(() => {
    fetchCameras()
  }, [])

  // Listen for alerts
  useEffect(() => {
    if (socket) {
      const handleAlert = (data) => {
        const cameraId = data.camera_id || data.data?.camera_id
        if (cameraId) {
          setAlerts(prev => ({
            ...prev,
            [cameraId]: {
              stray_index: data.stray_index || data.data?.stray_index,
              timestamp: new Date()
            }
          }))
        }
      }

      socket.on('alert', handleAlert)
      return () => socket.off('alert', handleAlert)
    }
  }, [socket])

  const fetchCameras = async () => {
    try {
      const res = await fetch('/api/demo/cameras')
      const data = await res.json()
      setCameras(data.cameras || [])
    } catch (err) {
      console.error('Failed to fetch cameras:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleCameraSelect = (camera) => {
    setSelectedCamera(camera)
  }

  if (loading) {
    return (
      <div className="h-[calc(100vh-120px)] bg-surface-800 rounded-2xl animate-pulse" />
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Mappa Telecamere</h1>
          <p className="text-surface-400 mt-1">
            {cameras.length} telecamere nel territorio - Salerno e provincia
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Legend */}
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-brand-500" />
              <span className="text-surface-400">Online</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-danger animate-pulse" />
              <span className="text-surface-400">Alert</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-surface-500" />
              <span className="text-surface-400">Offline</span>
            </div>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="relative">
        <div className={clsx(
          'transition-all duration-300',
          selectedCamera ? 'h-[calc(100vh-400px)]' : 'h-[calc(100vh-200px)]'
        )}>
          <CameraMap
            cameras={cameras}
            onCameraSelect={handleCameraSelect}
            center={[40.6824, 14.7681]}
            zoom={13}
            selectedCamera={selectedCamera}
            alerts={alerts}
          />
        </div>

        {/* Selected Camera Panel */}
        {selectedCamera && (
          <div className="mt-4 bg-surface-900/95 backdrop-blur-sm rounded-2xl border border-surface-800 overflow-hidden animate-fade-in-up">
            <div className="flex items-center justify-between p-4 border-b border-surface-800">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-brand-500/20 rounded-xl">
                  <Camera className="h-5 w-5 text-brand-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">{selectedCamera.name}</h3>
                  <div className="flex items-center gap-2 text-sm text-surface-400">
                    <MapPin className="h-3 w-3" />
                    <span>{selectedCamera.zone}</span>
                    <span className="text-surface-600">|</span>
                    <span className={clsx(
                      selectedCamera.status === 'active' ? 'text-success' : 'text-danger'
                    )}>
                      {selectedCamera.status === 'active' ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setSelectedCamera(null)}
                className="p-2 rounded-xl text-surface-400 hover:text-white hover:bg-surface-800 transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="p-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Video Feed */}
                <div className="aspect-video rounded-xl overflow-hidden bg-surface-950">
                  <video
                    src={selectedCamera.video_url}
                    className="w-full h-full object-cover"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                </div>

                {/* Camera Info */}
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-surface-800/50 rounded-xl p-3">
                      <p className="text-xs text-surface-400 mb-1">Camera ID</p>
                      <p className="text-white font-mono">{selectedCamera.id}</p>
                    </div>
                    <div className="bg-surface-800/50 rounded-xl p-3">
                      <p className="text-xs text-surface-400 mb-1">Status</p>
                      <p className={clsx(
                        'font-medium',
                        selectedCamera.status === 'active' ? 'text-success' : 'text-danger'
                      )}>
                        {selectedCamera.status === 'active' ? 'Operativa' : 'Non disponibile'}
                      </p>
                    </div>
                    <div className="bg-surface-800/50 rounded-xl p-3">
                      <p className="text-xs text-surface-400 mb-1">Latitudine</p>
                      <p className="text-white font-mono text-sm">{selectedCamera.location.lat.toFixed(4)}</p>
                    </div>
                    <div className="bg-surface-800/50 rounded-xl p-3">
                      <p className="text-xs text-surface-400 mb-1">Longitudine</p>
                      <p className="text-white font-mono text-sm">{selectedCamera.location.lng.toFixed(4)}</p>
                    </div>
                  </div>

                  {alerts[selectedCamera.id] && (
                    <div className="bg-danger/10 border border-danger/30 rounded-xl p-3">
                      <div className="flex items-center gap-2 text-danger">
                        <Activity className="h-4 w-4" />
                        <span className="text-sm font-medium">Alert Attivo</span>
                      </div>
                      <p className="text-xs text-danger/70 mt-1">
                        Stray Index: {alerts[selectedCamera.id].stray_index?.toFixed(2) || 'N/A'}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
