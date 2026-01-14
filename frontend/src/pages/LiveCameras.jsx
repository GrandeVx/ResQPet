import { useState, useEffect } from 'react'
import { Video, Grid3X3, LayoutGrid, RefreshCw } from 'lucide-react'
import { CameraCard } from '../components/cameras/CameraCard'
import clsx from 'clsx'

export default function LiveCameras({ socket, detections }) {
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [gridSize, setGridSize] = useState('3x3') // '2x2' or '3x3'
  const [expandedCamera, setExpandedCamera] = useState(null)

  useEffect(() => {
    fetchCameras()
  }, [])

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

  // Get latest detection for each camera
  const getDetectionForCamera = (cameraId) => {
    return detections?.find(d => d.camera_id === cameraId)
  }

  const handleExpand = (camera) => {
    setExpandedCamera(expandedCamera?.id === camera.id ? null : camera)
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 bg-surface-800 rounded-lg w-48 animate-pulse" />
        <div className={clsx(
          'grid gap-4',
          gridSize === '3x3' ? 'grid-cols-3' : 'grid-cols-2'
        )}>
          {[...Array(gridSize === '3x3' ? 9 : 4)].map((_, i) => (
            <div key={i} className="aspect-video bg-surface-800 rounded-2xl animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  const displayedCameras = gridSize === '2x2' ? cameras.slice(0, 4) : cameras

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Cameras</h1>
          <p className="text-surface-400 mt-1">
            <span className="text-success font-medium">
              {cameras.filter(c => c.status === 'active').length}
            </span>
            {' '}telecamere attive su {cameras.length}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Refresh */}
          <button
            onClick={fetchCameras}
            className="p-2 rounded-xl bg-surface-800 text-surface-400 hover:text-white hover:bg-surface-700 transition-colors"
            title="Aggiorna"
          >
            <RefreshCw className="h-5 w-5" />
          </button>

          {/* Grid Size Toggle */}
          <div className="flex items-center bg-surface-800 rounded-xl p-1">
            <button
              onClick={() => setGridSize('2x2')}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                gridSize === '2x2'
                  ? 'bg-brand-600 text-white'
                  : 'text-surface-400 hover:text-white'
              )}
              title="Grid 2x2"
            >
              <LayoutGrid className="h-4 w-4" />
            </button>
            <button
              onClick={() => setGridSize('3x3')}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                gridSize === '3x3'
                  ? 'bg-brand-600 text-white'
                  : 'text-surface-400 hover:text-white'
              )}
              title="Grid 3x3"
            >
              <Grid3X3 className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Expanded Camera View */}
      {expandedCamera && (
        <div className="mb-6">
          <div className="bg-surface-900 rounded-2xl border border-surface-800 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Video className="h-5 w-5 text-brand-400" />
                <div>
                  <h3 className="font-semibold text-white">{expandedCamera.name}</h3>
                  <p className="text-sm text-surface-400">{expandedCamera.zone}</p>
                </div>
              </div>
              <button
                onClick={() => setExpandedCamera(null)}
                className="text-surface-400 hover:text-white text-sm"
              >
                Chiudi
              </button>
            </div>
            <div className="aspect-video rounded-xl overflow-hidden">
              <video
                src={expandedCamera.video_url}
                className="w-full h-full object-cover"
                autoPlay
                muted
                loop
                playsInline
              />
            </div>
          </div>
        </div>
      )}

      {/* Camera Grid */}
      <div className={clsx(
        'grid gap-4',
        gridSize === '3x3'
          ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
          : 'grid-cols-1 md:grid-cols-2'
      )}>
        {displayedCameras.map((camera) => (
          <CameraCard
            key={camera.id}
            camera={camera}
            socket={socket}
            onExpand={handleExpand}
          />
        ))}
      </div>

      {/* Show more indicator for 2x2 */}
      {gridSize === '2x2' && cameras.length > 4 && (
        <div className="text-center py-4">
          <button
            onClick={() => setGridSize('3x3')}
            className="text-brand-400 hover:text-brand-300 text-sm font-medium"
          >
            Mostra tutte le {cameras.length} telecamere
          </button>
        </div>
      )}
    </div>
  )
}
