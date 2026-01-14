import { useEffect } from 'react'
import CameraFeed from './CameraFeed'

function CCTVGrid({ cameras, socket, detections }) {
  // Join all camera rooms on mount
  useEffect(() => {
    if (socket && cameras.length > 0) {
      cameras.forEach(camera => {
        socket.emit('join_camera', { camera_id: camera.id })
      })
    }
  }, [socket, cameras])

  // Group detections by camera
  const detectionsByCamera = detections.reduce((acc, det) => {
    const cameraId = det.camera_id || 'unknown'
    if (!acc[cameraId]) acc[cameraId] = []
    acc[cameraId].push(det)
    return acc
  }, {})

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Live Cameras</h2>
        <span className="text-sm text-gray-400">
          {cameras.filter(c => c.status === 'active').length} active
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {cameras.map(camera => (
          <CameraFeed
            key={camera.id}
            camera={camera}
            detections={detectionsByCamera[camera.id] || []}
            socket={socket}
          />
        ))}
      </div>
    </div>
  )
}

export default CCTVGrid
