import { useEffect, useState, useRef, useCallback } from 'react'
import { io } from 'socket.io-client'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:5001'

export function useSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [detections, setDetections] = useState([])
  const [alerts, setAlerts] = useState([]) // Session alerts history
  const [frames, setFrames] = useState({}) // camera_id -> frame_base64
  const socketRef = useRef(null)

  useEffect(() => {
    // Initialize socket connection
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    })

    socketRef.current = socket

    // Connection events
    socket.on('connect', () => {
      console.log('[Socket] Connected')
      setIsConnected(true)
    })

    socket.on('disconnect', () => {
      console.log('[Socket] Disconnected')
      setIsConnected(false)
    })

    socket.on('connection_status', (data) => {
      console.log('[Socket] Status:', data.message)
    })

    // Detection events
    socket.on('detection', (data) => {
      setDetections(prev => {
        // Keep last 100 detections
        const updated = [data.data, ...prev].slice(0, 100)
        return updated
      })
    })

    // Frame updates
    socket.on('frame_update', (data) => {
      setFrames(prev => ({
        ...prev,
        [data.camera_id]: {
          frame: data.frame,
          detections: data.detections,
          timestamp: data.timestamp
        }
      }))
    })

    // Alert events - store in session history
    socket.on('alert', (data) => {
      console.log('[Socket] Alert received:', data)

      // Add to alerts history
      const newAlert = {
        id: Date.now(),
        timestamp: data.timestamp || new Date().toISOString(),
        camera_id: data.data?.camera_id,
        stray_index: data.data?.stray_index,
        status: data.data?.status,
        breed: data.data?.breed,
        snapshot: data.data?.snapshot,
        bbox: data.data?.bbox,
        components: data.data?.components
      }

      setAlerts(prev => [newAlert, ...prev].slice(0, 100)) // Keep last 100

      // Play alert sound
      playAlertSound()
    })

    // Cleanup on unmount
    return () => {
      socket.disconnect()
    }
  }, [])

  // Join camera room
  const joinCamera = useCallback((cameraId) => {
    if (socketRef.current) {
      socketRef.current.emit('join_camera', { camera_id: cameraId })
    }
  }, [])

  // Leave camera room
  const leaveCamera = useCallback((cameraId) => {
    if (socketRef.current) {
      socketRef.current.emit('leave_camera', { camera_id: cameraId })
    }
  }, [])

  // Start analysis for camera
  const startAnalysis = useCallback((cameraId) => {
    if (socketRef.current) {
      socketRef.current.emit('start_analysis', { camera_id: cameraId })
    }
  }, [])

  // Stop analysis for camera
  const stopAnalysis = useCallback((cameraId) => {
    if (socketRef.current) {
      socketRef.current.emit('stop_analysis', { camera_id: cameraId })
    }
  }, [])

  return {
    socket: socketRef.current,
    isConnected,
    detections,
    alerts,
    frames,
    joinCamera,
    leaveCamera,
    startAnalysis,
    stopAnalysis,
  }
}

// Play alert sound
function playAlertSound() {
  try {
    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdH2Onq+2wrHHuMbR2dXQ1M3P0s/V1crKys7Qzs/Lx8TDw8PExcXGx8fIycrKy8zNzc7P0NHS09PU1dbW19jZ2tvc3d7f4OHi4+Tk5ebn6Onq6+zt7u/w8fLz9PX29/j5+vv8/f7/')
    audio.volume = 0.5
    audio.play().catch(() => {})
  } catch (e) {
    // Ignore audio errors
  }
}

export default useSocket
