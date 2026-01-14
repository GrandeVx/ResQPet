import { useRef, useEffect, useState, useCallback } from 'react'
import { Camera, MapPin, Maximize2, Volume2, VolumeX, AlertTriangle, Play, Pause, Dog } from 'lucide-react'
import clsx from 'clsx'
import StrayIndexBadge from '../StrayIndexBadge'

export function CameraCard({ camera, socket, onExpand, isExpanded = false }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const intervalRef = useRef(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [hasError, setHasError] = useState(false)
  const [isMuted, setIsMuted] = useState(true)

  // Detection state
  const [lastDetection, setLastDetection] = useState(null)
  const [detectionCount, setDetectionCount] = useState(0)

  // Start video playback
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const playVideo = async () => {
      try {
        video.muted = true
        video.loop = true
        video.playsInline = true
        await video.play()
        setIsPlaying(true)
        setHasError(false)
      } catch (err) {
        console.warn(`[CameraCard] Video play error for ${camera.id}:`, err.message)
        setHasError(true)
      }
    }

    if (video.readyState >= 2) {
      playVideo()
    } else {
      video.addEventListener('loadeddata', playVideo, { once: true })
    }

    return () => {
      video.removeEventListener('loadeddata', playVideo)
    }
  }, [camera.video_url, camera.id])

  // Listen for detection results from backend
  useEffect(() => {
    if (!socket) return

    const handleDetectionResult = (data) => {
      if (data.camera_id === camera.id && data.detections?.length > 0) {
        setDetectionCount(data.detections.length)
        setLastDetection(data.detections[0])

        // Clear detection overlay after 3 seconds
        setTimeout(() => {
          setLastDetection(null)
          setDetectionCount(0)
        }, 3000)
      }
    }

    socket.on('detection_result', handleDetectionResult)

    return () => {
      socket.off('detection_result', handleDetectionResult)
    }
  }, [socket, camera.id])

  // Capture and send frame to backend
  const captureAndSendFrame = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !socket) return

    const video = videoRef.current
    const canvas = canvasRef.current

    // Only capture if video is ready
    if (video.readyState < 2) return

    try {
      const ctx = canvas.getContext('2d')
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert to base64 JPEG
      const frameData = canvas.toDataURL('image/jpeg', 0.7)
      const base64Data = frameData.split(',')[1]

      // Send to backend via WebSocket
      socket.emit('analyze_frame', {
        camera_id: camera.id,
        frame: base64Data,
        timestamp: new Date().toISOString()
      })
    } catch (err) {
      console.error('[CameraCard] Frame capture error:', err)
    }
  }, [socket, camera.id])

  // Start/stop analysis
  const toggleAnalysis = () => {
    if (isAnalyzing) {
      // Stop analysis
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      setIsAnalyzing(false)
    } else {
      // Start analysis - capture frame every 500ms (2 FPS)
      setIsAnalyzing(true)
      captureAndSendFrame() // Capture immediately
      intervalRef.current = setInterval(captureAndSendFrame, 500)
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  const hasStrayAlert = lastDetection?.stray_index >= 0.7
  const hasWarning = lastDetection?.stray_index >= 0.3 && lastDetection?.stray_index < 0.7

  return (
    <div
      className={clsx(
        'group relative bg-surface-900 rounded-2xl border overflow-hidden transition-all duration-300',
        hasStrayAlert ? 'border-danger ring-2 ring-danger/30 shadow-glow-danger' :
        hasWarning ? 'border-warning ring-1 ring-warning/20' :
        'border-surface-800 hover:border-surface-700',
        isExpanded && 'col-span-2 row-span-2'
      )}
    >
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Video Feed */}
      <div className="relative aspect-video bg-surface-950">
        {!hasError ? (
          <video
            ref={videoRef}
            src={camera.video_url}
            className="w-full h-full object-cover"
            muted={isMuted}
            loop
            playsInline
            onError={() => setHasError(true)}
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-surface-500 gap-2">
            <Camera className="h-10 w-10 opacity-50" />
            <span className="text-xs">Video non disponibile</span>
          </div>
        )}

        {/* Gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-black/60 pointer-events-none" />

        {/* Alert flash overlay */}
        {hasStrayAlert && (
          <div className="absolute inset-0 bg-danger/20 animate-pulse pointer-events-none" />
        )}

        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 p-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Live/Analyzing indicator */}
            <div className={clsx(
              'flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium backdrop-blur-sm',
              isAnalyzing
                ? 'bg-success/20 text-success border border-success/30'
                : camera.status === 'active'
                  ? 'bg-brand-500/20 text-brand-400'
                  : 'bg-danger/20 text-danger'
            )}>
              <span className={clsx(
                'h-1.5 w-1.5 rounded-full',
                isAnalyzing ? 'bg-success animate-pulse' :
                camera.status === 'active' ? 'bg-brand-400' : 'bg-danger'
              )} />
              <span>{isAnalyzing ? 'ANALYZING' : camera.status === 'active' ? 'LIVE' : 'OFFLINE'}</span>
            </div>

            {/* Stray Alert Badge */}
            {hasStrayAlert && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-danger/30 text-danger text-xs font-bold backdrop-blur-sm animate-bounce">
                <AlertTriangle className="h-3 w-3" />
                <span>STRAY ALERT</span>
              </div>
            )}

            {hasWarning && !hasStrayAlert && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-warning/30 text-warning text-xs font-bold backdrop-blur-sm">
                <AlertTriangle className="h-3 w-3" />
                <span>POSSIBLY LOST</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="p-1.5 rounded-lg bg-black/40 text-white/70 hover:text-white hover:bg-black/60 transition-all opacity-0 group-hover:opacity-100 backdrop-blur-sm"
            >
              {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            </button>
            <button
              onClick={() => onExpand?.(camera)}
              className="p-1.5 rounded-lg bg-black/40 text-white/70 hover:text-white hover:bg-black/60 transition-all opacity-0 group-hover:opacity-100 backdrop-blur-sm"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Detection overlay */}
        {lastDetection && (
          <div className="absolute top-14 right-3 bg-black/80 backdrop-blur-sm rounded-xl p-3 animate-fade-in">
            <div className="flex items-center gap-3">
              <Dog className="h-6 w-6 text-white" />
              <div>
                <p className="text-xs text-white font-medium">
                  {detectionCount} cane{detectionCount !== 1 ? 'i' : ''} rilevato{detectionCount !== 1 ? 'i' : ''}
                </p>
                <StrayIndexBadge
                  strayIndex={lastDetection.stray_index}
                  status={lastDetection.status}
                />
                {lastDetection.breed && (
                  <p className="text-xs text-surface-400 mt-1">
                    Razza: {lastDetection.breed}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Bottom info */}
        <div className="absolute bottom-0 left-0 right-0 p-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-white text-sm truncate">
                {camera.name}
              </h3>
              <div className="flex items-center gap-1.5 mt-1">
                <MapPin className="h-3 w-3 text-surface-400" />
                <span className="text-xs text-surface-400">{camera.zone}</span>
              </div>
            </div>

            {/* Analysis toggle button */}
            <button
              onClick={toggleAnalysis}
              disabled={!socket || hasError}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                isAnalyzing
                  ? 'bg-danger/80 hover:bg-danger text-white'
                  : 'bg-success/80 hover:bg-success text-white',
                (!socket || hasError) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {isAnalyzing ? (
                <>
                  <Pause className="h-3.5 w-3.5" />
                  <span>Stop</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" />
                  <span>Analizza</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CameraCard
