import { useState, useEffect, useRef, useCallback } from 'react'
import { Camera, Video, AlertTriangle, Dog, Upload, Play, Pause, X, Film, Image } from 'lucide-react'
import StrayIndexBadge from './StrayIndexBadge'

function CameraFeed({ camera, detections, socket, onAlert }) {
  // File upload state
  const [file, setFile] = useState(null)
  const [fileType, setFileType] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  // Detection overlay (brief flash when detection occurs)
  const [lastDetection, setLastDetection] = useState(null)
  const [detectionCount, setDetectionCount] = useState(0)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const intervalRef = useRef(null)
  const fileInputRef = useRef(null)
  const imgRef = useRef(null)

  // Listen for detection results from backend (not frame updates)
  useEffect(() => {
    if (socket) {
      const handleDetectionResult = (data) => {
        if (data.camera_id === camera.id && data.detections?.length > 0) {
          // Update detection count
          setDetectionCount(data.detections.length)
          setLastDetection(data.detections[0])

          // Clear after 2 seconds
          setTimeout(() => {
            setLastDetection(null)
            setDetectionCount(0)
          }, 2000)
        }
      }

      const handleAlert = (data) => {
        if (data.data?.camera_id === camera.id && onAlert) {
          onAlert(data)
        }
      }

      socket.on('detection_result', handleDetectionResult)
      socket.on('alert', handleAlert)

      return () => {
        socket.off('detection_result', handleDetectionResult)
        socket.off('alert', handleAlert)
      }
    }
  }, [socket, camera.id, onAlert])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCapture()
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [])

  // Capture and send frame to backend for analysis (without expecting frame back)
  const captureAndSendFrame = useCallback(() => {
    if (!canvasRef.current || !socket) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    try {
      if (fileType === 'video' && videoRef.current && videoRef.current.readyState >= 2) {
        const video = videoRef.current
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      } else if (fileType === 'image' && imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0) {
        const img = imgRef.current
        canvas.width = img.naturalWidth || 640
        canvas.height = img.naturalHeight || 480
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      } else {
        return // Not ready yet
      }

      // Convert to base64 and send via WebSocket
      const frameData = canvas.toDataURL('image/jpeg', 0.7)
      const base64Data = frameData.split(',')[1]

      socket.emit('analyze_frame', {
        camera_id: camera.id,
        frame: base64Data,
        timestamp: new Date().toISOString()
      })
    } catch (err) {
      console.error('Frame capture error:', err)
    }
  }, [fileType, socket, camera.id])

  // Handle file selection
  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    stopCapture()

    const isVideo = selectedFile.type.startsWith('video/')
    const isImage = selectedFile.type.startsWith('image/')

    if (!isVideo && !isImage) {
      alert('Please select a video or image file')
      return
    }

    // Cleanup previous preview
    if (preview) URL.revokeObjectURL(preview)

    setFile(selectedFile)
    setFileType(isVideo ? 'video' : 'image')
    setPreview(URL.createObjectURL(selectedFile))
    setLastDetection(null)
    setDetectionCount(0)
  }

  // Start analysis
  const startCapture = () => {
    if (!file || !socket) return

    setIsAnalyzing(true)

    if (fileType === 'video' && videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }

    // Capture frame every 500ms (2 FPS for analysis) - video plays at normal speed
    intervalRef.current = setInterval(captureAndSendFrame, 500)
  }

  // Stop analysis
  const stopCapture = () => {
    setIsAnalyzing(false)

    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.pause()
    }
  }

  // Reset file
  const resetFile = () => {
    stopCapture()
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setFileType(null)
    setPreview(null)
    setLastDetection(null)
    setDetectionCount(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // Loop video
  const handleVideoEnded = () => {
    if (videoRef.current && isAnalyzing) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }

  const hasStrayAlert = lastDetection?.stray_index >= 0.7
  const hasPossiblyLostAlert = lastDetection?.stray_index >= 0.3 && lastDetection?.stray_index < 0.7

  return (
    <div className={`camera-feed relative ${hasStrayAlert ? 'ring-2 ring-red-500 alert-pulse' : hasPossiblyLostAlert ? 'ring-2 ring-yellow-500' : ''}`}>
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
      {/* Hidden img for image capture */}
      {fileType === 'image' && preview && (
        <img
          ref={imgRef}
          src={preview}
          className="hidden"
          alt=""
          onLoad={() => console.log('[CameraFeed] Hidden image loaded')}
        />
      )}

      {/* Camera Header */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/80 to-transparent p-3 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Camera className="h-4 w-4 text-white" />
            <span className="text-sm font-medium text-white">{camera.name}</span>
          </div>
          <div className="flex items-center space-x-2">
            {isAnalyzing && (
              <span className="flex items-center space-x-1">
                <span className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-xs text-green-400">LIVE</span>
              </span>
            )}
            {file && !isAnalyzing && (
              <span className="flex items-center space-x-1">
                <span className="h-2 w-2 bg-yellow-500 rounded-full"></span>
                <span className="text-xs text-yellow-400">READY</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Video Feed Area - Shows video/image directly, not analyzed frames */}
      <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
        {preview ? (
          fileType === 'video' ? (
            <video
              ref={videoRef}
              src={preview}
              onEnded={handleVideoEnded}
              className="w-full h-full object-contain"
              muted
              playsInline
              loop={false}
            />
          ) : (
            <img
              src={preview}
              alt="Preview"
              className="w-full h-full object-contain"
            />
          )
        ) : (
          // Upload prompt
          <div
            onClick={() => fileInputRef.current?.click()}
            className="text-center text-gray-500 cursor-pointer hover:text-gray-300 transition-colors p-4"
          >
            <Upload className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p className="text-sm font-medium">Click to upload</p>
            <p className="text-xs mt-1">Video or Image</p>
            <div className="flex justify-center space-x-3 mt-3">
              <span className="flex items-center text-xs">
                <Film className="h-3 w-3 mr-1" /> MP4
              </span>
              <span className="flex items-center text-xs">
                <Image className="h-3 w-3 mr-1" /> JPG/PNG
              </span>
            </div>
          </div>
        )}

        {/* Detection overlay - shows briefly when dog detected */}
        {lastDetection && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-12 right-2 bg-black/70 rounded-lg p-2 animate-pulse">
              <div className="flex items-center space-x-2">
                <Dog className="h-5 w-5 text-white" />
                <div>
                  <p className="text-xs text-white font-medium">
                    {detectionCount} dog{detectionCount !== 1 ? 's' : ''} detected
                  </p>
                  <StrayIndexBadge
                    strayIndex={lastDetection.stray_index}
                    status={lastDetection.status}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Alert indicator */}
        {hasStrayAlert && (
          <div className="absolute top-12 left-2 bg-red-500/90 rounded-lg p-2 animate-bounce">
            <div className="flex items-center space-x-1">
              <AlertTriangle className="h-4 w-4 text-white" />
              <span className="text-xs text-white font-bold">STRAY ALERT</span>
            </div>
          </div>
        )}
        {hasPossiblyLostAlert && (
          <div className="absolute top-12 left-2 bg-yellow-500/90 rounded-lg p-2">
            <div className="flex items-center space-x-1">
              <AlertTriangle className="h-4 w-4 text-white" />
              <span className="text-xs text-white font-bold">POSSIBLY LOST</span>
            </div>
          </div>
        )}
      </div>

      {/* Camera Footer */}
      <div className="camera-feed-overlay">
        {file ? (
          // Controls when file is loaded
          <div className="space-y-2">
            {/* File info and remove button */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-gray-300 text-sm truncate">
                {fileType === 'video' ? <Film className="h-4 w-4 flex-shrink-0" /> : <Image className="h-4 w-4 flex-shrink-0" />}
                <span className="truncate">{file.name}</span>
              </div>
              <button onClick={resetFile} className="text-gray-400 hover:text-red-400 p-1">
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Play/Stop button */}
            <div className="flex space-x-2">
              {!isAnalyzing ? (
                <button
                  onClick={startCapture}
                  disabled={!socket}
                  className="flex-1 flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white py-2 rounded text-sm transition-colors"
                >
                  <Play className="h-4 w-4" />
                  <span>Start</span>
                </button>
              ) : (
                <button
                  onClick={stopCapture}
                  className="flex-1 flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 text-white py-2 rounded text-sm transition-colors"
                >
                  <Pause className="h-4 w-4" />
                  <span>Stop</span>
                </button>
              )}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-gray-600 hover:bg-gray-500 text-white py-2 px-3 rounded text-sm transition-colors"
              >
                <Upload className="h-4 w-4" />
              </button>
            </div>
          </div>
        ) : (
          // Default state - prompt to upload
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">No feed loaded</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default CameraFeed
