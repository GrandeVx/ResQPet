import { useState, useRef, useCallback } from 'react'
import { Upload, Film, Image, X, Play, Pause, RotateCcw } from 'lucide-react'

function VideoUploader({ cameraId, onFrameCapture, socket }) {
  const [file, setFile] = useState(null)
  const [fileType, setFileType] = useState(null) // 'video' or 'image'
  const [isPlaying, setIsPlaying] = useState(false)
  const [preview, setPreview] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const intervalRef = useRef(null)
  const fileInputRef = useRef(null)

  // Handle file selection
  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return

    // Cleanup previous
    stopCapture()

    const isVideo = selectedFile.type.startsWith('video/')
    const isImage = selectedFile.type.startsWith('image/')

    if (!isVideo && !isImage) {
      alert('Please select a video or image file')
      return
    }

    setFile(selectedFile)
    setFileType(isVideo ? 'video' : 'image')

    // Create preview URL
    const url = URL.createObjectURL(selectedFile)
    setPreview(url)
  }

  // Capture frame from video and send to backend
  const captureAndSendFrame = useCallback(() => {
    if (!canvasRef.current || !socket) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    if (fileType === 'video' && videoRef.current) {
      const video = videoRef.current
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    } else if (fileType === 'image' && preview) {
      const img = new window.Image()
      img.src = preview
      canvas.width = img.naturalWidth || 640
      canvas.height = img.naturalHeight || 480
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }

    // Convert to base64 and send via WebSocket
    const frameData = canvas.toDataURL('image/jpeg', 0.8)
    const base64Data = frameData.split(',')[1]

    socket.emit('analyze_frame', {
      camera_id: cameraId,
      frame: base64Data,
      timestamp: new Date().toISOString()
    })

    if (onFrameCapture) {
      onFrameCapture(base64Data)
    }
  }, [fileType, preview, socket, cameraId, onFrameCapture])

  // Start capturing frames
  const startCapture = () => {
    if (!file || !socket) return

    setIsPlaying(true)

    if (fileType === 'video' && videoRef.current) {
      videoRef.current.play()
      // Capture frame every 500ms (2 FPS)
      intervalRef.current = setInterval(captureAndSendFrame, 500)
    } else if (fileType === 'image') {
      // For images, send once per second in loop
      captureAndSendFrame()
      intervalRef.current = setInterval(captureAndSendFrame, 1000)
    }
  }

  // Stop capturing
  const stopCapture = () => {
    setIsPlaying(false)

    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.pause()
    }
  }

  // Reset/remove file
  const resetFile = () => {
    stopCapture()

    if (preview) {
      URL.revokeObjectURL(preview)
    }

    setFile(null)
    setFileType(null)
    setPreview(null)

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Handle video loop
  const handleVideoEnded = () => {
    if (videoRef.current && isPlaying) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {!file ? (
        // Upload Area
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-gray-750 transition-all"
        >
          <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-300 font-medium">Click to upload video or image</p>
          <p className="text-gray-500 text-sm mt-2">
            Supports MP4, AVI, MOV, JPG, PNG
          </p>
          <div className="flex justify-center space-x-4 mt-4">
            <span className="flex items-center text-gray-400 text-sm">
              <Film className="h-4 w-4 mr-1" /> Video
            </span>
            <span className="flex items-center text-gray-400 text-sm">
              <Image className="h-4 w-4 mr-1" /> Image
            </span>
          </div>
        </div>
      ) : (
        // Preview and Controls
        <div className="space-y-4">
          {/* Preview */}
          <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
            {fileType === 'video' ? (
              <video
                ref={videoRef}
                src={preview}
                onEnded={handleVideoEnded}
                className="w-full h-full object-contain"
                muted
                playsInline
              />
            ) : (
              <img
                src={preview}
                alt="Preview"
                className="w-full h-full object-contain"
              />
            )}

            {/* Playing indicator */}
            {isPlaying && (
              <div className="absolute top-2 right-2 flex items-center space-x-1 bg-red-500 px-2 py-1 rounded">
                <span className="h-2 w-2 bg-white rounded-full animate-pulse"></span>
                <span className="text-xs text-white font-medium">ANALYZING</span>
              </div>
            )}
          </div>

          {/* File Info */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2 text-gray-400">
              {fileType === 'video' ? (
                <Film className="h-4 w-4" />
              ) : (
                <Image className="h-4 w-4" />
              )}
              <span className="truncate max-w-48">{file.name}</span>
            </div>
            <button
              onClick={resetFile}
              className="text-gray-400 hover:text-red-400 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Controls */}
          <div className="flex space-x-2">
            {!isPlaying ? (
              <button
                onClick={startCapture}
                disabled={!socket}
                className="flex-1 flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-2 px-4 rounded-lg transition-colors"
              >
                <Play className="h-5 w-5" />
                <span>Start Analysis</span>
              </button>
            ) : (
              <button
                onClick={stopCapture}
                className="flex-1 flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition-colors"
              >
                <Pause className="h-5 w-5" />
                <span>Stop Analysis</span>
              </button>
            )}

            <button
              onClick={resetFile}
              className="flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded-lg transition-colors"
            >
              <RotateCcw className="h-5 w-5" />
            </button>
          </div>

          {/* Connection Status */}
          {!socket && (
            <p className="text-yellow-500 text-sm text-center">
              Waiting for backend connection...
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default VideoUploader
