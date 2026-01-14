import { useState } from 'react'
import { AlertTriangle, Clock, MapPin, Check, X, ZoomIn, Dog } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { it } from 'date-fns/locale'
import StrayIndexBadge from './StrayIndexBadge'

function AlertPanel({ alerts, onAcknowledge }) {
  const [selectedAlert, setSelectedAlert] = useState(null)
  const unacknowledged = alerts.filter(a => !a.acknowledged)

  return (
    <>
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              <h3 className="font-semibold text-white">Alert Log</h3>
            </div>
            {unacknowledged.length > 0 && (
              <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full animate-pulse">
                {unacknowledged.length} new
              </span>
            )}
          </div>
        </div>

        {/* Alert List */}
        <div className="max-h-96 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No alerts yet</p>
              <p className="text-xs mt-1">Alerts will appear when stray dogs are detected</p>
            </div>
          ) : (
            <div className="p-2 space-y-2">
              {alerts.map((alert, idx) => (
                <AlertCard
                  key={alert.id || idx}
                  alert={alert}
                  onView={() => setSelectedAlert(alert)}
                  onAcknowledge={onAcknowledge}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modal for viewing snapshot */}
      {selectedAlert && (
        <AlertModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onAcknowledge={onAcknowledge}
        />
      )}
    </>
  )
}

function AlertCard({ alert, onView, onAcknowledge }) {
  const strayIndex = alert.stray_index || alert.data?.stray_index || 0
  const status = alert.status || alert.data?.status
  const isUrgent = strayIndex >= 0.7 && !alert.acknowledged
  const isPossiblyLost = strayIndex >= 0.3 && strayIndex < 0.7
  const hasSnapshot = alert.snapshot || alert.data?.snapshot

  return (
    <div className={`alert-card ${isUrgent ? 'alert-card-urgent' : isPossiblyLost ? 'alert-card-warning' : ''}`}>
      <div className="flex items-start gap-3">
        {/* Thumbnail */}
        {hasSnapshot && (
          <button
            onClick={onView}
            className="flex-shrink-0 w-16 h-16 rounded overflow-hidden bg-gray-700 hover:ring-2 hover:ring-blue-500 transition-all"
          >
            <img
              src={`data:image/jpeg;base64,${alert.snapshot || alert.data?.snapshot}`}
              alt="Detection snapshot"
              className="w-full h-full object-cover"
            />
          </button>
        )}

        <div className="flex-1 min-w-0">
          {/* Camera & Time */}
          <div className="flex items-center space-x-2 text-xs text-gray-400 mb-1">
            <MapPin className="h-3 w-3" />
            <span>{alert.camera_id || alert.data?.camera_id}</span>
            <span>-</span>
            <Clock className="h-3 w-3" />
            <span>
              {alert.timestamp || alert.data?.timestamp
                ? formatDistanceToNow(new Date(alert.timestamp || alert.data?.timestamp), {
                    addSuffix: true,
                    locale: it
                  })
                : 'Just now'}
            </span>
          </div>

          {/* Status Badge */}
          <div className="mb-1">
            <StrayIndexBadge
              strayIndex={alert.stray_index || alert.data?.stray_index}
              status={alert.status || alert.data?.status}
            />
          </div>

          {/* Breed Info */}
          {(alert.breed || alert.data?.breed) && (
            <p className="text-xs text-gray-400">
              <Dog className="h-3 w-3 inline mr-1" />
              {alert.breed || alert.data?.breed}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-1">
          {hasSnapshot && (
            <button
              onClick={onView}
              className="p-1.5 text-gray-400 hover:text-blue-400 hover:bg-blue-500/10 rounded transition-colors"
              title="View snapshot"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
          )}
          {!alert.acknowledged && onAcknowledge && (
            <button
              onClick={() => onAcknowledge(alert.id)}
              className="p-1.5 text-gray-400 hover:text-green-400 hover:bg-green-500/10 rounded transition-colors"
              title="Acknowledge alert"
            >
              <Check className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function AlertModal({ alert, onClose, onAcknowledge }) {
  const snapshot = alert.snapshot || alert.data?.snapshot
  const strayIndex = alert.stray_index || alert.data?.stray_index
  const status = alert.status || alert.data?.status
  const breed = alert.breed || alert.data?.breed
  const cameraId = alert.camera_id || alert.data?.camera_id
  const timestamp = alert.timestamp || alert.data?.timestamp
  const components = alert.components || alert.data?.components || {}

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <h3 className="font-semibold text-white">Stray Alert Details</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {/* Snapshot */}
          {snapshot && (
            <div className="mb-4">
              <img
                src={`data:image/jpeg;base64,${snapshot}`}
                alt="Detection snapshot"
                className="w-full rounded-lg"
              />
            </div>
          )}

          {/* Details Grid */}
          <div className="grid grid-cols-2 gap-4">
            {/* Left Column */}
            <div className="space-y-3">
              <div>
                <p className="text-xs text-gray-400 mb-1">Camera</p>
                <p className="text-white font-medium">{cameraId}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">Time</p>
                <p className="text-white">
                  {timestamp
                    ? new Date(timestamp).toLocaleString('it-IT')
                    : 'Unknown'}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">Breed</p>
                <p className="text-white">{breed || 'Unknown'}</p>
              </div>
            </div>

            {/* Right Column */}
            <div className="space-y-3">
              <div>
                <p className="text-xs text-gray-400 mb-1">Stray Index</p>
                <StrayIndexBadge strayIndex={strayIndex} status={status} />
              </div>

              {/* Component Breakdown */}
              {Object.keys(components).length > 0 && (
                <div>
                  <p className="text-xs text-gray-400 mb-2">Analysis Breakdown</p>
                  <div className="space-y-1">
                    {Object.entries(components).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-xs text-gray-400 capitalize">{key}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                value >= 0.7 ? 'bg-red-500' :
                                value >= 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${value * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-white w-8">
                            {(value * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-gray-700">
          {!alert.acknowledged && onAcknowledge && (
            <button
              onClick={() => {
                onAcknowledge(alert.id)
                onClose()
              }}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
            >
              <Check className="h-4 w-4" />
              <span>Acknowledge</span>
            </button>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

export default AlertPanel
