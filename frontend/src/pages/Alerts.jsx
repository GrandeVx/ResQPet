import { useState } from 'react'
import { Bell, Filter, Clock, MapPin, Search, X, ZoomIn, Dog, Activity } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { it } from 'date-fns/locale'
import StrayIndexBadge from '../components/StrayIndexBadge'
import clsx from 'clsx'

export default function Alerts({ alerts = [] }) {
  const [filter, setFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAlert, setSelectedAlert] = useState(null)

  // Filter alerts
  const filteredAlerts = alerts.filter(alert => {
    const si = alert.stray_index || alert.data?.stray_index || 0

    // Filter by type
    if (filter === 'stray' && si < 0.7) return false
    if (filter === 'lost' && (si < 0.3 || si >= 0.7)) return false
    if (filter === 'owned' && si >= 0.3) return false

    // Filter by search
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      const cameraId = (alert.camera_id || alert.data?.camera_id || '').toLowerCase()
      const breed = (alert.breed || alert.data?.breed || '').toLowerCase()
      if (!cameraId.includes(query) && !breed.includes(query)) return false
    }

    return true
  })

  const getAlertStats = () => {
    const total = alerts.length
    const stray = alerts.filter(a => (a.stray_index || 0) >= 0.7).length
    const lost = alerts.filter(a => (a.stray_index || 0) >= 0.3 && (a.stray_index || 0) < 0.7).length
    return { total, stray, lost }
  }

  const stats = getAlertStats()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Cronologia Alert</h1>
          <p className="text-surface-400 mt-1">
            {stats.total} alert totali - {stats.stray} randagi, {stats.lost} smarriti
          </p>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-500" />
          <input
            type="text"
            placeholder="Cerca per camera o razza..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-surface-900 border border-surface-800 rounded-xl text-white placeholder-surface-500 focus:outline-none focus:border-brand-500 transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-500 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Filter Buttons */}
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-surface-400" />
          {[
            { key: 'all', label: 'Tutti', count: stats.total },
            { key: 'stray', label: 'Randagi', count: stats.stray },
            { key: 'lost', label: 'Smarriti', count: stats.lost },
          ].map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={clsx(
                'px-3 py-2 rounded-xl text-sm font-medium transition-colors flex items-center gap-2',
                filter === f.key
                  ? 'bg-brand-600 text-white'
                  : 'bg-surface-800 text-surface-400 hover:text-white hover:bg-surface-700'
              )}
            >
              {f.label}
              <span className={clsx(
                'px-1.5 py-0.5 rounded text-xs',
                filter === f.key ? 'bg-white/20' : 'bg-surface-700'
              )}>
                {f.count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Alerts Timeline */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-16 text-surface-500">
            <Bell className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="text-lg">Nessun alert trovato</p>
            <p className="text-sm mt-1">
              {searchQuery ? 'Prova a modificare la ricerca' : 'Gli alert appariranno quando verranno rilevati cani randagi'}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert, idx) => (
            <AlertCard
              key={alert.id || idx}
              alert={alert}
              onView={() => setSelectedAlert(alert)}
            />
          ))
        )}
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <AlertModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
        />
      )}
    </div>
  )
}

function AlertCard({ alert, onView }) {
  const strayIndex = alert.stray_index || alert.data?.stray_index || 0
  const isStray = strayIndex >= 0.7
  const isLost = strayIndex >= 0.3 && strayIndex < 0.7
  const snapshot = alert.snapshot || alert.data?.snapshot
  const cameraId = alert.camera_id || alert.data?.camera_id
  const timestamp = alert.timestamp || alert.data?.timestamp
  const breed = alert.breed || alert.data?.breed

  return (
    <div
      className={clsx(
        'bg-surface-900/80 backdrop-blur-sm rounded-2xl border overflow-hidden transition-all hover:shadow-lg cursor-pointer',
        isStray ? 'border-danger/50 hover:border-danger' :
        isLost ? 'border-warning/50 hover:border-warning' :
        'border-surface-800 hover:border-surface-700'
      )}
      onClick={onView}
    >
      <div className="flex items-start gap-4 p-4">
        {/* Thumbnail */}
        {snapshot ? (
          <div className="relative flex-shrink-0">
            <img
              src={`data:image/jpeg;base64,${snapshot}`}
              alt="Detection"
              className="w-24 h-24 object-cover rounded-xl"
            />
            <button className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 hover:opacity-100 transition-opacity rounded-xl">
              <ZoomIn className="h-6 w-6 text-white" />
            </button>
          </div>
        ) : (
          <div className="w-24 h-24 bg-surface-800 rounded-xl flex items-center justify-center flex-shrink-0">
            <Dog className="h-8 w-8 text-surface-600" />
          </div>
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <StrayIndexBadge
              strayIndex={strayIndex}
              status={alert.status || alert.data?.status}
            />
            {isStray && (
              <span className="px-2 py-0.5 bg-danger/20 text-danger text-xs rounded-lg font-medium animate-pulse">
                URGENTE
              </span>
            )}
          </div>

          <div className="flex items-center gap-4 text-sm text-surface-400 mb-2">
            <span className="flex items-center gap-1.5">
              <MapPin className="h-3.5 w-3.5" />
              {cameraId || 'Camera'}
            </span>
            <span className="flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5" />
              {timestamp
                ? formatDistanceToNow(new Date(timestamp), { addSuffix: true, locale: it })
                : 'Adesso'}
            </span>
          </div>

          {breed && (
            <p className="text-sm text-surface-300">
              <Dog className="h-3.5 w-3.5 inline mr-1.5" />
              Razza: {breed}
            </p>
          )}
        </div>

        {/* Stray Index */}
        <div className="text-right flex-shrink-0">
          <p className="text-xs text-surface-400 mb-1">Stray Index</p>
          <p className={clsx(
            'text-2xl font-bold font-mono',
            isStray ? 'text-danger' : isLost ? 'text-warning' : 'text-success'
          )}>
            {strayIndex.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  )
}

function AlertModal({ alert, onClose }) {
  const snapshot = alert.snapshot || alert.data?.snapshot
  const strayIndex = alert.stray_index || alert.data?.stray_index || 0
  const components = alert.components || alert.data?.components || {}
  const cameraId = alert.camera_id || alert.data?.camera_id
  const timestamp = alert.timestamp || alert.data?.timestamp
  const breed = alert.breed || alert.data?.breed

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-surface-900 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden border border-surface-800 animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-800">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-danger" />
            <h3 className="font-semibold text-white">Dettaglio Alert</h3>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-surface-400 hover:text-white hover:bg-surface-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(90vh-120px)]">
          {snapshot && (
            <img
              src={`data:image/jpeg;base64,${snapshot}`}
              alt="Detection"
              className="w-full rounded-xl mb-4"
            />
          )}

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-surface-800/50 rounded-xl p-3">
              <p className="text-xs text-surface-400 mb-1">Camera</p>
              <p className="text-white font-medium">{cameraId || 'N/A'}</p>
            </div>
            <div className="bg-surface-800/50 rounded-xl p-3">
              <p className="text-xs text-surface-400 mb-1">Timestamp</p>
              <p className="text-white font-medium">
                {timestamp ? new Date(timestamp).toLocaleString('it-IT') : 'N/A'}
              </p>
            </div>
            <div className="bg-surface-800/50 rounded-xl p-3">
              <p className="text-xs text-surface-400 mb-1">Razza</p>
              <p className="text-white font-medium">{breed || 'Non identificata'}</p>
            </div>
            <div className="bg-surface-800/50 rounded-xl p-3">
              <p className="text-xs text-surface-400 mb-1">Stray Index</p>
              <p className={clsx(
                'text-xl font-bold font-mono',
                strayIndex >= 0.7 ? 'text-danger' : strayIndex >= 0.3 ? 'text-warning' : 'text-success'
              )}>
                {strayIndex.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Component Breakdown */}
          {Object.keys(components).length > 0 && (
            <div className="bg-surface-800/50 rounded-xl p-4">
              <p className="text-sm text-surface-400 mb-3 font-medium">Analisi Componenti</p>
              <div className="space-y-3">
                {Object.entries(components).map(([key, value]) => (
                  <div key={key} className="flex items-center gap-3">
                    <span className="text-sm text-surface-300 capitalize w-20">{key}</span>
                    <div className="flex-1 h-2 bg-surface-700 rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          'h-full rounded-full transition-all',
                          value >= 0.7 ? 'bg-danger' :
                          value >= 0.4 ? 'bg-warning' : 'bg-success'
                        )}
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-mono text-white w-12 text-right">
                      {(value * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 p-4 border-t border-surface-800">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-surface-800 hover:bg-surface-700 text-white rounded-xl transition-colors"
          >
            Chiudi
          </button>
        </div>
      </div>
    </div>
  )
}
