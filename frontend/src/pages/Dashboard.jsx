import { useState, useEffect } from 'react'
import { Dog, AlertTriangle, Camera, Activity, TrendingUp, Clock, Zap } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from 'recharts'
import { KPICard } from '../components/dashboard/KPICard'
import { formatDistanceToNow } from 'date-fns'
import { it } from 'date-fns/locale'
import clsx from 'clsx'

export default function Dashboard({ socket }) {
  const [stats, setStats] = useState(null)
  const [recentAlerts, setRecentAlerts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
    fetchAlerts()
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/demo/stats')
      const data = await res.json()
      setStats(data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchAlerts = async () => {
    try {
      const res = await fetch('/api/alerts')
      const data = await res.json()
      setRecentAlerts((data.alerts || []).slice(0, 5))
    } catch (err) {
      console.error('Failed to fetch alerts:', err)
    }
  }

  if (loading || !stats) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-surface-800 rounded-lg w-48" />
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-surface-800 rounded-2xl" />
          ))}
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2 h-80 bg-surface-800 rounded-2xl" />
          <div className="h-80 bg-surface-800 rounded-2xl" />
        </div>
      </div>
    )
  }

  const chartData = stats.history.labels.map((label, i) => ({
    name: label,
    detections: stats.history.detections[i],
    alerts: stats.history.alerts[i],
  }))

  const pieData = [
    { name: 'Padronali', value: stats.categories.owned, color: '#10b981' },
    { name: 'Smarriti', value: stats.categories.possibly_lost, color: '#f59e0b' },
    { name: 'Randagi', value: stats.categories.likely_stray, color: '#ef4444' },
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-surface-400 mt-1">Panoramica del sistema di monitoraggio</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-800 rounded-xl text-sm text-surface-400">
          <Clock className="h-4 w-4" />
          <span>Ultimo aggiornamento: {stats.last_detection}</span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Rilevamenti Oggi"
          value={stats.detections_today}
          subtitle="cani identificati"
          change="+12%"
          trend="up"
          icon={Dog}
          color="brand"
        />
        <KPICard
          title="Alert Attivi"
          value={stats.active_alerts}
          subtitle="richiedono attenzione"
          icon={AlertTriangle}
          color="danger"
        />
        <KPICard
          title="Telecamere"
          value={`${stats.cameras_online}/${stats.cameras_total}`}
          subtitle={`uptime ${stats.system_uptime}`}
          icon={Camera}
          color="success"
        />
        <KPICard
          title="Stray Index Medio"
          value={stats.avg_stray_index.toFixed(2)}
          subtitle="indice di randagismo"
          icon={Activity}
          color="warning"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart */}
        <div className="lg:col-span-2 bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="font-semibold text-white text-lg">Attivita Settimanale</h3>
              <p className="text-sm text-surface-400 mt-1">Rilevamenti e alert degli ultimi 7 giorni</p>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-brand-500" />
                <span className="text-surface-400">Rilevamenti</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-danger" />
                <span className="text-surface-400">Alert</span>
              </div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorDetections" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorAlerts" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '12px',
                    boxShadow: '0 10px 40px rgba(0,0,0,0.3)',
                  }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Area
                  type="monotone"
                  dataKey="detections"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  fill="url(#colorDetections)"
                  name="Rilevamenti"
                />
                <Area
                  type="monotone"
                  dataKey="alerts"
                  stroke="#ef4444"
                  strokeWidth={2}
                  fill="url(#colorAlerts)"
                  name="Alert"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie Chart - Categories */}
        <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
          <h3 className="font-semibold text-white text-lg mb-2">Distribuzione</h3>
          <p className="text-sm text-surface-400 mb-4">Classificazione cani rilevati</p>

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-2 mt-4">
            {pieData.map((item) => (
              <div key={item.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-surface-300">{item.name}</span>
                </div>
                <span className="text-sm font-mono text-white">{item.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Alerts */}
        <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-white">Alert Recenti</h3>
            <Zap className="h-5 w-5 text-warning" />
          </div>
          <div className="space-y-3">
            {recentAlerts.length === 0 ? (
              <div className="text-center py-8 text-surface-500">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Nessun alert recente</p>
              </div>
            ) : (
              recentAlerts.map((alert, idx) => (
                <div
                  key={alert.id || idx}
                  className="flex items-center gap-3 p-3 bg-surface-800/50 rounded-xl hover:bg-surface-800 transition-colors"
                >
                  <div className={clsx(
                    'h-2 w-2 rounded-full flex-shrink-0',
                    (alert.stray_index || 0) >= 0.7 ? 'bg-danger animate-pulse' : 'bg-warning'
                  )} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white truncate">
                      {alert.camera_id || 'Camera'}
                    </p>
                    <p className="text-xs text-surface-400">
                      SI: {(alert.stray_index || 0).toFixed(2)} - {alert.timestamp
                        ? formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true, locale: it })
                        : 'ora'}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-white">Stato Sistema</h3>
            <Activity className="h-5 w-5 text-success" />
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-surface-400">Backend API</span>
              <span className="flex items-center gap-2 text-success text-sm">
                <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
                Operativo
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-surface-400">WebSocket</span>
              <span className="flex items-center gap-2 text-success text-sm">
                <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
                Connesso
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-surface-400">ML Pipeline</span>
              <span className="flex items-center gap-2 text-success text-sm">
                <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
                Attivo
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-surface-400">Telecamere Online</span>
              <span className="text-white font-mono">{stats.cameras_online}/{stats.cameras_total}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-surface-400">Uptime</span>
              <span className="text-white font-mono">{stats.system_uptime}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-surface-400">Totale Rilevamenti</span>
              <span className="text-white font-mono">{stats.total_detections.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
