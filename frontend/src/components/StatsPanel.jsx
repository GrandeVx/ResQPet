import { Dog, AlertTriangle, Activity, TrendingUp } from 'lucide-react'

function StatsPanel({ stats }) {
  const statItems = [
    {
      label: 'Dogs Detected',
      value: stats.totalDetections,
      icon: Dog,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10'
    },
    {
      label: 'Active Alerts',
      value: stats.activeAlerts,
      icon: AlertTriangle,
      color: 'text-red-400',
      bgColor: 'bg-red-500/10'
    },
    {
      label: 'Avg Stray Index',
      value: typeof stats.avgStrayIndex === 'number'
        ? stats.avgStrayIndex.toFixed(2)
        : stats.avgStrayIndex,
      icon: Activity,
      color: getIndexColor(stats.avgStrayIndex),
      bgColor: getIndexBgColor(stats.avgStrayIndex)
    }
  ]

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <TrendingUp className="h-5 w-5 text-blue-400" />
          <h3 className="font-semibold text-white">Live Statistics</h3>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="p-4 space-y-4">
        {statItems.map((stat, idx) => (
          <StatCard key={idx} {...stat} />
        ))}

        {/* Stray Index Legend */}
        <div className="pt-4 border-t border-gray-700">
          <p className="text-xs text-gray-500 mb-2">Stray Index Legend</p>
          <div className="space-y-1">
            <LegendItem color="bg-green-500" label="< 0.3" description="Padronale" />
            <LegendItem color="bg-yellow-500" label="0.3-0.7" description="Possibile Smarrito" />
            <LegendItem color="bg-red-500" label="> 0.7" description="Probabile Randagio" />
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, icon: Icon, color, bgColor }) {
  return (
    <div className="stat-card flex items-center space-x-3">
      <div className={`p-2 rounded-lg ${bgColor}`}>
        <Icon className={`h-5 w-5 ${color}`} />
      </div>
      <div>
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-xs text-gray-400">{label}</p>
      </div>
    </div>
  )
}

function LegendItem({ color, label, description }) {
  return (
    <div className="flex items-center space-x-2 text-xs">
      <span className={`w-3 h-3 rounded ${color}`}></span>
      <span className="text-gray-400">{label}:</span>
      <span className="text-gray-300">{description}</span>
    </div>
  )
}

function getIndexColor(index) {
  const val = parseFloat(index) || 0
  if (val < 0.3) return 'text-green-400'
  if (val < 0.7) return 'text-yellow-400'
  return 'text-red-400'
}

function getIndexBgColor(index) {
  const val = parseFloat(index) || 0
  if (val < 0.3) return 'bg-green-500/10'
  if (val < 0.7) return 'bg-yellow-500/10'
  return 'bg-red-500/10'
}

export default StatsPanel
