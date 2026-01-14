import { TrendingUp, TrendingDown } from 'lucide-react'
import clsx from 'clsx'

export function KPICard({
  title,
  value,
  subtitle,
  change,
  trend,
  icon: Icon,
  color = 'brand'
}) {
  const colorClasses = {
    brand: {
      bg: 'bg-brand-500/10',
      text: 'text-brand-400',
      border: 'border-brand-500/20',
      glow: 'shadow-glow-brand',
    },
    danger: {
      bg: 'bg-danger/10',
      text: 'text-danger',
      border: 'border-danger/20',
      glow: 'shadow-glow-danger',
    },
    warning: {
      bg: 'bg-warning/10',
      text: 'text-warning',
      border: 'border-warning/20',
      glow: '',
    },
    success: {
      bg: 'bg-success/10',
      text: 'text-success',
      border: 'border-success/20',
      glow: 'shadow-glow-success',
    },
  }

  const colors = colorClasses[color] || colorClasses.brand

  return (
    <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-5 hover:border-surface-700 transition-all duration-300 group">
      <div className="flex items-start justify-between">
        <div className={clsx(
          'p-3 rounded-xl transition-all duration-300',
          colors.bg,
          colors.border,
          'border',
          'group-hover:scale-110'
        )}>
          <Icon className={clsx('h-5 w-5', colors.text)} />
        </div>

        {change && (
          <div className={clsx(
            'flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium',
            trend === 'up' ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'
          )}>
            {trend === 'up' ? (
              <TrendingUp className="h-3 w-3" />
            ) : (
              <TrendingDown className="h-3 w-3" />
            )}
            <span>{change}</span>
          </div>
        )}
      </div>

      <div className="mt-4">
        <p className="text-3xl font-bold text-white font-mono tracking-tight">
          {value}
        </p>
        <p className="text-sm text-surface-400 mt-1 font-medium">
          {title}
        </p>
        {subtitle && (
          <p className="text-xs text-surface-500 mt-0.5">
            {subtitle}
          </p>
        )}
      </div>
    </div>
  )
}

export default KPICard
