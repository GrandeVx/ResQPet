import { clsx } from 'clsx'

function StrayIndexBadge({ strayIndex, status, showValue = true }) {
  const getStatusConfig = () => {
    if (strayIndex < 0.3 || status === 'owned') {
      return {
        label: 'Padronale',
        className: 'stray-index-owned',
        emoji: ''
      }
    } else if (strayIndex < 0.7 || status === 'possibly_lost') {
      return {
        label: 'Possibile Smarrito',
        className: 'stray-index-lost',
        emoji: ''
      }
    } else {
      return {
        label: 'Probabile Randagio',
        className: 'stray-index-stray',
        emoji: ''
      }
    }
  }

  const config = getStatusConfig()

  return (
    <span className={clsx('stray-index-badge', config.className)}>
      {config.emoji && <span className="mr-1">{config.emoji}</span>}
      {showValue && strayIndex !== undefined && (
        <span className="font-mono mr-1">{strayIndex.toFixed(2)}</span>
      )}
      <span>{config.label}</span>
    </span>
  )
}

export default StrayIndexBadge
