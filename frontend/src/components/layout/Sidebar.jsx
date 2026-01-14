import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Video,
  Map,
  Bell,
  Settings,
  Dog,
  Activity,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import clsx from 'clsx'
import { useState } from 'react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Live Cameras', href: '/cameras', icon: Video },
  { name: 'Map View', href: '/map', icon: Map },
  { name: 'Alerts', href: '/alerts', icon: Bell },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar({ isConnected, collapsed, onToggle }) {
  return (
    <aside
      className={clsx(
        'fixed inset-y-0 left-0 z-50 bg-surface-900 border-r border-surface-800 transition-all duration-300 flex flex-col',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-surface-800">
        <div className="p-2 bg-gradient-to-br from-brand-500 to-brand-600 rounded-xl shadow-glow-brand flex-shrink-0">
          <Dog className="h-6 w-6 text-white" />
        </div>
        {!collapsed && (
          <div className="animate-fade-in">
            <h1 className="text-lg font-bold text-white tracking-tight">ResQPet</h1>
            <p className="text-xs text-surface-400">CCTV Monitoring</p>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className="px-3 py-4">
        <div className={clsx(
          'flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm transition-colors',
          isConnected
            ? 'bg-success/10 text-success border border-success/20'
            : 'bg-danger/10 text-danger border border-danger/20'
        )}>
          <Activity className="h-4 w-4 flex-shrink-0" />
          {!collapsed && (
            <span className="animate-fade-in font-medium">
              {isConnected ? 'System Online' : 'Disconnected'}
            </span>
          )}
          <span className={clsx(
            'h-2 w-2 rounded-full flex-shrink-0',
            collapsed ? '' : 'ml-auto',
            isConnected ? 'bg-success animate-pulse' : 'bg-danger'
          )} />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-2 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200',
              isActive
                ? 'bg-brand-600 text-white shadow-glow-brand'
                : 'text-surface-400 hover:text-white hover:bg-surface-800'
            )}
            title={collapsed ? item.name : undefined}
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!collapsed && (
              <span className="animate-fade-in">{item.name}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="px-3 py-4 border-t border-surface-800">
        <button
          onClick={onToggle}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-surface-400 hover:text-white hover:bg-surface-800 transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <>
              <ChevronLeft className="h-5 w-5" />
              <span className="text-sm">Collapse</span>
            </>
          )}
        </button>
      </div>

      {/* Footer */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-surface-800 animate-fade-in">
          <p className="text-xs text-surface-500 text-center">
            ResQPet v2.0
          </p>
          <p className="text-xs text-surface-600 text-center mt-0.5">
            Demo Mode
          </p>
        </div>
      )}
    </aside>
  )
}

export default Sidebar
