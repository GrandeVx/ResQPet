import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { Toaster } from 'sonner'
import clsx from 'clsx'

export function Layout({ children, isConnected }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <div className="min-h-screen bg-surface-950">
      {/* Background pattern */}
      <div className="fixed inset-0 bg-grid-pattern bg-[size:50px_50px] pointer-events-none opacity-50" />

      {/* Subtle gradient overlay */}
      <div className="fixed inset-0 bg-gradient-to-br from-brand-950/20 via-transparent to-surface-950 pointer-events-none" />

      <Sidebar
        isConnected={isConnected}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main
        className={clsx(
          'relative transition-all duration-300',
          sidebarCollapsed ? 'pl-16' : 'pl-64'
        )}
      >
        <div className="p-6 min-h-screen">
          {children}
        </div>
      </main>

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        expand={false}
        richColors
        theme="dark"
        toastOptions={{
          style: {
            background: '#1e293b',
            border: '1px solid #334155',
            color: '#f1f5f9',
          },
          className: 'rounded-xl shadow-lg',
        }}
      />
    </div>
  )
}

export default Layout
