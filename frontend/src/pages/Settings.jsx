import { useState } from 'react'
import {
  Settings as SettingsIcon,
  Bell,
  Camera,
  Shield,
  Database,
  Palette,
  Globe,
  Zap,
  Info,
  ChevronRight,
  Moon,
  Sun
} from 'lucide-react'
import clsx from 'clsx'

const settingSections = [
  {
    id: 'notifications',
    icon: Bell,
    title: 'Notifiche',
    description: 'Configura soglie alert e notifiche',
    color: 'brand',
  },
  {
    id: 'cameras',
    icon: Camera,
    title: 'Telecamere',
    description: 'Gestisci sorgenti video e stream',
    color: 'success',
  },
  {
    id: 'security',
    icon: Shield,
    title: 'Sicurezza',
    description: 'Autenticazione e controllo accessi',
    color: 'warning',
  },
  {
    id: 'data',
    icon: Database,
    title: 'Dati',
    description: 'Backup, esportazione e archiviazione',
    color: 'danger',
  },
]

export default function Settings() {
  const [activeSection, setActiveSection] = useState(null)
  const [darkMode, setDarkMode] = useState(true)

  const colorClasses = {
    brand: 'bg-brand-500/10 text-brand-400 group-hover:bg-brand-500/20',
    success: 'bg-success/10 text-success group-hover:bg-success/20',
    warning: 'bg-warning/10 text-warning group-hover:bg-warning/20',
    danger: 'bg-danger/10 text-danger group-hover:bg-danger/20',
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Impostazioni</h1>
        <p className="text-surface-400 mt-1">Configurazione del sistema ResQPet</p>
      </div>

      {/* Quick Settings */}
      <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
        <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="h-5 w-5 text-warning" />
          Impostazioni Rapide
        </h3>

        <div className="space-y-4">
          {/* Dark Mode Toggle */}
          <div className="flex items-center justify-between py-3 border-b border-surface-800">
            <div className="flex items-center gap-3">
              {darkMode ? <Moon className="h-5 w-5 text-brand-400" /> : <Sun className="h-5 w-5 text-warning" />}
              <div>
                <p className="text-white font-medium">Tema Scuro</p>
                <p className="text-sm text-surface-400">Interfaccia ottimizzata per ambienti poco illuminati</p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={clsx(
                'relative w-12 h-6 rounded-full transition-colors',
                darkMode ? 'bg-brand-600' : 'bg-surface-700'
              )}
            >
              <span
                className={clsx(
                  'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                  darkMode ? 'left-7' : 'left-1'
                )}
              />
            </button>
          </div>

          {/* Language */}
          <div className="flex items-center justify-between py-3 border-b border-surface-800">
            <div className="flex items-center gap-3">
              <Globe className="h-5 w-5 text-success" />
              <div>
                <p className="text-white font-medium">Lingua</p>
                <p className="text-sm text-surface-400">Seleziona la lingua dell'interfaccia</p>
              </div>
            </div>
            <select className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-brand-500">
              <option value="it">Italiano</option>
              <option value="en">English</option>
            </select>
          </div>

          {/* Alert Threshold */}
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-3">
              <Bell className="h-5 w-5 text-danger" />
              <div>
                <p className="text-white font-medium">Soglia Alert</p>
                <p className="text-sm text-surface-400">Stray Index minimo per generare alert</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="100"
                defaultValue="70"
                className="w-24 accent-brand-500"
              />
              <span className="text-white font-mono text-sm w-12">0.70</span>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {settingSections.map((section) => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className="group p-6 bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 text-left hover:border-surface-700 transition-all"
          >
            <div className="flex items-start justify-between">
              <div className={clsx(
                'p-3 rounded-xl transition-colors',
                colorClasses[section.color]
              )}>
                <section.icon className="h-6 w-6" />
              </div>
              <ChevronRight className="h-5 w-5 text-surface-600 group-hover:text-surface-400 transition-colors" />
            </div>
            <h3 className="font-semibold text-white mt-4">{section.title}</h3>
            <p className="text-sm text-surface-400 mt-1">{section.description}</p>
          </button>
        ))}
      </div>

      {/* System Info */}
      <div className="bg-surface-900/80 backdrop-blur-sm rounded-2xl border border-surface-800 p-6">
        <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Info className="h-5 w-5 text-brand-400" />
          Informazioni Sistema
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-surface-800/50 rounded-xl p-4">
            <p className="text-xs text-surface-400 mb-1">Versione</p>
            <p className="text-white font-mono">2.0.0-demo</p>
          </div>
          <div className="bg-surface-800/50 rounded-xl p-4">
            <p className="text-xs text-surface-400 mb-1">Modello ML</p>
            <p className="text-white font-mono text-sm">ResQPet Fusion</p>
          </div>
          <div className="bg-surface-800/50 rounded-xl p-4">
            <p className="text-xs text-surface-400 mb-1">Backend</p>
            <p className="text-white font-mono text-sm">Flask + SocketIO</p>
          </div>
          <div className="bg-surface-800/50 rounded-xl p-4">
            <p className="text-xs text-surface-400 mb-1">Frontend</p>
            <p className="text-white font-mono text-sm">React + Vite</p>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-surface-800">
          <p className="text-xs text-surface-500 text-center">
            ResQPet - Sistema di Identificazione Automatizzata dello Stato di Abbandono nei Cani
          </p>
          <p className="text-xs text-surface-600 text-center mt-1">
            Progetto di Fondamenti di Intelligenza Artificiale - Universita degli Studi di Salerno
          </p>
        </div>
      </div>
    </div>
  )
}
