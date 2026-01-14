import { Dog, Wifi, WifiOff, Settings, BarChart3 } from 'lucide-react'

function Header({ isConnected }) {
  return (
    <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Dog className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">ResQPet</h1>
              <p className="text-xs text-gray-400">CCTV Monitoring System</p>
            </div>
          </div>

          {/* Status & Actions */}
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-sm ${
              isConnected
                ? 'bg-green-500/20 text-green-400'
                : 'bg-red-500/20 text-red-400'
            }`}>
              {isConnected ? (
                <>
                  <Wifi className="h-4 w-4" />
                  <span>Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-4 w-4" />
                  <span>Disconnected</span>
                </>
              )}
            </div>

            {/* Action Buttons */}
            <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
              <BarChart3 className="h-5 w-5" />
            </button>
            <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
