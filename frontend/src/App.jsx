import { Routes, Route } from 'react-router-dom'
import { useSocket } from './hooks/useSocket'
import { Layout } from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import LiveCameras from './pages/LiveCameras'
import MapViewPage from './pages/MapView'
import Alerts from './pages/Alerts'
import Settings from './pages/Settings'

function App() {
  const { socket, isConnected, detections, alerts } = useSocket()

  return (
    <Layout isConnected={isConnected}>
      <Routes>
        <Route path="/" element={<Dashboard socket={socket} />} />
        <Route path="/cameras" element={<LiveCameras socket={socket} detections={detections} />} />
        <Route path="/map" element={<MapViewPage socket={socket} />} />
        <Route path="/alerts" element={<Alerts alerts={alerts} />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App
