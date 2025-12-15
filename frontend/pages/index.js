import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'
import StatsCards from '../components/StatsCards'
import AlertsPanel from '../components/AlertsPanel'
import TrafficChart from '../components/TrafficChart'
import AttackDistribution from '../components/AttackDistribution'
import TrafficMonitor from '../components/TrafficMonitor'

const API_BASE = 'http://localhost:8000'

export default function Dashboard() {
    const [stats, setStats] = useState(null)
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [isSimulating, setIsSimulating] = useState(false)
    const [simulatedTraffic, setSimulatedTraffic] = useState(null)
    const [showTrafficMonitor, setShowTrafficMonitor] = useState(false)

    // Load saved simulation data from localStorage on mount
    useEffect(() => {
        const savedTraffic = localStorage.getItem('simulatedTraffic')
        if (savedTraffic) {
            try {
                const data = JSON.parse(savedTraffic)
                setSimulatedTraffic(data)
                setShowTrafficMonitor(true)
            } catch (e) {
                console.error('Failed to parse saved traffic data')
            }
        }
    }, [])

    // Fetch stats from API
    const fetchStats = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/stats`)
            if (response.ok) {
                const data = await response.json()
                setStats(data)
                setError(null)
            }
        } catch (err) {
            console.error('Failed to fetch stats:', err)
            setError('Failed to connect to API')
        }
    }

    // Fetch alerts from API
    const fetchAlerts = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/alerts?limit=10`)
            if (response.ok) {
                const data = await response.json()
                setAlerts(data.alerts || [])
            }
        } catch (err) {
            console.error('Failed to fetch alerts:', err)
        }
    }

    // Simulate traffic for demo
    const simulateTraffic = async () => {
        setIsSimulating(true)
        try {
            const response = await fetch(`${API_BASE}/api/simulate?num_samples=30&anomaly_ratio=0.35`, {
                method: 'POST'
            })

            if (response.ok) {
                const data = await response.json()
                setSimulatedTraffic(data)
                setShowTrafficMonitor(true)

                // Save to localStorage for persistence
                localStorage.setItem('simulatedTraffic', JSON.stringify(data))

                // Refresh stats and alerts to show updated data
                await fetchStats()
                await fetchAlerts()
            } else {
                const error = await response.text()
                console.error('Simulation failed:', error)
            }
        } catch (err) {
            console.error('Simulation failed:', err)
        } finally {
            setIsSimulating(false)
        }
    }

    // Clear simulation data
    const clearSimulation = () => {
        setSimulatedTraffic(null)
        setShowTrafficMonitor(false)
        localStorage.removeItem('simulatedTraffic')
    }

    // Initial load
    useEffect(() => {
        const loadData = async () => {
            setLoading(true)
            try {
                await Promise.all([fetchStats(), fetchAlerts()])
            } catch (err) {
                setError('Failed to connect to API. Make sure the backend is running.')
            } finally {
                setLoading(false)
            }
        }

        loadData()

        // Refresh every 30 seconds
        const interval = setInterval(() => {
            fetchStats()
            fetchAlerts()
        }, 30000)

        return () => clearInterval(interval)
    }, [])

    return (
        <>
            <Head>
                <title>Network Anomaly Detection | Dashboard</title>
                <meta name="description" content="Real-time network anomaly detection and monitoring dashboard" />
                <link rel="icon" href="/favicon.ico" />
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet" />
            </Head>

            <div className="dashboard">
                <Sidebar />

                <main className="main-content">
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                        <div>
                            <h1>Network Monitor</h1>
                            <p style={{ marginTop: '0.5rem' }}>Powered by <strong style={{ color: 'var(--accent-primary)' }}>Two-Stage XGBoost AI</strong> — Cloud-Trained on 67M+ Records</p>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            <div className="live-indicator">Live</div>
                            <button
                                className="btn btn-primary"
                                onClick={simulateTraffic}
                                disabled={isSimulating}
                            >
                                {isSimulating ? '⏳ Generating...' : '🌐 Simulate Traffic'}
                            </button>
                        </div>
                    </div>

                    {loading ? (
                        <div className="loading">
                            <div className="spinner"></div>
                        </div>
                    ) : error && !stats ? (
                        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
                            <h3>Connection Error</h3>
                            <p style={{ marginTop: '0.5rem' }}>{error}</p>
                            <button className="btn btn-primary" style={{ marginTop: '1.5rem' }} onClick={() => window.location.reload()}>
                                Retry
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Stats Cards */}
                            <StatsCards stats={stats} />

                            {/* Traffic Monitor (shows after simulation or if saved data exists) */}
                            {showTrafficMonitor && simulatedTraffic && (
                                <TrafficMonitor
                                    data={simulatedTraffic}
                                    onClose={clearSimulation}
                                />
                            )}

                            {/* Charts Row */}
                            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">Traffic Analysis</span>
                                    </div>
                                    <TrafficChart />
                                </div>

                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">Attack Distribution</span>
                                    </div>
                                    <AttackDistribution stats={stats} />
                                </div>
                            </div>

                            {/* Alerts Panel */}
                            <AlertsPanel alerts={alerts} onRefresh={fetchAlerts} />
                        </>
                    )}
                </main>
            </div>
        </>
    )
}
