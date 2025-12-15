import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'

const API_BASE = 'http://localhost:8000'

export default function Alerts() {
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState('all') // all, critical, high, medium
    const [showAcknowledged, setShowAcknowledged] = useState(true)

    // Fetch alerts
    const fetchAlerts = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/alerts?limit=100`)
            if (response.ok) {
                const data = await response.json()
                setAlerts(data.alerts || [])
            }
        } catch (err) {
            console.error('Failed to fetch alerts:', err)
        } finally {
            setLoading(false)
        }
    }

    // Acknowledge alert
    const acknowledgeAlert = async (alertId) => {
        try {
            await fetch(`${API_BASE}/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            })
            // Update local state
            setAlerts(alerts.map(a =>
                a.id === alertId ? { ...a, is_acknowledged: true } : a
            ))
        } catch (err) {
            console.error('Failed to acknowledge alert:', err)
        }
    }

    // Acknowledge all
    const acknowledgeAll = async () => {
        for (const alert of alerts.filter(a => !a.is_acknowledged)) {
            await acknowledgeAlert(alert.id)
        }
    }

    useEffect(() => {
        fetchAlerts()
        const interval = setInterval(fetchAlerts, 10000)
        return () => clearInterval(interval)
    }, [])

    // Filter alerts
    const filteredAlerts = alerts.filter(alert => {
        if (!showAcknowledged && alert.is_acknowledged) return false
        if (filter !== 'all' && alert.severity !== filter) return false
        return true
    })

    const formatTime = (timestamp) => {
        return new Date(timestamp).toLocaleString()
    }

    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return 'var(--danger)'
            case 'high': return 'var(--warning)'
            case 'medium': return 'var(--info)'
            default: return 'var(--text-muted)'
        }
    }

    const unacknowledgedCount = alerts.filter(a => !a.is_acknowledged).length

    return (
        <>
            <Head>
                <title>Alerts | Network Anomaly Detection</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
            </Head>

            <div className="dashboard">
                <Sidebar activePage="alerts" />

                <main className="main-content">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                        <div>
                            <h1>Security Alerts</h1>
                            <p style={{ marginTop: '0.5rem' }}>Monitor and manage detected security threats</p>
                        </div>
                        {unacknowledgedCount > 0 && (
                            <button className="btn btn-primary" onClick={acknowledgeAll}>
                                ✓ Acknowledge All ({unacknowledgedCount})
                            </button>
                        )}
                    </div>

                    {/* Stats Row */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div className="stat-card">
                            <div className="stat-value">{alerts.length}</div>
                            <div className="stat-label">Total Alerts</div>
                        </div>
                        <div className="stat-card danger">
                            <div className="stat-value" style={{ color: 'var(--danger)' }}>
                                {alerts.filter(a => a.severity === 'critical').length}
                            </div>
                            <div className="stat-label">Critical</div>
                        </div>
                        <div className="stat-card warning">
                            <div className="stat-value" style={{ color: 'var(--warning)' }}>
                                {alerts.filter(a => a.severity === 'high').length}
                            </div>
                            <div className="stat-label">High</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value" style={{ color: 'var(--success)' }}>
                                {alerts.filter(a => a.is_acknowledged).length}
                            </div>
                            <div className="stat-label">Acknowledged</div>
                        </div>
                    </div>

                    {/* Filters */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Filter:</span>

                            {['all', 'critical', 'high', 'medium'].map(f => (
                                <button
                                    key={f}
                                    className={`btn ${filter === f ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setFilter(f)}
                                    style={{ textTransform: 'capitalize' }}
                                >
                                    {f}
                                </button>
                            ))}

                            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <input
                                    type="checkbox"
                                    id="showAck"
                                    checked={showAcknowledged}
                                    onChange={(e) => setShowAcknowledged(e.target.checked)}
                                />
                                <label htmlFor="showAck" style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Show acknowledged
                                </label>
                            </div>

                            <button className="btn btn-secondary" onClick={fetchAlerts}>
                                🔄 Refresh
                            </button>
                        </div>
                    </div>

                    {/* Alerts List */}
                    <div className="card">
                        {loading ? (
                            <div className="loading">
                                <div className="spinner"></div>
                            </div>
                        ) : filteredAlerts.length === 0 ? (
                            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                                <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>✅</div>
                                <p>No alerts to show</p>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                                {filteredAlerts.map((alert) => (
                                    <div
                                        key={alert.id}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '1rem',
                                            padding: '1rem',
                                            background: 'var(--bg-secondary)',
                                            borderRadius: 'var(--radius-md)',
                                            borderLeft: `4px solid ${getSeverityColor(alert.severity)}`,
                                            opacity: alert.is_acknowledged ? 0.6 : 1
                                        }}
                                    >
                                        {/* Icon */}
                                        <div style={{
                                            width: '50px',
                                            height: '50px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            background: `${getSeverityColor(alert.severity)}20`,
                                            borderRadius: 'var(--radius-md)',
                                            fontSize: '1.5rem'
                                        }}>
                                            {alert.severity === 'critical' ? '🚨' : alert.severity === 'high' ? '⚠️' : '🔔'}
                                        </div>

                                        {/* Content */}
                                        <div style={{ flex: 1 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                                                <span style={{ fontWeight: 600 }}>{alert.attack_type}</span>
                                                <span
                                                    className="badge"
                                                    style={{
                                                        background: `${getSeverityColor(alert.severity)}20`,
                                                        color: getSeverityColor(alert.severity),
                                                        textTransform: 'uppercase',
                                                        fontSize: '0.65rem'
                                                    }}
                                                >
                                                    {alert.severity}
                                                </span>
                                            </div>
                                            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                                                {formatTime(alert.timestamp)}
                                            </div>
                                        </div>

                                        {/* Score */}
                                        <div style={{ textAlign: 'right' }}>
                                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 600 }}>
                                                {alert.score.toFixed(2)}
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Score</div>
                                        </div>

                                        {/* Actions */}
                                        {!alert.is_acknowledged && (
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => acknowledgeAlert(alert.id)}
                                            >
                                                ✓ Ack
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </main>
            </div>
        </>
    )
}
