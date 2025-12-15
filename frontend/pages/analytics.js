import { useState, useEffect, useRef, useCallback } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'
import { Chart, registerables } from 'chart.js'

if (typeof window !== 'undefined') {
    Chart.register(...registerables)
}

const API_BASE = 'http://localhost:8000'

// IP Detail Modal Component
function IPDetailModal({ ip, onClose }) {
    const [details, setDetails] = useState(null)
    const [loading, setLoading] = useState(true)
    const [activeTab, setActiveTab] = useState('overview')

    useEffect(() => {
        const fetchDetails = async () => {
            try {
                const response = await fetch(`${API_BASE}/api/ip/${ip}/details`)
                if (response.ok) {
                    const data = await response.json()
                    setDetails(data)
                }
            } catch (err) {
                console.error('Failed to fetch IP details:', err)
            } finally {
                setLoading(false)
            }
        }
        fetchDetails()
    }, [ip])

    const getThreatColor = (level) => {
        switch (level) {
            case 'critical': return '#ef4444'
            case 'high': return '#f59e0b'
            case 'medium': return '#eab308'
            case 'low': return '#10b981'
            default: return '#64748b'
        }
    }

    const getCountryFlag = (code) => {
        if (!code) return '🌍'
        return code.toUpperCase().replace(/./g, char =>
            String.fromCodePoint(127397 + char.charCodeAt())
        )
    }

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content ip-detail-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <span style={{ fontSize: '2rem' }}>
                            {details?.geolocation?.country_code ? getCountryFlag(details.geolocation.country_code) : '🌐'}
                        </span>
                        <div>
                            <h2 style={{ margin: 0 }}>{ip}</h2>
                            <span style={{ color: '#64748b', fontSize: '0.875rem' }}>
                                {details?.geolocation?.city}, {details?.geolocation?.country || 'Unknown'}
                            </span>
                        </div>
                    </div>
                    <button className="close-btn" onClick={onClose}>×</button>
                </div>

                {loading ? (
                    <div className="loading"><div className="spinner"></div></div>
                ) : details ? (
                    <>
                        {/* Threat Score Banner */}
                        <div style={{
                            background: `linear-gradient(135deg, ${getThreatColor(details.threat_assessment?.level)}22, transparent)`,
                            borderLeft: `4px solid ${getThreatColor(details.threat_assessment?.level)}`,
                            padding: '1rem 1.5rem',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}>
                            <div>
                                <div style={{ color: '#94a3b8', fontSize: '0.75rem', textTransform: 'uppercase' }}>
                                    Threat Level
                                </div>
                                <div style={{
                                    color: getThreatColor(details.threat_assessment?.level),
                                    fontSize: '1.25rem',
                                    fontWeight: '700',
                                    textTransform: 'uppercase'
                                }}>
                                    {details.threat_assessment?.level || 'Unknown'}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{
                                    fontSize: '2.5rem',
                                    fontWeight: '700',
                                    color: getThreatColor(details.threat_assessment?.level)
                                }}>
                                    {details.threat_assessment?.score || 0}
                                </div>
                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>/ 100</div>
                            </div>
                        </div>

                        {/* Tabs */}
                        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', padding: '0 1.5rem' }}>
                            {['overview', 'network', 'activity'].map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    style={{
                                        padding: '1rem 1.5rem',
                                        background: 'transparent',
                                        border: 'none',
                                        color: activeTab === tab ? '#00d4ff' : '#64748b',
                                        borderBottom: activeTab === tab ? '2px solid #00d4ff' : '2px solid transparent',
                                        cursor: 'pointer',
                                        textTransform: 'capitalize',
                                        fontWeight: '500'
                                    }}
                                >
                                    {tab}
                                </button>
                            ))}
                        </div>

                        {/* Tab Content */}
                        <div style={{ padding: '1.5rem' }}>
                            {activeTab === 'overview' && (
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                                    {/* Status Cards */}
                                    <div className="detail-card">
                                        <h4>Status</h4>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                                            {details.status?.is_malicious && <span className="badge badge-danger">Malicious</span>}
                                            {details.status?.is_blocked && <span className="badge badge-warning">Blocked</span>}
                                            {details.status?.is_whitelisted && <span className="badge badge-success">Whitelisted</span>}
                                            {details.status?.is_blacklisted && <span className="badge badge-danger">Blacklisted</span>}
                                            {!details.status?.is_malicious && !details.status?.is_blocked &&
                                                !details.status?.is_whitelisted && !details.status?.is_blacklisted &&
                                                <span className="badge badge-info">Normal</span>}
                                        </div>
                                    </div>

                                    {/* Location */}
                                    <div className="detail-card">
                                        <h4>Location</h4>
                                        <div style={{ marginTop: '0.5rem' }}>
                                            <div>{details.geolocation?.city}, {details.geolocation?.region}</div>
                                            <div style={{ color: '#64748b' }}>{details.geolocation?.country}</div>
                                        </div>
                                    </div>

                                    {/* Statistics */}
                                    <div className="detail-card" style={{ gridColumn: 'span 2' }}>
                                        <h4>Traffic Statistics</h4>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginTop: '0.75rem' }}>
                                            <div>
                                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#00d4ff' }}>
                                                    {details.statistics?.total_packets || 0}
                                                </div>
                                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Packets</div>
                                            </div>
                                            <div>
                                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#8b5cf6' }}>
                                                    {((details.statistics?.total_bytes || 0) / 1024).toFixed(1)} KB
                                                </div>
                                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Data</div>
                                            </div>
                                            <div>
                                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#10b981' }}>
                                                    {details.statistics?.total_connections || 0}
                                                </div>
                                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Connections</div>
                                            </div>
                                            <div>
                                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f59e0b' }}>
                                                    {details.statistics?.ports_accessed || 0}
                                                </div>
                                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Ports</div>
                                            </div>
                                            <div>
                                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#ef4444' }}>
                                                    {details.statistics?.syn_count || 0}
                                                </div>
                                                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>SYN Flags</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {activeTab === 'network' && (
                                <div className="detail-card">
                                    <h4>Network Information</h4>
                                    <table style={{ width: '100%', marginTop: '1rem' }}>
                                        <tbody>
                                            <tr>
                                                <td style={{ color: '#64748b', padding: '0.5rem 0' }}>ISP</td>
                                                <td style={{ textAlign: 'right' }}>{details.network?.isp || 'Unknown'}</td>
                                            </tr>
                                            <tr>
                                                <td style={{ color: '#64748b', padding: '0.5rem 0' }}>Organization</td>
                                                <td style={{ textAlign: 'right' }}>{details.network?.org || 'Unknown'}</td>
                                            </tr>
                                            <tr>
                                                <td style={{ color: '#64748b', padding: '0.5rem 0' }}>AS Number</td>
                                                <td style={{ textAlign: 'right' }}>{details.network?.as_number || 'Unknown'}</td>
                                            </tr>
                                            <tr>
                                                <td style={{ color: '#64748b', padding: '0.5rem 0' }}>Coordinates</td>
                                                <td style={{ textAlign: 'right' }}>
                                                    {details.geolocation?.latitude?.toFixed(4)}, {details.geolocation?.longitude?.toFixed(4)}
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            )}

                            {activeTab === 'activity' && (
                                <div className="detail-card">
                                    <h4>Recent Activity ({details.activity_count || 0} total)</h4>
                                    {details.recent_activities?.length > 0 ? (
                                        <div style={{ marginTop: '1rem' }}>
                                            {details.recent_activities.map((activity, i) => (
                                                <div key={i} style={{
                                                    padding: '0.75rem',
                                                    borderLeft: '3px solid #00d4ff',
                                                    background: 'rgba(0,212,255,0.05)',
                                                    marginBottom: '0.5rem',
                                                    borderRadius: '0 4px 4px 0'
                                                }}>
                                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                        <span style={{ fontWeight: '500' }}>{activity.type}</span>
                                                        <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
                                                            {new Date(activity.timestamp).toLocaleTimeString()}
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div style={{ color: '#64748b', marginTop: '1rem' }}>No recent activity recorded</div>
                                    )}
                                </div>
                            )}
                        </div>
                    </>
                ) : (
                    <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                        Failed to load IP details
                    </div>
                )}
            </div>
        </div>
    )
}

export default function Analytics() {
    const [liveStats, setLiveStats] = useState(null)
    const [evaluation, setEvaluation] = useState(null)
    const [threatTimeline, setThreatTimeline] = useState(null)
    const [loading, setLoading] = useState(true)
    const [autoRefresh, setAutoRefresh] = useState(true)
    const [refreshInterval, setRefreshInterval] = useState(5000)
    const [selectedIP, setSelectedIP] = useState(null)
    const [lastUpdate, setLastUpdate] = useState(null)

    const timelineChartRef = useRef(null)
    const timelineChartInstance = useRef(null)
    const attackChartRef = useRef(null)
    const attackChartInstance = useRef(null)
    const threatChartRef = useRef(null)
    const threatChartInstance = useRef(null)

    // Fetch live stats
    const fetchLiveStats = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/live-stats`)
            if (response.ok) {
                const data = await response.json()
                setLiveStats(data)
                setLastUpdate(new Date())
            }
        } catch (err) {
            console.error('Failed to fetch live stats:', err)
        }
    }, [])

    // Fetch evaluation metrics
    const fetchEvaluation = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/evaluate`)
            if (response.ok) {
                const data = await response.json()
                setEvaluation(data)
            }
        } catch (err) {
            console.error('Failed to fetch evaluation:', err)
        }
    }, [])

    // Fetch threat timeline
    const fetchThreatTimeline = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/threat-timeline?hours=24`)
            if (response.ok) {
                const data = await response.json()
                setThreatTimeline(data)
            }
        } catch (err) {
            console.error('Failed to fetch timeline:', err)
        } finally {
            setLoading(false)
        }
    }, [])

    // Initial fetch
    useEffect(() => {
        fetchLiveStats()
        fetchEvaluation()
        fetchThreatTimeline()
    }, [fetchLiveStats, fetchEvaluation, fetchThreatTimeline])

    // Auto-refresh
    useEffect(() => {
        if (!autoRefresh) return

        const interval = setInterval(() => {
            fetchLiveStats()
        }, refreshInterval)

        return () => clearInterval(interval)
    }, [autoRefresh, refreshInterval, fetchLiveStats])

    // Threat Timeline Chart
    useEffect(() => {
        if (!threatChartRef.current || !threatTimeline?.timeline) return

        if (threatChartInstance.current) {
            threatChartInstance.current.destroy()
        }

        const labels = threatTimeline.timeline.map(t => t.hour)
        const criticalData = threatTimeline.timeline.map(t => t.critical)
        const highData = threatTimeline.timeline.map(t => t.high)
        const mediumData = threatTimeline.timeline.map(t => t.medium)

        threatChartInstance.current = new Chart(threatChartRef.current, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Critical',
                        data: criticalData,
                        backgroundColor: '#ef4444',
                        borderRadius: 4,
                    },
                    {
                        label: 'High',
                        data: highData,
                        backgroundColor: '#f59e0b',
                        borderRadius: 4,
                    },
                    {
                        label: 'Medium',
                        data: mediumData,
                        backgroundColor: '#eab308',
                        borderRadius: 4,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { color: '#94a3b8', usePointStyle: true } }
                },
                scales: {
                    x: { stacked: true, grid: { display: false }, ticks: { color: '#64748b' } },
                    y: { stacked: true, grid: { color: 'rgba(42, 53, 72, 0.5)' }, ticks: { color: '#64748b' } }
                }
            }
        })

        return () => threatChartInstance.current?.destroy()
    }, [threatTimeline])

    // Attack Distribution Chart
    useEffect(() => {
        if (!attackChartRef.current || !liveStats?.attack_distribution) return

        if (attackChartInstance.current) {
            attackChartInstance.current.destroy()
        }

        const dist = liveStats.attack_distribution
        const labels = Object.keys(dist)
        const data = Object.values(dist)

        if (labels.length === 0) return

        const colors = ['#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6', '#10b981', '#ec4899', '#06b6d4']

        attackChartInstance.current = new Chart(attackChartRef.current, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: { position: 'right', labels: { color: '#94a3b8', usePointStyle: true, padding: 15 } }
                }
            }
        })

        return () => attackChartInstance.current?.destroy()
    }, [liveStats])

    const formatUptime = (seconds) => {
        const hrs = Math.floor(seconds / 3600)
        const mins = Math.floor((seconds % 3600) / 60)
        return `${hrs}h ${mins}m`
    }

    return (
        <>
            <Head>
                <title>Live Analytics | Network Anomaly Detection</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
            </Head>

            <style jsx global>{`
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(4px);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                .modal-content {
                    background: var(--bg-card);
                    border-radius: 12px;
                    max-width: 700px;
                    width: 95%;
                    max-height: 85vh;
                    overflow-y: auto;
                    border: 1px solid var(--border-color);
                }
                .modal-header {
                    padding: 1.5rem;
                    border-bottom: 1px solid var(--border-color);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .close-btn {
                    background: transparent;
                    border: none;
                    color: #64748b;
                    font-size: 2rem;
                    cursor: pointer;
                    line-height: 1;
                }
                .close-btn:hover { color: #ef4444; }
                .detail-card {
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    padding: 1rem;
                }
                .detail-card h4 {
                    color: #94a3b8;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    margin: 0;
                }
                .live-indicator {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.875rem;
                    color: #10b981;
                }
                .live-indicator .dot {
                    width: 8px;
                    height: 8px;
                    background: #10b981;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; transform: scale(1); }
                    50% { opacity: 0.5; transform: scale(1.2); }
                }
                .top-talker-row {
                    display: grid;
                    grid-template-columns: 1fr repeat(4, auto);
                    gap: 1rem;
                    padding: 0.75rem;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .top-talker-row:hover {
                    background: rgba(0, 212, 255, 0.1);
                }
                .metric-card {
                    background: linear-gradient(135deg, rgba(0,212,255,0.1), transparent);
                    border: 1px solid rgba(0,212,255,0.2);
                    border-radius: 12px;
                    padding: 1.25rem;
                    text-align: center;
                }
                .metric-value {
                    font-size: 2rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #00d4ff, #7c3aed);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                .metric-label {
                    color: #64748b;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    margin-top: 0.25rem;
                }
            `}</style>

            <div className="dashboard">
                <Sidebar activePage="analytics" />

                <main className="main-content">
                    {/* Header with Live Controls */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                        <div>
                            <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                Live Analytics
                                {autoRefresh && (
                                    <span className="live-indicator">
                                        <span className="dot"></span>
                                        LIVE
                                    </span>
                                )}
                            </h1>
                            <p style={{ marginTop: '0.5rem', color: '#64748b' }}>
                                {lastUpdate ? `Last updated: ${lastUpdate.toLocaleTimeString()}` : 'Loading...'}
                                {liveStats?.uptime_seconds && ` • Uptime: ${formatUptime(liveStats.uptime_seconds)}`}
                            </p>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            <select
                                value={refreshInterval}
                                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                                style={{
                                    padding: '0.5rem 1rem',
                                    background: 'var(--bg-card)',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: 'var(--radius-md)',
                                    color: 'var(--text-primary)'
                                }}
                            >
                                <option value={2000}>2s</option>
                                <option value={5000}>5s</option>
                                <option value={10000}>10s</option>
                                <option value={30000}>30s</option>
                            </select>
                            <button
                                onClick={() => setAutoRefresh(!autoRefresh)}
                                className={`btn ${autoRefresh ? 'btn-danger' : 'btn-primary'}`}
                            >
                                {autoRefresh ? '⏸ Pause' : '▶ Resume'}
                            </button>
                            <button onClick={() => { fetchLiveStats(); fetchThreatTimeline(); }} className="btn btn-secondary">
                                🔄 Refresh Now
                            </button>
                        </div>
                    </div>

                    {loading ? (
                        <div className="loading"><div className="spinner"></div></div>
                    ) : (
                        <>
                            {/* Live Metrics Row */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                                <div className="metric-card">
                                    <div className="metric-value">{liveStats?.summary?.total_flows?.toLocaleString() || 0}</div>
                                    <div className="metric-label">Total Flows</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value" style={{ background: 'linear-gradient(135deg, #ef4444, #f59e0b)', WebkitBackgroundClip: 'text' }}>
                                        {liveStats?.summary?.total_anomalies || 0}
                                    </div>
                                    <div className="metric-label">Anomalies</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value">{liveStats?.summary?.detection_rate || 0}%</div>
                                    <div className="metric-label">Detection Rate</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value" style={{ background: 'linear-gradient(135deg, #ef4444, #dc2626)', WebkitBackgroundClip: 'text' }}>
                                        {liveStats?.summary?.malicious_ips || 0}
                                    </div>
                                    <div className="metric-label">Malicious IPs</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value" style={{ background: 'linear-gradient(135deg, #f59e0b, #eab308)', WebkitBackgroundClip: 'text' }}>
                                        {liveStats?.summary?.blocked_ips || 0}
                                    </div>
                                    <div className="metric-label">Blocked IPs</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-value" style={{ background: 'linear-gradient(135deg, #10b981, #059669)', WebkitBackgroundClip: 'text' }}>
                                        {liveStats?.sniffer?.packets_captured || 0}
                                    </div>
                                    <div className="metric-label">Packets</div>
                                </div>
                            </div>

                            {/* Charts Row */}
                            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">Threat Timeline (24h)</span>
                                    </div>
                                    <div style={{ height: '280px', padding: '1rem' }}>
                                        <canvas ref={threatChartRef}></canvas>
                                    </div>
                                </div>

                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">Attack Distribution</span>
                                    </div>
                                    <div style={{ height: '280px', padding: '1rem' }}>
                                        {liveStats?.attack_distribution && Object.keys(liveStats.attack_distribution).length > 0 ? (
                                            <canvas ref={attackChartRef}></canvas>
                                        ) : (
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                                                No attacks detected yet
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Bottom Row: Top Talkers & Recent Alerts */}
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                                {/* Top Talkers */}
                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">🔥 Top Talkers</span>
                                        <span style={{ color: '#64748b', fontSize: '0.75rem' }}>Click for details</span>
                                    </div>
                                    <div style={{ padding: '0.5rem' }}>
                                        <div style={{
                                            display: 'grid',
                                            gridTemplateColumns: '1fr repeat(4, auto)',
                                            gap: '1rem',
                                            padding: '0.5rem 0.75rem',
                                            color: '#64748b',
                                            fontSize: '0.75rem',
                                            textTransform: 'uppercase'
                                        }}>
                                            <span>IP Address</span>
                                            <span>Packets</span>
                                            <span>Bytes</span>
                                            <span>Conns</span>
                                            <span>SYN</span>
                                        </div>
                                        {liveStats?.top_talkers?.length > 0 ? (
                                            liveStats.top_talkers.slice(0, 8).map((talker, i) => (
                                                <div
                                                    key={i}
                                                    className="top-talker-row"
                                                    onClick={() => setSelectedIP(talker.ip)}
                                                >
                                                    <span style={{ fontWeight: '500', color: '#00d4ff' }}>{talker.ip}</span>
                                                    <span>{talker.packets}</span>
                                                    <span>{(talker.bytes / 1024).toFixed(1)} KB</span>
                                                    <span>{talker.connections}</span>
                                                    <span style={{ color: talker.syn_count > 10 ? '#ef4444' : 'inherit' }}>
                                                        {talker.syn_count}
                                                    </span>
                                                </div>
                                            ))
                                        ) : (
                                            <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                                                No traffic recorded yet
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Recent Alerts */}
                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">🚨 Recent Alerts</span>
                                    </div>
                                    <div style={{ padding: '1rem' }}>
                                        {liveStats?.recent_alerts?.length > 0 ? (
                                            liveStats.recent_alerts.map((alert, i) => (
                                                <div
                                                    key={i}
                                                    style={{
                                                        padding: '0.75rem',
                                                        borderLeft: `3px solid ${alert.severity === 'critical' ? '#ef4444' :
                                                                alert.severity === 'high' ? '#f59e0b' : '#eab308'
                                                            }`,
                                                        background: 'rgba(0,0,0,0.2)',
                                                        marginBottom: '0.5rem',
                                                        borderRadius: '0 8px 8px 0'
                                                    }}
                                                >
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                        <div>
                                                            <span className={`badge badge-${alert.severity === 'critical' ? 'danger' :
                                                                    alert.severity === 'high' ? 'warning' : 'info'
                                                                }`}>{alert.severity}</span>
                                                            <span style={{ marginLeft: '0.75rem', fontWeight: '500' }}>
                                                                {alert.attack_type}
                                                            </span>
                                                        </div>
                                                        <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
                                                            {new Date(alert.timestamp).toLocaleTimeString()}
                                                        </span>
                                                    </div>
                                                </div>
                                            ))
                                        ) : (
                                            <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                                                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</div>
                                                No recent alerts
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Model Performance */}
                            <div className="card" style={{ marginTop: '1.5rem' }}>
                                <div className="card-header">
                                    <span className="card-title">📊 Model Performance</span>
                                </div>
                                <div style={{ padding: '1.5rem' }}>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem' }}>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#00d4ff' }}>
                                                {evaluation?.training_metrics?.training_accuracy
                                                    ? `${(evaluation.training_metrics.training_accuracy * 100).toFixed(0)}%`
                                                    : 'N/A'}
                                            </div>
                                            <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Accuracy</div>
                                        </div>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#8b5cf6' }}>
                                                {evaluation?.training_metrics?.training_precision
                                                    ? `${(evaluation.training_metrics.training_precision * 100).toFixed(0)}%`
                                                    : 'N/A'}
                                            </div>
                                            <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Precision</div>
                                        </div>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#10b981' }}>
                                                {evaluation?.training_metrics?.training_recall
                                                    ? `${(evaluation.training_metrics.training_recall * 100).toFixed(0)}%`
                                                    : 'N/A'}
                                            </div>
                                            <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Recall</div>
                                        </div>
                                        <div style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#f59e0b' }}>
                                                {evaluation?.training_metrics?.training_f1
                                                    ? `${(evaluation.training_metrics.training_f1 * 100).toFixed(0)}%`
                                                    : 'N/A'}
                                            </div>
                                            <div style={{ color: '#64748b', fontSize: '0.875rem' }}>F1 Score</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </main>
            </div>

            {/* IP Detail Modal */}
            {selectedIP && (
                <IPDetailModal ip={selectedIP} onClose={() => setSelectedIP(null)} />
            )}
        </>
    )
}
