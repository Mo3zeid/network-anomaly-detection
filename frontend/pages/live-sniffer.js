import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'

const API_BASE = 'http://localhost:8000'

export default function LiveSniffer() {
    const [isRunning, setIsRunning] = useState(false)
    const [loading, setLoading] = useState(false)
    const [stats, setStats] = useState({ packets_captured: 0, active_flows: 0 })
    const [flows, setFlows] = useState([])
    const [interfaceName, setInterfaceName] = useState('')
    const [maliciousIPs, setMaliciousIPs] = useState([])
    const [blockingIP, setBlockingIP] = useState(null)

    // Fetch malicious IPs
    const fetchMaliciousIPs = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/firewall/malicious`)
            const data = await res.json()
            setMaliciousIPs(data.malicious_ips || [])
        } catch (e) {
            console.error("Error fetching malicious IPs:", e)
        }
    }

    // Check backend sniffer status and sync with frontend state
    const checkSnifferStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/sniffer/status`)
            const data = await res.json()
            setIsRunning(data.is_running)  // Sync with backend state
            setStats({
                packets_captured: data.packets_captured,
                active_flows: data.active_flows
            })
            return data.is_running
        } catch (e) {
            console.error("Error checking sniffer status:", e)
            return false
        }
    }

    // Initial load: check if sniffer is already running on backend
    useEffect(() => {
        const init = async () => {
            await checkSnifferStatus()
            await fetchMaliciousIPs()
        }
        init()
    }, [])

    // Poll for updates - runs continuously to stay in sync with backend
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                // Always check status to stay in sync
                const statusRes = await fetch(`${API_BASE}/api/sniffer/status`)
                const statusData = await statusRes.json()

                // Sync running state with backend
                setIsRunning(statusData.is_running)
                setStats({
                    packets_captured: statusData.packets_captured,
                    active_flows: statusData.active_flows
                })

                // Only fetch flows if sniffer is running
                if (statusData.is_running) {
                    const flowRes = await fetch(`${API_BASE}/api/sniffer/latest`)
                    const flowData = await flowRes.json()
                    if (flowData.flows && flowData.flows.length > 0) {
                        setFlows(prev => [...flowData.flows, ...prev].slice(0, 100))
                    }
                }

                // Update malicious IPs list
                await fetchMaliciousIPs()
            } catch (e) {
                console.error("Polling error:", e)
            }
        }, 2000)

        return () => clearInterval(interval)
    }, []) // No dependency on isRunning - always poll

    const toggleSniffer = async () => {
        setLoading(true)
        try {
            if (isRunning) {
                await fetch(`${API_BASE}/api/sniffer/stop`, { method: 'POST' })
                setIsRunning(false)
            } else {
                await fetch(`${API_BASE}/api/sniffer/start?interface=${interfaceName}`, { method: 'POST' })
                setIsRunning(true)
            }
        } catch (e) {
            alert("Failed to toggle sniffer: " + e.message)
        } finally {
            setLoading(false)
        }
    }

    const blockIP = async (ip) => {
        setBlockingIP(ip)
        try {
            const res = await fetch(`${API_BASE}/api/firewall/block`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip })
            })
            const data = await res.json()
            if (data.success) {
                await fetchMaliciousIPs()
                alert(`✅ IP ${ip} has been blocked!`)
            } else {
                alert(`❌ Failed to block IP: ${data.detail || 'Unknown error'}`)
            }
        } catch (e) {
            alert(`❌ Error blocking IP: ${e.message}\n\nNote: You need to run the backend as Administrator to modify firewall rules.`)
        } finally {
            setBlockingIP(null)
        }
    }

    const unblockIP = async (ip) => {
        setBlockingIP(ip)
        try {
            const res = await fetch(`${API_BASE}/api/firewall/unblock`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip })
            })
            const data = await res.json()
            if (data.success) {
                await fetchMaliciousIPs()
                alert(`✅ IP ${ip} has been unblocked!`)
            } else {
                alert(`❌ Failed to unblock IP: ${data.detail || 'Unknown error'}`)
            }
        } catch (e) {
            alert(`❌ Error unblocking IP: ${e.message}`)
        } finally {
            setBlockingIP(null)
        }
    }

    return (
        <>
            <Head>
                <title>Live Sniffer | Network Anomaly Detection</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
            </Head>

            <div className="dashboard">
                <Sidebar activePage="sniffer" />

                <main className="main-content">
                    <div style={{ marginBottom: '2rem' }}>
                        <h1>Live Packet Sniffer</h1>
                        <p style={{ marginTop: '0.5rem' }}>Real-time packet capture with <strong style={{ color: 'var(--accent-primary)' }}>AI-Powered Anomaly Detection</strong> (99.9% Accuracy)</p>
                    </div>

                    {/* Controls */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            <div style={{ flex: 1 }}>
                                <input
                                    type="text"
                                    placeholder="Interface (e.g. WiFi, eth0) - Leave empty for default"
                                    value={interfaceName}
                                    onChange={(e) => setInterfaceName(e.target.value)}
                                    disabled={isRunning}
                                    style={{
                                        width: '100%',
                                        padding: '0.75rem',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        color: 'var(--text-primary)'
                                    }}
                                />
                            </div>
                            <button
                                className={`btn ${isRunning ? 'btn-danger' : 'btn-primary'}`}
                                onClick={toggleSniffer}
                                disabled={loading}
                                style={{ minWidth: '150px' }}
                            >
                                {loading ? '...' : isRunning ? '🛑 Stop Sniffing' : '▶ Start Sniffing'}
                            </button>
                        </div>
                    </div>

                    {/* Stats */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div className="stat-card">
                            <div className="stat-value">{stats.packets_captured}</div>
                            <div className="stat-label">Packets Captured</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.active_flows}</div>
                            <div className="stat-label">Active Flows</div>
                        </div>
                        <div className="stat-card" style={{ borderColor: 'var(--danger)' }}>
                            <div className="stat-value" style={{ color: 'var(--danger)' }}>{maliciousIPs.length}</div>
                            <div className="stat-label">Malicious IPs</div>
                        </div>
                    </div>

                    {/* Malicious IPs Section */}
                    {maliciousIPs.length > 0 && (
                        <div className="card" style={{ marginBottom: '1.5rem', borderColor: 'var(--danger)' }}>
                            <div className="card-header" style={{ borderBottomColor: 'rgba(239, 68, 68, 0.3)' }}>
                                <span className="card-title" style={{ color: 'var(--danger)' }}>🚨 Malicious Traffic IPs</span>
                            </div>
                            <div style={{ overflowX: 'auto' }}>
                                <table className="table">
                                    <thead>
                                        <tr>
                                            <th>IP Address</th>
                                            <th>Attack Type</th>
                                            <th>Score</th>
                                            <th>Hits</th>
                                            <th>Status</th>
                                            <th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {maliciousIPs.map((item, idx) => (
                                            <tr key={idx}>
                                                <td style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>{item.ip}</td>
                                                <td><span className="badge badge-danger">{item.attack_type}</span></td>
                                                <td>{item.score?.toFixed(2)}</td>
                                                <td>{item.hit_count}</td>
                                                <td>
                                                    <span className={`badge ${item.blocked ? 'badge-success' : 'badge-warning'}`}>
                                                        {item.blocked ? '🔒 Blocked' : '⚠️ Active'}
                                                    </span>
                                                </td>
                                                <td>
                                                    {item.blocked ? (
                                                        <button
                                                            className="btn btn-sm"
                                                            onClick={() => unblockIP(item.ip)}
                                                            disabled={blockingIP === item.ip}
                                                            style={{ background: 'var(--success)', color: 'white', padding: '0.25rem 0.75rem', fontSize: '0.8rem' }}
                                                        >
                                                            {blockingIP === item.ip ? '...' : '✓ Unblock'}
                                                        </button>
                                                    ) : (
                                                        <button
                                                            className="btn btn-sm btn-danger"
                                                            onClick={() => blockIP(item.ip)}
                                                            disabled={blockingIP === item.ip}
                                                            style={{ padding: '0.25rem 0.75rem', fontSize: '0.8rem' }}
                                                        >
                                                            {blockingIP === item.ip ? '...' : '🚫 Block'}
                                                        </button>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Live Results */}
                    <div className="card">
                        <div className="card-header">
                            <span className="card-title">Analysed Flows (Live)</span>
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Source</th>
                                        <th>Destination</th>
                                        <th>Protocol</th>
                                        <th>Status</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {flows.map((flow, idx) => (
                                        <tr key={idx}>
                                            <td style={{ fontFamily: 'monospace' }}>{flow.src_ip}:{flow.src_port}</td>
                                            <td style={{ fontFamily: 'monospace' }}>{flow.dst_ip}:{flow.dst_port}</td>
                                            <td><span className="badge">{flow.protocol}</span></td>
                                            <td>
                                                <span className={`badge ${flow.is_attack ? 'badge-danger' : 'badge-success'}`}>
                                                    {flow.is_attack ? 'Anomaly' : 'Normal'}
                                                </span>
                                            </td>
                                            <td>
                                                {flow.is_attack && (
                                                    <div style={{ fontSize: '0.85rem' }}>
                                                        <strong>{flow.attack_type}</strong> (Score: {flow.score?.toFixed(2)})
                                                    </div>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                    {flows.length === 0 && (
                                        <tr>
                                            <td colSpan="5" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                                                {isRunning ? 'Waiting for completed flows...' : 'Start sniffer to see live traffic'}
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </main>
            </div>

            <style jsx>{`
                .btn-danger {
                    background: var(--danger);
                    color: white;
                }
                .btn-danger:hover {
                    background: #dc2626;
                }
                .badge-warning {
                    background: #f59e0b;
                    color: white;
                }
            `}</style>
        </>
    )
}
