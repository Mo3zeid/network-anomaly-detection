import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'

const API_BASE = 'http://localhost:8000'

export default function Rules() {
    const [rules, setRules] = useState({
        max_packets_per_minute: 1000,
        max_bytes_per_minute: 10000000,
        max_connections_per_minute: 100,
        anomaly_score_threshold: 3.0,
        use_ml_detection: true,
        use_rate_detection: true,
        use_port_scan_detection: true,
        port_scan_threshold: 20,
        syn_flood_threshold: 100
    })
    const [whitelist, setWhitelist] = useState([])
    const [blacklist, setBlacklist] = useState([])
    const [newWhitelistIP, setNewWhitelistIP] = useState('')
    const [newBlacklistIP, setNewBlacklistIP] = useState('')
    const [ipStats, setIpStats] = useState([])
    const [saved, setSaved] = useState(false)
    const [loading, setLoading] = useState(true)

    // Fetch current rules
    useEffect(() => {
        fetchRules()
        fetchIpStats()
    }, [])

    const fetchRules = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/rules`)
            const data = await res.json()
            setRules(data.rules)
            setWhitelist(data.whitelist || [])
            setBlacklist(data.blacklist || [])
        } catch (e) {
            console.error("Error fetching rules:", e)
        } finally {
            setLoading(false)
        }
    }

    const fetchIpStats = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/rules/ip-stats`)
            const data = await res.json()
            setIpStats(data.ip_stats || [])
        } catch (e) {
            console.error("Error fetching IP stats:", e)
        }
    }

    const saveRules = async () => {
        try {
            await fetch(`${API_BASE}/api/rules`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(rules)
            })
            setSaved(true)
            setTimeout(() => setSaved(false), 3000)
        } catch (e) {
            alert("Failed to save rules: " + e.message)
        }
    }

    const addToWhitelist = async () => {
        if (!newWhitelistIP.trim()) return
        try {
            await fetch(`${API_BASE}/api/rules/whitelist/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: newWhitelistIP })
            })
            setWhitelist([...whitelist, newWhitelistIP])
            setNewWhitelistIP('')
        } catch (e) {
            alert("Error: " + e.message)
        }
    }

    const removeFromWhitelist = async (ip) => {
        try {
            await fetch(`${API_BASE}/api/rules/whitelist/remove`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip })
            })
            setWhitelist(whitelist.filter(i => i !== ip))
        } catch (e) {
            alert("Error: " + e.message)
        }
    }

    const addToBlacklist = async () => {
        if (!newBlacklistIP.trim()) return
        try {
            await fetch(`${API_BASE}/api/rules/blacklist/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: newBlacklistIP })
            })
            setBlacklist([...blacklist, newBlacklistIP])
            setNewBlacklistIP('')
        } catch (e) {
            alert("Error: " + e.message)
        }
    }

    const removeFromBlacklist = async (ip) => {
        try {
            await fetch(`${API_BASE}/api/rules/blacklist/remove`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip })
            })
            setBlacklist(blacklist.filter(i => i !== ip))
        } catch (e) {
            alert("Error: " + e.message)
        }
    }

    if (loading) {
        return <div className="dashboard"><Sidebar /><main className="main-content"><p>Loading...</p></main></div>
    }

    return (
        <>
            <Head>
                <title>Detection Rules | Network Anomaly Detection</title>
            </Head>

            <div className="dashboard">
                <Sidebar activePage="rules" />

                <main className="main-content">
                    <div style={{ marginBottom: '2rem' }}>
                        <h1>Detection Rules</h1>
                        <p style={{ marginTop: '0.5rem' }}>Configure thresholds and manage IP whitelists/blacklists</p>
                    </div>

                    {/* Rate Limits */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">📊 Rate Limits (per IP, per minute)</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Max Packets/min: {rules.max_packets_per_minute}
                                </label>
                                <input
                                    type="range"
                                    min="100"
                                    max="10000"
                                    step="100"
                                    value={rules.max_packets_per_minute}
                                    onChange={(e) => setRules({ ...rules, max_packets_per_minute: parseInt(e.target.value) })}
                                    style={{ width: '100%' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Max MB/min: {(rules.max_bytes_per_minute / 1000000).toFixed(0)}
                                </label>
                                <input
                                    type="range"
                                    min="1000000"
                                    max="100000000"
                                    step="1000000"
                                    value={rules.max_bytes_per_minute}
                                    onChange={(e) => setRules({ ...rules, max_bytes_per_minute: parseInt(e.target.value) })}
                                    style={{ width: '100%' }}
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Max Connections/min: {rules.max_connections_per_minute}
                                </label>
                                <input
                                    type="range"
                                    min="10"
                                    max="500"
                                    step="10"
                                    value={rules.max_connections_per_minute}
                                    onChange={(e) => setRules({ ...rules, max_connections_per_minute: parseInt(e.target.value) })}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Detection Methods */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">🔍 Detection Methods</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                    <input
                                        type="checkbox"
                                        checked={rules.use_ml_detection}
                                        onChange={(e) => setRules({ ...rules, use_ml_detection: e.target.checked })}
                                    />
                                    <span>Use ML Detection</span>
                                </label>
                                {rules.use_ml_detection && (
                                    <div style={{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}>
                                        <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                            ML Score Threshold: {rules.anomaly_score_threshold}
                                        </label>
                                        <input
                                            type="range"
                                            min="1"
                                            max="10"
                                            step="0.5"
                                            value={rules.anomaly_score_threshold}
                                            onChange={(e) => setRules({ ...rules, anomaly_score_threshold: parseFloat(e.target.value) })}
                                            style={{ width: '100%' }}
                                        />
                                    </div>
                                )}
                            </div>
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                    <input
                                        type="checkbox"
                                        checked={rules.use_rate_detection}
                                        onChange={(e) => setRules({ ...rules, use_rate_detection: e.target.checked })}
                                    />
                                    <span>Use Rate-Based Detection</span>
                                </label>
                            </div>
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                    <input
                                        type="checkbox"
                                        checked={rules.use_port_scan_detection}
                                        onChange={(e) => setRules({ ...rules, use_port_scan_detection: e.target.checked })}
                                    />
                                    <span>Port Scan Detection (threshold: {rules.port_scan_threshold} ports)</span>
                                </label>
                            </div>
                            <div>
                                <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    SYN Flood Threshold: {rules.syn_flood_threshold} SYN/min
                                </label>
                                <input
                                    type="range"
                                    min="10"
                                    max="500"
                                    step="10"
                                    value={rules.syn_flood_threshold}
                                    onChange={(e) => setRules({ ...rules, syn_flood_threshold: parseInt(e.target.value) })}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        </div>
                        <div style={{ marginTop: '1rem' }}>
                            <button className="btn btn-primary" onClick={saveRules}>
                                💾 Save Rules
                            </button>
                            {saved && <span style={{ marginLeft: '1rem', color: 'var(--success)' }}>✓ Saved!</span>}
                        </div>
                    </div>

                    {/* IP Lists */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                        {/* Whitelist */}
                        <div className="card" style={{ borderColor: 'var(--success)' }}>
                            <div className="card-header">
                                <span className="card-title" style={{ color: 'var(--success)' }}>✅ Whitelist (Never Flag)</span>
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                                <input
                                    type="text"
                                    placeholder="Enter IP address"
                                    value={newWhitelistIP}
                                    onChange={(e) => setNewWhitelistIP(e.target.value)}
                                    style={{
                                        flex: 1,
                                        padding: '0.5rem',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        color: 'var(--text-primary)'
                                    }}
                                />
                                <button className="btn btn-primary" onClick={addToWhitelist}>Add</button>
                            </div>
                            <div style={{ maxHeight: '150px', overflow: 'auto' }}>
                                {whitelist.map(ip => (
                                    <div key={ip} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', marginBottom: '0.25rem' }}>
                                        <span style={{ fontFamily: 'monospace' }}>{ip}</span>
                                        <button onClick={() => removeFromWhitelist(ip)} style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer' }}>✕</button>
                                    </div>
                                ))}
                                {whitelist.length === 0 && <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No whitelisted IPs</p>}
                            </div>
                        </div>

                        {/* Blacklist */}
                        <div className="card" style={{ borderColor: 'var(--danger)' }}>
                            <div className="card-header">
                                <span className="card-title" style={{ color: 'var(--danger)' }}>🚫 Blacklist (Always Flag)</span>
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                                <input
                                    type="text"
                                    placeholder="Enter IP address"
                                    value={newBlacklistIP}
                                    onChange={(e) => setNewBlacklistIP(e.target.value)}
                                    style={{
                                        flex: 1,
                                        padding: '0.5rem',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        color: 'var(--text-primary)'
                                    }}
                                />
                                <button className="btn btn-danger" onClick={addToBlacklist} style={{ background: 'var(--danger)', color: 'white' }}>Add</button>
                            </div>
                            <div style={{ maxHeight: '150px', overflow: 'auto' }}>
                                {blacklist.map(ip => (
                                    <div key={ip} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', marginBottom: '0.25rem' }}>
                                        <span style={{ fontFamily: 'monospace' }}>{ip}</span>
                                        <button onClick={() => removeFromBlacklist(ip)} style={{ background: 'none', border: 'none', color: 'var(--success)', cursor: 'pointer' }}>✓</button>
                                    </div>
                                ))}
                                {blacklist.length === 0 && <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No blacklisted IPs</p>}
                            </div>
                        </div>
                    </div>

                    {/* IP Statistics */}
                    <div className="card">
                        <div className="card-header">
                            <span className="card-title">📈 Live IP Statistics</span>
                            <button className="btn" onClick={fetchIpStats} style={{ marginLeft: 'auto' }}>🔄 Refresh</button>
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>IP Address</th>
                                        <th>Packets</th>
                                        <th>Bytes</th>
                                        <th>Connections</th>
                                        <th>Ports</th>
                                        <th>SYN Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {ipStats.map((stat, idx) => (
                                        <tr key={idx}>
                                            <td style={{ fontFamily: 'monospace' }}>{stat.ip}</td>
                                            <td>{stat.packets}</td>
                                            <td>{(stat.bytes / 1000).toFixed(1)} KB</td>
                                            <td>{stat.connections}</td>
                                            <td>{stat.ports_accessed}</td>
                                            <td>{stat.syn_count}</td>
                                        </tr>
                                    ))}
                                    {ipStats.length === 0 && (
                                        <tr>
                                            <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                                                No traffic tracked yet. Start the sniffer to see live stats.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </main>
            </div>
        </>
    )
}
