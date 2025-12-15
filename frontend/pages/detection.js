import { useState, useRef } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'

const API_BASE = 'http://localhost:8000'

export default function Detection() {
    const [file, setFile] = useState(null)
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [method, setMethod] = useState('combined')
    const fileInputRef = useRef(null)

    // Handle file selection
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            setFile(selectedFile)
            setResults(null)
            setError(null)
        }
    }

    // Parse CSV file
    const parseCSV = (text) => {
        const lines = text.trim().split('\n')
        const headers = lines[0].split(',').map(h => h.trim())

        const flows = []
        for (let i = 1; i < Math.min(lines.length, 101); i++) { // Limit to 100 rows
            const values = lines[i].split(',')
            const flow = {}
            headers.forEach((header, idx) => {
                flow[header] = values[idx]?.trim()
            })
            flows.push(flow)
        }
        return flows
    }

    // Helper to safely parse floats (handle Infinity/NaN)
    const safeFloat = (val) => {
        const num = parseFloat(val)
        return isFinite(num) ? num : 0
    }

    // Map CSV columns to API format
    const mapToApiFormat = (flows) => {
        return flows.map(flow => ({
            flow_duration: safeFloat(flow['Flow Duration']),
            total_fwd_packets: parseInt(flow['Total Fwd Packets']) || 0,
            total_bwd_packets: parseInt(flow['Total Backward Packets']) || 0,
            flow_bytes_per_sec: safeFloat(flow['Flow Bytes/s']),
            flow_packets_per_sec: safeFloat(flow['Flow Packets/s']),
            fwd_packet_length_mean: safeFloat(flow['Fwd Packet Length Mean']),
            bwd_packet_length_mean: safeFloat(flow['Bwd Packet Length Mean']),
            flow_iat_mean: safeFloat(flow['Flow IAT Mean']),
            fwd_iat_mean: safeFloat(flow['Fwd IAT Mean']),
            bwd_iat_mean: safeFloat(flow['Bwd IAT Mean']),
            fwd_psh_flags: parseInt(flow['Fwd PSH Flags']) || 0,
            syn_flag_count: parseInt(flow['SYN Flag Count']) || 0,
            ack_flag_count: parseInt(flow['ACK Flag Count']) || 0,
            packet_length_variance: safeFloat(flow['Packet Length Variance']),
            average_packet_size: safeFloat(flow['Average Packet Size'])
        }))
    }

    // Analyze the file
    const analyzeFile = async () => {
        if (!file) return

        setLoading(true)
        setError(null)

        try {
            // Read file content
            const text = await file.text()
            const flows = parseCSV(text)
            const apiFlows = mapToApiFormat(flows)

            // Send to API
            const response = await fetch(`${API_BASE}/api/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    flows: apiFlows,
                    method: method
                })
            })

            if (!response.ok) {
                throw new Error('API request failed')
            }

            const data = await response.json()
            setResults(data)
        } catch (err) {
            setError(err.message || 'Analysis failed')
        } finally {
            setLoading(false)
        }
    }

    // Export results as CSV
    const exportResults = () => {
        if (!results) return

        const headers = ['Index', 'Is Anomaly', 'Score', 'Attack Type', 'Confidence']
        const rows = results.results.map(r => [
            r.index,
            r.is_anomaly,
            r.score.toFixed(4),
            r.attack_type || 'N/A',
            r.attack_confidence?.toFixed(4) || 'N/A'
        ])

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
        const blob = new Blob([csv], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)

        const a = document.createElement('a')
        a.href = url
        a.download = 'detection_results.csv'
        a.click()
    }

    return (
        <>
            <Head>
                <title>Detection | Network Anomaly Detection</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
            </Head>

            <div className="dashboard">
                <Sidebar activePage="detection" />

                <main className="main-content">
                    <div style={{ marginBottom: '2rem' }}>
                        <h1>Anomaly Detection</h1>
                        <p style={{ marginTop: '0.5rem' }}>Upload a CSV file to analyze network traffic using <strong style={{ color: 'var(--accent-primary)' }}>Two-Stage XGBoost AI</strong></p>
                    </div>

                    {/* Upload Section */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">Upload Network Data</span>
                        </div>

                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                            {/* File Input */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    CSV File
                                </label>
                                <input
                                    type="file"
                                    accept=".csv"
                                    ref={fileInputRef}
                                    onChange={handleFileChange}
                                    style={{ display: 'none' }}
                                />
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    style={{
                                        padding: '1rem',
                                        border: '2px dashed var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        cursor: 'pointer',
                                        textAlign: 'center',
                                        transition: 'all 0.2s'
                                    }}
                                    onMouseOver={(e) => e.currentTarget.style.borderColor = 'var(--accent-primary)'}
                                    onMouseOut={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
                                >
                                    {file ? (
                                        <div>
                                            <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📄</div>
                                            <div style={{ fontWeight: 500 }}>{file.name}</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                {(file.size / 1024).toFixed(1)} KB
                                            </div>
                                        </div>
                                    ) : (
                                        <div>
                                            <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📁</div>
                                            <div>Click to select CSV file</div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Detection Method */}
                            <div style={{ minWidth: '180px' }}>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Detection Method
                                </label>
                                <select
                                    value={method}
                                    onChange={(e) => setMethod(e.target.value)}
                                    style={{
                                        width: '100%',
                                        padding: '0.75rem',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        color: 'var(--text-primary)',
                                        fontSize: '0.875rem'
                                    }}
                                >
                                    <option value="combined">Combined (All Methods)</option>
                                    <option value="statistical">Statistical Only</option>
                                    <option value="isolation_forest">Isolation Forest Only</option>
                                </select>
                            </div>

                            {/* Analyze Button */}
                            <button
                                className="btn btn-primary"
                                onClick={analyzeFile}
                                disabled={!file || loading}
                                style={{ height: '46px' }}
                            >
                                {loading ? '⏳ Analyzing...' : '🔍 Analyze'}
                            </button>
                        </div>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="card" style={{ background: 'rgba(239, 68, 68, 0.1)', borderColor: 'var(--danger)', marginBottom: '1.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--danger)' }}>
                                <span>⚠️</span>
                                <span>{error}</span>
                            </div>
                        </div>
                    )}

                    {/* Results */}
                    {results && (
                        <>
                            {/* Summary Cards */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                                <div className="stat-card">
                                    <div className="stat-value">{results.total_flows}</div>
                                    <div className="stat-label">Total Flows</div>
                                </div>
                                <div className="stat-card danger">
                                    <div className="stat-value" style={{ background: 'var(--danger)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                                        {results.anomalies_detected}
                                    </div>
                                    <div className="stat-label">Anomalies Detected</div>
                                </div>
                                <div className="stat-card warning">
                                    <div className="stat-value" style={{ background: 'var(--warning)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                                        {(results.detection_rate * 100).toFixed(1)}%
                                    </div>
                                    <div className="stat-label">Detection Rate</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{results.method_used}</div>
                                    <div className="stat-label">Method Used</div>
                                </div>
                            </div>

                            {/* Results Table */}
                            <div className="card">
                                <div className="card-header">
                                    <span className="card-title">Detection Results</span>
                                    <button className="btn btn-secondary" onClick={exportResults}>
                                        📥 Export CSV
                                    </button>
                                </div>

                                <div style={{ overflowX: 'auto' }}>
                                    <table className="table">
                                        <thead>
                                            <tr>
                                                <th>#</th>
                                                <th>Status</th>
                                                <th>Anomaly Score</th>
                                                <th>Attack Type</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {results.results.slice(0, 50).map((result, idx) => (
                                                <tr key={idx}>
                                                    <td>{result.index}</td>
                                                    <td>
                                                        <span className={`badge ${result.is_anomaly ? 'badge-danger' : 'badge-success'}`}>
                                                            {result.is_anomaly ? '⚠️ Anomaly' : '✓ Normal'}
                                                        </span>
                                                    </td>
                                                    <td>
                                                        <span style={{ fontFamily: 'var(--font-mono)' }}>
                                                            {result.score.toFixed(4)}
                                                        </span>
                                                    </td>
                                                    <td>{result.attack_type || '-'}</td>
                                                    <td>
                                                        {result.attack_confidence ? (
                                                            <span style={{ fontFamily: 'var(--font-mono)' }}>
                                                                {(result.attack_confidence * 100).toFixed(1)}%
                                                            </span>
                                                        ) : '-'}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {results.results.length > 50 && (
                                    <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                                        Showing 50 of {results.results.length} results. Export to see all.
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </main>
            </div>
        </>
    )
}
