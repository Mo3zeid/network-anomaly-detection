import { useState } from 'react'

export default function TrafficMonitor({ data, onClose }) {
    const [selectedFlow, setSelectedFlow] = useState(null)
    const [filter, setFilter] = useState('all') // all, attacks, normal

    if (!data) return null

    const flows = data.flows || []

    const filteredFlows = flows.filter(flow => {
        if (filter === 'attacks') return flow.is_attack
        if (filter === 'normal') return !flow.is_attack
        return true
    })

    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return '#ef4444'
            case 'high': return '#f59e0b'
            case 'medium': return '#3b82f6'
            default: return '#10b981'
        }
    }

    const getProtocolBadge = (protocol) => {
        const colors = {
            'TCP': '#3b82f6',
            'UDP': '#8b5cf6',
            'ICMP': '#f59e0b'
        }
        return colors[protocol] || '#64748b'
    }

    return (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                    <h3 style={{ margin: 0 }}>🌐 Live Traffic Monitor</h3>
                    <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                        {data.message}
                    </p>
                </div>
                <button className="btn btn-secondary" onClick={onClose}>✕ Close</button>
            </div>

            {/* Summary Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
                <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
                        {data.summary?.total_flows || 0}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Flows</div>
                </div>
                <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#ef4444' }}>
                        {data.summary?.simulated_attacks || 0}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Attacks</div>
                </div>
                <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>
                        {data.summary?.ml_detected_anomalies || 0}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ML Detected</div>
                </div>
                <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>
                        {data.summary?.detection_accuracy || 0}%
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Accuracy</div>
                </div>
            </div>

            {/* Filter Buttons */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                {['all', 'attacks', 'normal'].map(f => (
                    <button
                        key={f}
                        className={`btn ${filter === f ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setFilter(f)}
                        style={{ textTransform: 'capitalize' }}
                    >
                        {f} ({f === 'all' ? flows.length : flows.filter(fl => f === 'attacks' ? fl.is_attack : !fl.is_attack).length})
                    </button>
                ))}
            </div>

            {/* Traffic List */}
            <div style={{ display: 'flex', gap: '1rem' }}>
                {/* Flow List */}
                <div style={{ flex: 1, maxHeight: '400px', overflowY: 'auto' }}>
                    {filteredFlows.map((flow, idx) => (
                        <div
                            key={flow.flow_id || idx}
                            onClick={() => setSelectedFlow(flow)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '1rem',
                                padding: '0.75rem',
                                marginBottom: '0.5rem',
                                background: selectedFlow?.flow_id === flow.flow_id ? 'var(--bg-hover)' : 'var(--bg-secondary)',
                                borderRadius: 'var(--radius-sm)',
                                borderLeft: `3px solid ${flow.is_attack ? getSeverityColor(flow.severity) : '#10b981'}`,
                                cursor: 'pointer',
                                transition: 'background 0.15s'
                            }}
                        >
                            {/* Status Icon */}
                            <div style={{ fontSize: '1.25rem' }}>
                                {flow.is_attack ? '🚨' : '✓'}
                            </div>

                            {/* Connection Info */}
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                        {flow.src_ip}:{flow.src_port}
                                    </span>
                                    <span style={{ color: 'var(--text-muted)' }}>→</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                        {flow.dst_ip}:{flow.dst_port}
                                    </span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <span
                                        className="badge"
                                        style={{
                                            background: `${getProtocolBadge(flow.protocol)}20`,
                                            color: getProtocolBadge(flow.protocol),
                                            fontSize: '0.65rem'
                                        }}
                                    >
                                        {flow.protocol}
                                    </span>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        {flow.service}
                                    </span>
                                </div>
                            </div>

                            {/* Attack Type */}
                            {flow.is_attack && (
                                <span
                                    className="badge"
                                    style={{
                                        background: `${getSeverityColor(flow.severity)}20`,
                                        color: getSeverityColor(flow.severity),
                                        fontSize: '0.7rem'
                                    }}
                                >
                                    {flow.attack_type}
                                </span>
                            )}

                            {/* ML Score */}
                            {flow.ml_score !== undefined && (
                                <span style={{
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: '0.8rem',
                                    color: flow.ml_detected ? '#ef4444' : '#10b981'
                                }}>
                                    {flow.ml_score.toFixed(2)}
                                </span>
                            )}
                        </div>
                    ))}
                </div>

                {/* Detail Panel */}
                {selectedFlow && (
                    <div style={{
                        width: '350px',
                        padding: '1rem',
                        background: 'var(--bg-secondary)',
                        borderRadius: 'var(--radius-md)',
                        maxHeight: '400px',
                        overflowY: 'auto'
                    }}>
                        <h4 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            {selectedFlow.is_attack ? '🚨' : '✓'} Flow Details
                        </h4>

                        {/* Connection */}
                        <div style={{ marginBottom: '1rem' }}>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Connection</div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                                {selectedFlow.src_ip}:{selectedFlow.src_port} → {selectedFlow.dst_ip}:{selectedFlow.dst_port}
                            </div>
                        </div>

                        {/* Protocol & Service */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '1rem' }}>
                            <div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Protocol</div>
                                <div>{selectedFlow.protocol}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Service</div>
                                <div>{selectedFlow.service}</div>
                            </div>
                        </div>

                        {/* Flow Metrics */}
                        <div style={{ marginBottom: '1rem' }}>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Flow Metrics</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
                                <div>Duration: <span style={{ color: 'var(--accent-primary)' }}>{(selectedFlow.flow_duration / 1000).toFixed(2)}ms</span></div>
                                <div>Packets: <span style={{ color: 'var(--accent-primary)' }}>{selectedFlow.total_fwd_packets + selectedFlow.total_bwd_packets}</span></div>
                                <div>Bytes/s: <span style={{ color: 'var(--accent-primary)' }}>{selectedFlow.flow_bytes_per_sec.toFixed(0)}</span></div>
                                <div>SYN Flags: <span style={{ color: 'var(--accent-primary)' }}>{selectedFlow.syn_flag_count}</span></div>
                            </div>
                        </div>

                        {/* Attack Info */}
                        {selectedFlow.is_attack && selectedFlow.attack_info && (
                            <div style={{
                                padding: '1rem',
                                background: `${getSeverityColor(selectedFlow.severity)}10`,
                                borderRadius: 'var(--radius-sm)',
                                marginBottom: '1rem'
                            }}>
                                <div style={{
                                    fontSize: '0.85rem',
                                    fontWeight: 600,
                                    color: getSeverityColor(selectedFlow.severity),
                                    marginBottom: '0.5rem'
                                }}>
                                    ⚠️ {selectedFlow.attack_type}
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                                    {selectedFlow.attack_info.description}
                                </div>

                                <div style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
                                    <strong>Indicators:</strong>
                                </div>
                                <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                    {selectedFlow.attack_info.indicators?.map((ind, i) => (
                                        <li key={i}>{ind}</li>
                                    ))}
                                </ul>

                                <div style={{ fontSize: '0.75rem', marginTop: '0.75rem' }}>
                                    <strong>Mitigation:</strong>
                                    <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                        {selectedFlow.attack_info.mitigation}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ML Detection Result */}
                        <div style={{
                            padding: '0.75rem',
                            background: selectedFlow.ml_detected ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                            borderRadius: 'var(--radius-sm)'
                        }}>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>ML Detection</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{
                                    fontWeight: 600,
                                    color: selectedFlow.ml_detected ? '#ef4444' : '#10b981'
                                }}>
                                    {selectedFlow.ml_detected ? '⚠️ Anomaly Detected' : '✓ Normal Traffic'}
                                </span>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                    Score: {selectedFlow.ml_score?.toFixed(4) || 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
