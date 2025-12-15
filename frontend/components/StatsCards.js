export default function StatsCards({ stats }) {
    const defaultStats = {
        total_flows_analyzed: 0,
        total_anomalies_detected: 0,
        detection_rate: 0,
        uptime_seconds: 0
    }

    const data = stats || defaultStats

    // Format uptime
    const formatUptime = (seconds) => {
        const hours = Math.floor(seconds / 3600)
        const mins = Math.floor((seconds % 3600) / 60)
        return `${hours}h ${mins}m`
    }

    return (
        <div className="stats-grid">
            {/* Total Flows */}
            <div className="stat-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <div className="stat-value">{data.total_flows_analyzed.toLocaleString()}</div>
                        <div className="stat-label">Total Flows Analyzed</div>
                    </div>
                    <div style={{ fontSize: '2rem', opacity: 0.3 }}>📊</div>
                </div>
            </div>

            {/* Anomalies Detected */}
            <div className="stat-card danger">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <div className="stat-value" style={{ background: 'linear-gradient(135deg, #ef4444, #f59e0b)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            {data.total_anomalies_detected.toLocaleString()}
                        </div>
                        <div className="stat-label">Anomalies Detected</div>
                    </div>
                    <div style={{ fontSize: '2rem', opacity: 0.3 }}>🚨</div>
                </div>
            </div>

            {/* Detection Rate */}
            <div className="stat-card warning">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <div className="stat-value" style={{ background: 'linear-gradient(135deg, #f59e0b, #10b981)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            {(data.detection_rate * 100).toFixed(1)}%
                        </div>
                        <div className="stat-label">Detection Rate</div>
                    </div>
                    <div style={{ fontSize: '2rem', opacity: 0.3 }}>📈</div>
                </div>
            </div>

            {/* Uptime */}
            <div className="stat-card success">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <div className="stat-value" style={{ background: 'linear-gradient(135deg, #10b981, #00d4ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            {formatUptime(data.uptime_seconds)}
                        </div>
                        <div className="stat-label">System Uptime</div>
                    </div>
                    <div style={{ fontSize: '2rem', opacity: 0.3 }}>⏱️</div>
                </div>
            </div>
        </div>
    )
}
