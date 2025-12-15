export default function AlertsPanel({ alerts = [], onRefresh }) {
    const formatTime = (timestamp) => {
        const date = new Date(timestamp)
        return date.toLocaleTimeString()
    }

    const getSeverityClass = (severity) => {
        return `alert-item severity-${severity}`
    }

    const getSeverityIcon = (severity) => {
        switch (severity) {
            case 'critical': return '🔴'
            case 'high': return '🟠'
            case 'medium': return '🟡'
            default: return '🔵'
        }
    }

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">Recent Alerts</span>
                <button className="btn btn-secondary" onClick={onRefresh}>
                    🔄 Refresh
                </button>
            </div>

            {alerts.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>✅</div>
                    <p>No alerts detected</p>
                    <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>The system is running normally</p>
                </div>
            ) : (
                <div className="alerts-list">
                    {alerts.map((alert, index) => (
                        <div key={alert.id || index} className={getSeverityClass(alert.severity)}>
                            <div className="alert-icon">
                                {getSeverityIcon(alert.severity)}
                            </div>
                            <div className="alert-content">
                                <div className="alert-type">{alert.attack_type}</div>
                                <div className="alert-time">{formatTime(alert.timestamp)}</div>
                            </div>
                            <div className="alert-score">
                                Score: {alert.score.toFixed(2)}
                            </div>
                            <span className={`badge ${alert.is_acknowledged ? 'badge-success' : 'badge-danger'}`}>
                                {alert.is_acknowledged ? 'Acknowledged' : 'New'}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
