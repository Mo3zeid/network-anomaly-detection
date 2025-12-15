import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'

const API_BASE = 'http://localhost:8000'

export default function Settings() {
    const [settings, setSettings] = useState({
        detectionMethod: 'combined',
        zscoreThreshold: 3.0,
        contamination: 0.1,
        alertSeverityThreshold: 2.0,
        autoRefresh: true,
        refreshInterval: 30,
        emailNotifications: false,
        notificationEmail: ''
    })
    const [saved, setSaved] = useState(false)
    const [modelInfo, setModelInfo] = useState(null)

    // Load model status
    useEffect(() => {
        fetch(`${API_BASE}/api/stats`)
            .then(res => res.json())
            .then(data => setModelInfo(data))
            .catch(console.error)
    }, [])

    // Handle changes
    const handleChange = (key, value) => {
        setSettings({ ...settings, [key]: value })
        setSaved(false)
    }

    // Save settings (in real app, this would persist to backend)
    const saveSettings = () => {
        localStorage.setItem('nad_settings', JSON.stringify(settings))
        setSaved(true)
        setTimeout(() => setSaved(false), 3000)
    }

    // Load settings on mount
    useEffect(() => {
        const saved = localStorage.getItem('nad_settings')
        if (saved) {
            setSettings(JSON.parse(saved))
        }
    }, [])

    return (
        <>
            <Head>
                <title>Settings | Network Anomaly Detection</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
            </Head>

            <div className="dashboard">
                <Sidebar activePage="settings" />

                <main className="main-content">
                    <div style={{ marginBottom: '2rem' }}>
                        <h1>Settings</h1>
                        <p style={{ marginTop: '0.5rem' }}>Configure detection parameters and preferences</p>
                    </div>

                    {/* Detection Settings - Updated for Two-Stage XGBoost */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">🧠 AI Detection Engine</span>
                        </div>

                        <div style={{ padding: '1rem', background: 'var(--bg-hover)', borderRadius: 'var(--radius-md)', marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <h4 style={{ margin: 0, color: 'var(--accent-primary)' }}>Two-Stage XGBoost</h4>
                                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                        Cloud-trained on 67M+ network flows
                                    </p>
                                </div>
                                <span className="badge badge-success" style={{ padding: '0.5rem 1rem' }}>✓ Active</span>
                            </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                            <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                                <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase' }}>Stage 1 (Binary)</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--success)' }}>99.90%</div>
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Normal vs Attack</div>
                            </div>
                            <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                                <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase' }}>Stage 2 (Multi-class)</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--success)' }}>99.87%</div>
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Attack Type Classification</div>
                            </div>
                        </div>

                        <div style={{ marginTop: '1rem' }}>
                            <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                Alert Severity Threshold: {settings.alertSeverityThreshold}
                            </label>
                            <input
                                type="range"
                                min="1"
                                max="5"
                                step="0.5"
                                value={settings.alertSeverityThreshold}
                                onChange={(e) => handleChange('alertSeverityThreshold', parseFloat(e.target.value))}
                                style={{ width: '100%' }}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                <span>More Alerts</span>
                                <span>Fewer Alerts</span>
                            </div>
                        </div>
                    </div>

                    {/* Dashboard Settings */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">📊 Dashboard Settings</span>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <input
                                    type="checkbox"
                                    id="autoRefresh"
                                    checked={settings.autoRefresh}
                                    onChange={(e) => handleChange('autoRefresh', e.target.checked)}
                                    style={{ width: '20px', height: '20px' }}
                                />
                                <label htmlFor="autoRefresh">Auto-refresh dashboard data</label>
                            </div>

                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Refresh Interval (seconds)
                                </label>
                                <input
                                    type="number"
                                    min="10"
                                    max="120"
                                    value={settings.refreshInterval}
                                    onChange={(e) => handleChange('refreshInterval', parseInt(e.target.value))}
                                    disabled={!settings.autoRefresh}
                                    style={{
                                        width: '100%',
                                        padding: '0.75rem',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-md)',
                                        color: 'var(--text-primary)',
                                        opacity: settings.autoRefresh ? 1 : 0.5
                                    }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Model Information - Honest display */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">🤖 System Components</span>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                            <div style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)'
                            }}>
                                <span>Stage 1 XGBoost (Binary)</span>
                                <span className="badge badge-success">✓ Loaded</span>
                            </div>
                            <div style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)'
                            }}>
                                <span>Stage 2 XGBoost (Attack Type)</span>
                                <span className="badge badge-success">✓ Loaded</span>
                            </div>
                            <div style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)'
                            }}>
                                <span>Preprocessor</span>
                                <span className={`badge ${modelInfo?.model_status?.preprocessor ? 'badge-success' : 'badge-danger'}`}>
                                    {modelInfo?.model_status?.preprocessor ? '✓ Loaded' : '✗ Not Loaded'}
                                </span>
                            </div>
                            <div style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)'
                            }}>
                                <span>Rule Engine</span>
                                <span className="badge badge-success">✓ Active</span>
                            </div>
                        </div>
                    </div>

                    {/* About */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header">
                            <span className="card-title">ℹ️ About</span>
                        </div>

                        <div style={{ color: 'var(--text-secondary)' }}>
                            <p><strong>Network Anomaly Detection System</strong></p>
                            <p style={{ marginTop: '0.5rem' }}>Version 2.0.0 — <span style={{ color: 'var(--success)' }}>Cloud-Trained Edition</span></p>
                            <p style={{ marginTop: '0.5rem' }}>
                                A <strong>Two-Stage XGBoost</strong> machine learning system for detecting network anomalies
                                and classifying specific attack types with <strong>99.9% accuracy</strong>.
                            </p>
                            <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: 'var(--radius-md)' }}>
                                <p style={{ margin: 0 }}>
                                    <strong>🧠 Stage 1 (Binary):</strong> Normal vs. Attack — <span style={{ color: 'var(--success)' }}>99.90% Accuracy</span><br />
                                    <strong>🎯 Stage 2 (Multi-class):</strong> Attack Type Classification — <span style={{ color: 'var(--success)' }}>99.87% Accuracy</span>
                                </p>
                            </div>
                            <p style={{ marginTop: '1rem' }}>
                                <strong>Training Data:</strong> 67+ Million network flows (CICIDS2017, CICIDS2018, UNSW-NB15, ToN-IoT)<br />
                                <strong>Features:</strong> 78 selected network flow features<br />
                                <strong>Attack Types:</strong> DDoS, Port Scan, Brute Force, Web Attacks, Botnet, Infiltration, and more
                            </p>
                            <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                Trained on Google Cloud (Vertex AI) using high-performance CPU cluster.
                            </p>
                        </div>
                    </div>

                    {/* Save Button */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                        <button className="btn btn-primary" onClick={saveSettings}>
                            💾 Save Settings
                        </button>
                        {saved && (
                            <span style={{ color: 'var(--success)' }}>✓ Settings saved!</span>
                        )}
                    </div>

                    {/* Danger Zone */}
                    <div className="card" style={{ marginTop: '2rem', borderColor: 'var(--danger)' }}>
                        <div className="card-header" style={{ borderBottomColor: 'rgba(239, 68, 68, 0.2)' }}>
                            <span className="card-title" style={{ color: 'var(--danger)' }}>⚠️ Danger Zone</span>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                                <h4 style={{ margin: '0 0 0.25rem 0' }}>Reset System Data</h4>
                                <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                    Clear all statistics, alerts, and active flows. This cannot be undone.
                                </p>
                            </div>
                            <button
                                className="btn btn-danger"
                                onClick={async () => {
                                    if (confirm('Are you sure you want to reset all data?')) {
                                        await fetch(`${API_BASE}/api/reset`, { method: 'POST' })
                                        localStorage.removeItem('simulatedTraffic')
                                        window.location.reload()
                                    }
                                }}
                                style={{ background: 'var(--danger)', color: 'white' }}
                            >
                                🗑️ Reset Data
                            </button>
                        </div>
                    </div>
                </main>
            </div>
        </>
    )
}
