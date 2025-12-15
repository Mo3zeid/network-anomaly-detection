import Link from 'next/link'
import { useRouter } from 'next/router'

export default function Sidebar({ activePage }) {
    const router = useRouter()
    const currentPath = router.pathname

    const navItems = [
        { id: 'dashboard', label: 'Dashboard', icon: '📊', href: '/' },
        { id: 'sniffer', label: 'Live Sniffer', icon: '🕷️', href: '/live-sniffer' },
        { id: 'rules', label: 'Rules', icon: '⚖️', href: '/rules' },
        { id: 'detection', label: 'Detection', icon: '🔍', href: '/detection' },
        { id: 'alerts', label: 'Alerts', icon: '🚨', href: '/alerts' },
        { id: 'analytics', label: 'Analytics', icon: '📈', href: '/analytics' },
        { id: 'settings', label: 'Settings', icon: '⚙️', href: '/settings' },
    ]

    const isActive = (item) => {
        if (activePage) return activePage === item.id
        if (item.href === '/') return currentPath === '/'
        return currentPath.startsWith(item.href)
    }

    return (
        <aside className="sidebar">
            {/* Logo */}
            <div className="logo">
                <div className="logo-icon">🛡️</div>
                <span className="logo-text">NetGuard</span>
            </div>

            {/* Navigation */}
            <nav>
                <ul className="nav-menu">
                    {navItems.map((item) => (
                        <li key={item.id}>
                            <Link href={item.href} style={{ textDecoration: 'none' }}>
                                <div className={`nav-item ${isActive(item) ? 'active' : ''}`}>
                                    <span>{item.icon}</span>
                                    <span>{item.label}</span>
                                </div>
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>

            {/* Status */}
            <div style={{ marginTop: 'auto', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                    <span className="status-dot online"></span>
                    <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>System Online</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    🧠 Two-Stage XGBoost ✓
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--success)', marginTop: '0.25rem' }}>
                    99.9% Accuracy
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    API: Connected ✓
                </div>
            </div>
        </aside>
    )
}
