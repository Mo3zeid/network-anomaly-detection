import { useEffect, useRef } from 'react'
import { Chart, registerables } from 'chart.js'

// Register Chart.js components
if (typeof window !== 'undefined') {
    Chart.register(...registerables)
}

export default function AttackDistribution({ stats }) {
    const chartRef = useRef(null)
    const chartInstance = useRef(null)

    useEffect(() => {
        if (!chartRef.current) return

        // Destroy existing chart
        if (chartInstance.current) {
            chartInstance.current.destroy()
        }

        // Get attack distribution data
        const distribution = stats?.attack_distribution || {}
        const labels = Object.keys(distribution)
        const data = Object.values(distribution)

        // Use demo data if no real data
        const demoLabels = labels.length > 0 ? labels : ['DDoS', 'Port Scan', 'Web Attack', 'Brute Force', 'Normal']
        const demoData = data.length > 0 ? data : [35, 25, 20, 15, 5]

        const colors = [
            '#ef4444',  // Red
            '#f59e0b',  // Orange
            '#8b5cf6',  // Purple
            '#3b82f6',  // Blue
            '#10b981',  // Green
            '#ec4899',  // Pink
            '#06b6d4',  // Cyan
        ]

        const ctx = chartRef.current.getContext('2d')

        chartInstance.current = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: demoLabels,
                datasets: [{
                    data: demoData,
                    backgroundColor: colors.slice(0, demoLabels.length),
                    borderColor: '#1a2234',
                    borderWidth: 3,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            usePointStyle: true,
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1a2234',
                        titleColor: '#f0f4f8',
                        bodyColor: '#94a3b8',
                        borderColor: '#2a3548',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function (context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0)
                                const percentage = ((context.raw / total) * 100).toFixed(1)
                                return `${context.label}: ${context.raw} (${percentage}%)`
                            }
                        }
                    }
                }
            }
        })

        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy()
            }
        }
    }, [stats])

    return (
        <div className="chart-container" style={{ height: '280px' }}>
            <canvas ref={chartRef}></canvas>
        </div>
    )
}
