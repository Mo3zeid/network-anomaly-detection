import { useEffect, useRef, useState } from 'react'
import { Chart, registerables } from 'chart.js'

// Register Chart.js components
if (typeof window !== 'undefined') {
    Chart.register(...registerables)
}

export default function TrafficChart() {
    const chartRef = useRef(null)
    const chartInstance = useRef(null)
    const [trafficData, setTrafficData] = useState({
        labels: [],
        normal: [],
        anomaly: []
    })

    // Generate initial demo data
    useEffect(() => {
        const labels = []
        const normal = []
        const anomaly = []

        const now = new Date()
        for (let i = 11; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 5 * 60000)
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
            normal.push(Math.floor(Math.random() * 50) + 30)
            anomaly.push(Math.floor(Math.random() * 10))
        }

        setTrafficData({ labels, normal, anomaly })
    }, [])

    // Create/update chart
    useEffect(() => {
        if (!chartRef.current || trafficData.labels.length === 0) return

        // Destroy existing chart
        if (chartInstance.current) {
            chartInstance.current.destroy()
        }

        const ctx = chartRef.current.getContext('2d')

        chartInstance.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trafficData.labels,
                datasets: [
                    {
                        label: 'Normal Traffic',
                        data: trafficData.normal,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: '#00d4ff',
                    },
                    {
                        label: 'Anomalies',
                        data: trafficData.anomaly,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: '#ef4444',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            color: '#94a3b8',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1a2234',
                        titleColor: '#f0f4f8',
                        bodyColor: '#94a3b8',
                        borderColor: '#2a3548',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function (context) {
                                return `${context.dataset.label}: ${context.parsed.y} flows`
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(42, 53, 72, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(42, 53, 72, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b'
                        },
                        beginAtZero: true
                    }
                }
            }
        })

        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy()
            }
        }
    }, [trafficData])

    return (
        <div className="chart-container">
            <canvas ref={chartRef}></canvas>
        </div>
    )
}
