export class DepthChart {
    constructor(id) {
        const ctx = document.getElementById(id).getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Depth',
                    data: [],
                    backgroundColor: '#eeeeee',
                    hoverBackgroundColor: '#111',
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: { 
                        beginAtZero: true,
                        max: 30, // Adjust based on your typical MCTS depth
                        grid: { color: '#f5f5f5' },
                        ticks: { font: { size: 9 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    push(val) {
        this.chart.data.labels.push("");
        this.chart.data.datasets[0].data.push(val);

        if (this.chart.data.labels.length > 60) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
        }
        this.chart.update();
    }

    reset() {
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.update();
    }
}