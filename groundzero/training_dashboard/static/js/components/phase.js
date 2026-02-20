export class PhaseChart {
    constructor(id) {
        const ctx = document.getElementById(id).getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Opening (1-20)', 'Midgame (21-40)', 'Endgame (41+)'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#f5f5f5', '#eeeeee', '#111111'],
                    borderWidth: 1,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: { legend: { display: false } }
            }
        });
    }

    update(phaseData) {
        if (!phaseData) return;
        this.chart.data.datasets[0].data = [
            phaseData.opening, 
            phaseData.midgame, 
            phaseData.endgame
        ];
        this.chart.update();
    }
}