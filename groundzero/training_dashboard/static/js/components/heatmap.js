// static/js/components/heatmap.js
export class SearchHeatmap {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    }

    render(heatmapData) {
        if (!this.container) return;
        this.container.innerHTML = "";
        
        const data = heatmapData || {};
        
        // Render 8x8 grid from Rank 8 down to 1
        for (let r = 8; r >= 1; r--) {
            for (let f = 0; f < 8; f++) {
                const sq = this.files[f] + r;
                const cell = document.createElement("div");
                cell.className = "h-cell";
                
                // Scale intensity based on your MCTS visit distribution
                const val = data[sq] || 0;
                
                // Using a "Laboratory Red" for the heat intensity
                cell.style.background = `rgba(255, 70, 0, ${Math.min(0.85, val * 1.2)})`; 
                cell.style.border = "0.5px solid #f0f0f0";
                
                this.container.appendChild(cell);
            }
        }
    }
}