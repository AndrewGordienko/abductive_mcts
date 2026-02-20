import { API } from './api.js';
import { WorkerList } from './components/worker_list.js';
import { SearchHeatmap } from './components/heatmap.js';
import { EvalChart } from './components/eval.js';
import { DepthChart } from './components/depth.js';
import { PhaseChart } from './components/phase.js';
import { LatencyChart } from './components/latency.js';
import { EntropyChart } from './components/entropy.js';

// Global UI State
let activeWorkerId = "0";
let gameHistory = [];
let historyIndex = -1;
let isLive = true;

// Setup Chessboard.js
function pieceTheme(piece) {
    return '/static/js/pieces/' + piece.toLowerCase() + '.png';
}
const board = Chessboard('board', {
    position: 'start',
    draggable: false,
    pieceTheme: pieceTheme
});

// Initialize Components
const components = {
    workers: new WorkerList('worker-list', (id) => {
        activeWorkerId = id;
        isLive = true;
        // Reset charts on worker switch for clean telemetry
        Object.values(components).forEach(c => {
            if (typeof c.reset === 'function') c.reset();
        });
    }),
    heatmap: new SearchHeatmap('heatmap-grid'),
    evalChart: new EvalChart('evalChart'),
    depthChart: new DepthChart('depthChart'),
    phaseChart: new PhaseChart('phaseChart'),
    latencyChart: new LatencyChart('latencyChart'),
    entropyChart: new EntropyChart('entropyChart')
};

// History Navigation Logic
const updateHistoryView = () => {
    if (gameHistory.length > 0 && historyIndex >= 0) {
        board.position(gameHistory[historyIndex]);
        const moveIdxEl = document.getElementById('move-idx');
        if (moveIdxEl) moveIdxEl.innerText = `MOVE: ${historyIndex}`;
    }
    const liveBtn = document.getElementById('live-view');
    if (liveBtn) liveBtn.classList.toggle('active', isLive);
};

document.getElementById('prev-move').onclick = () => {
    if (historyIndex > 0) {
        historyIndex--;
        isLive = false;
        updateHistoryView();
    }
};

document.getElementById('next-move').onclick = () => {
    if (historyIndex < gameHistory.length - 1) {
        historyIndex++;
        if (historyIndex === gameHistory.length - 1) isLive = true;
        updateHistoryView();
    }
};

document.getElementById('live-view').onclick = () => {
    isLive = true;
    historyIndex = gameHistory.length - 1;
    updateHistoryView();
};

async function sync() {
    const data = await API.getStatus();
    if (!data) return;

    // Global Stats
    const bufferCountEl = document.getElementById('buffer-count');
    if (bufferCountEl) bufferCountEl.innerText = data.buffer_count;
    
    components.workers.render(data.workers, activeWorkerId);

    const activeStats = data.workers[activeWorkerId];
    if (activeStats) {
        // 1. Handle History Data
        if (activeStats.history_fens) {
            gameHistory = activeStats.history_fens;
            if (isLive) {
                historyIndex = gameHistory.length - 1;
                board.position(activeStats.fen);
                const moveIdxEl = document.getElementById('move-idx');
                if (moveIdxEl) moveIdxEl.innerText = `MOVE: ${historyIndex}`;
            }
        }

        // 2. Update Sidebar Labels (Safe update with checks)
        const updateText = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.innerText = val !== undefined ? val : "-";
        };

        updateText('active-worker-id', activeWorkerId);
        updateText('stat-turn', activeStats.turn);
        updateText('stat-moves', activeStats.move_count);
        updateText('stat-depth', activeStats.last_depth);
        updateText('stat-games', activeStats.total_games);
        updateText('stat-samples', activeStats.total_samples);
        
        // Value conversion from [-1, 1] to [0, 100]%
        const winProb = ((activeStats.value + 1) / 2 * 100);
        updateText('win-rate', `${winProb.toFixed(1)}%`);

        // 3. Update Charts & Heatmap
        // FIX: Check for 'Thinking' to accommodate new batched status strings
        const isActive = activeStats.status.includes("Thinking") || activeStats.status.includes("Searching");
        
        if (isActive && isLive) {
            components.evalChart.push(winProb);
            components.depthChart.push(activeStats.last_depth);
            components.entropyChart.push(activeStats.entropy);
            components.latencyChart.push(activeStats.inference_ms);
            components.phaseChart.update(activeStats.phase_times);
        }
        
        if (activeStats.heatmap) {
            components.heatmap.render(activeStats.heatmap);
        }
    }
}

setInterval(sync, 1000);
sync();