const API_URL = "http://localhost:8000";

// --- DOM Elements ---
const inputs = {
    tv: document.getElementById('tvBudget'),
    radio: document.getElementById('radioBudget'),
    social: document.getElementById('socialBudget')
};

const displays = {
    tv: document.getElementById('tvValue'),
    radio: document.getElementById('radioValue'),
    social: document.getElementById('socialValue')
};

const outputs = {
    sales: document.getElementById('salesValue'),
    roi: document.getElementById('roiValue'),
    perf: document.getElementById('perfValue'),
    baseRoi: document.getElementById('baseRoi')
};

const shapUI = {
    tv: { fill: document.getElementById('shapFillTv'), val: document.getElementById('shapValTv') },
    radio: { fill: document.getElementById('shapFillRadio'), val: document.getElementById('shapValRadio') },
    social: { fill: document.getElementById('shapFillSocial'), val: document.getElementById('shapValSocial') },
    perf: { fill: document.getElementById('shapFillPerf'), val: document.getElementById('shapValPerf') }
};

const btnSimulate = document.getElementById('btnSimulate');

// --- Global Chart Setup ---
let variationChart;
let simulationCount = 0;
const chartLabels = [];
const roiData = [];
const salesData = [];

function initChart() {
    const ctx = document.getElementById('variationChart').getContext('2d');
    variationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: 'ROI Estimé (%)',
                    data: roiData,
                    borderColor: '#6a5acd',
                    backgroundColor: 'rgba(106, 90, 205, 0.2)',
                    yAxisID: 'y-roi',
                    tension: 0.3,
                    fill: true,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#6a5acd',
                    pointRadius: 4,
                    borderWidth: 2
                },
                {
                    label: 'Ventes Estimées (K€)',
                    data: salesData,
                    borderColor: '#4a90e2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    yAxisID: 'y-sales',
                    tension: 0.3,
                    fill: false,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#4a90e2',
                    pointRadius: 4,
                    borderWidth: 2,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#9ba1a6', font: { family: 'Outfit' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ba1a6', font: { family: 'Outfit' } }
                },
                'y-roi': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'ROI (%)', color: '#6a5acd', font: { family: 'Outfit' } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#6a5acd', font: { family: 'Outfit' } }
                },
                'y-sales': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Ventes (K€)', color: '#4a90e2', font: { family: 'Outfit' } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#4a90e2', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

function addDataToChart(roi, sales) {
    simulationCount++;
    chartLabels.push(`Sim ${simulationCount}`);
    roiData.push(roi);
    salesData.push(sales);
    
    // Garder les 10 dernières simulations pour que le graphe reste lisible
    if (chartLabels.length > 10) {
        chartLabels.shift();
        roiData.shift();
        salesData.shift();
    }
    variationChart.update();
}

// --- Event Listeners ---
Object.keys(inputs).forEach(key => {
    inputs[key].addEventListener('input', (e) => {
        displays[key].textContent = parseFloat(e.target.value).toFixed(1);
    });
});

btnSimulate.addEventListener('click', runSimulation);

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    runSimulation();
});

// --- Core Logic ---

function getPayloadAndInvestment() {
    const tvBudget = parseFloat(inputs.tv.value);
    const radioBudget = parseFloat(inputs.radio.value);
    const socialBudget = parseFloat(inputs.social.value);
    const totalInvestment = tvBudget + radioBudget + socialBudget;
    
    const payload = {
        "TV": tvBudget,
        "Radio": radioBudget,
        "Social Media": socialBudget
    };
    return { payload, totalInvestment };
}

async function runSimulation() {
    setLoadingState(btnSimulate, true);
    const { payload, totalInvestment } = getPayloadAndInvestment();

    try {
        const [perfRes, shapRes] = await Promise.all([
            fetch(`${API_URL}/predict/performance`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }),
            fetch(`${API_URL}/predict/shap_impact`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
        ]);

        if (!perfRes.ok || !shapRes.ok) throw new Error("Erreur API.");

        const perfData = await perfRes.json();
        const shapData = await shapRes.json();

        updateUI(perfData, shapData, totalInvestment);
        
        const expectedSales = totalInvestment > 0 ? totalInvestment * (shapData.Predicted_ROI / 100 + 1) : 0;
        addDataToChart(shapData.Predicted_ROI, expectedSales);

    } catch (error) {
        console.error(error);
    } finally {
        setLoadingState(btnSimulate, false, "Simuler & Analyser");
    }
}

// --- UI Updates ---

function updateUI(perfData, shapData, totalInvestment) {
    // CORRECTION : utilise Performance_Segment (segmentation non supervisée)
    outputs.perf.textContent = perfData.Performance_Segment;

    const roiPredicted = shapData.Predicted_ROI;
    animateValue(outputs.roi, parseFloat(outputs.roi.textContent) || 0, roiPredicted, 1000);
    outputs.baseRoi.textContent = `${shapData.Base_ROI_Average}%`;

    const expectedSales = totalInvestment > 0 ? totalInvestment * (roiPredicted / 100 + 1) : 0;
    animateValue(outputs.sales, parseFloat(outputs.sales.textContent) || 0, expectedSales, 1000);

    const maxShapRange = 40; 
    updateShapRow(shapUI.tv, shapData.SHAP_Impact_Breakdown["TV"], maxShapRange);
    updateShapRow(shapUI.radio, shapData.SHAP_Impact_Breakdown["Radio"], maxShapRange);
    updateShapRow(shapUI.social, shapData.SHAP_Impact_Breakdown["Social Media"], maxShapRange);
    // CORRECTION : utilise Segment_Cluster (cohérent avec la clé renvoyée par l'API)
    updateShapRow(shapUI.perf, shapData.SHAP_Impact_Breakdown["Segment_Cluster"], maxShapRange);
}

function updateShapRow(uiElements, value, maxRange) {
    const isPositive = value >= 0;
    const sign = isPositive ? "+" : "";
    uiElements.val.textContent = `${sign}${value.toFixed(1)}%`;
    uiElements.val.className = `shap-value ${isPositive ? 'pos' : 'neg'}`;

    let widthPercent = (Math.abs(value) / maxRange) * 50;
    if (widthPercent > 50) widthPercent = 50; 

    uiElements.fill.className = `shap-fill ${isPositive ? 'shap-positive' : 'shap-negative'}`;
    uiElements.fill.style.width = `${widthPercent}%`;
    
    if (isPositive) {
        uiElements.fill.style.left = "50%";
    } else {
        uiElements.fill.style.left = `${50 - widthPercent}%`;
    }
}

function setLoadingState(btn, isLoading, originalText = "") {
    if (isLoading) {
        btn.textContent = "Calcul...";
        btn.style.opacity = "0.7";
    } else {
        btn.textContent = originalText;
        btn.style.opacity = "1";
    }
}

// Simple counter animation
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = start + progress * (end - start);
        obj.innerHTML = current.toFixed(1);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
