class SimpleNN {
    constructor(input_dim, hidden_dim, output_dim) {
        this.input_dim = input_dim;
        this.hidden_dim = hidden_dim;
        this.output_dim = output_dim;

        // Initialize weights and biases (Random between -0.5 and 0.5)
        this.W1 = Array.from({ length: input_dim }, () =>
            Array.from({ length: hidden_dim }, () => Math.random() - 0.5));
        this.b1 = Array(hidden_dim).fill(0);
        this.W2 = Array.from({ length: hidden_dim }, () =>
            Array.from({ length: output_dim }, () => Math.random() - 0.5));
        this.b2 = Array(output_dim).fill(0);

        // Storage for activations and gradients
        this.a1 = null;
        this.y_hat = null;
        this.last_gradients = null;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoid_derivative(x) {
        return x * (1 - x);
    }

    // Matrix multiplication helper
    matmul(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const rowsB = B.length;
        const colsB = B[0].length;

        if (colsA !== rowsB) throw new Error("Shape mismatch");

        let C = Array.from({ length: rowsA }, () => Array(colsB).fill(0));

        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    forward(X) {
        // Z1 = X * W1 + b1
        let Z1 = [];
        for (let i = 0; i < X.length; i++) {
            let row = [];
            for (let j = 0; j < this.hidden_dim; j++) {
                let sum = this.b1[j];
                for (let k = 0; k < this.input_dim; k++) {
                    sum += X[i][k] * this.W1[k][j];
                }
                row.push(sum);
            }
            Z1.push(row);
        }

        // a1 = sigmoid(Z1)
        this.a1 = Z1.map(row => row.map(val => this.sigmoid(val)));

        // Z2 = a1 * W2 + b2
        let Z2 = [];
        for (let i = 0; i < this.a1.length; i++) {
            let row = [];
            for (let j = 0; j < this.output_dim; j++) {
                let sum = this.b2[j];
                for (let k = 0; k < this.hidden_dim; k++) {
                    sum += this.a1[i][k] * this.W2[k][j];
                }
                row.push(sum);
            }
            Z2.push(row);
        }

        // y_hat = sigmoid(Z2)
        this.y_hat = Z2.map(row => row.map(val => this.sigmoid(val)));
        return this.y_hat;
    }

    backward(X, y, lr) {
        const m = X.length;

        // Calculate gradients
        // dz2 = y_hat - y
        let dz2 = [];
        for (let i = 0; i < m; i++) {
            dz2.push([this.y_hat[i][0] - y[i][0]]);
        }

        // dW2 = (1/m) * a1.T * dz2
        // Transpose a1
        let a1_T = this.a1[0].map((_, colIndex) => this.a1.map(row => row[colIndex]));
        let dW2 = this.matmul(a1_T, dz2).map(row => row.map(val => val / m));

        // db2 = (1/m) * sum(dz2)
        let db2 = [dz2.reduce((acc, val) => acc + val[0], 0) / m];

        // dz1 = (dz2 * W2.T) * sigmoid_derivative(a1)
        let W2_T = this.W2[0].map((_, colIndex) => this.W2.map(row => row[colIndex]));
        let da1 = this.matmul(dz2, W2_T); // roughly

        // Element-wise multiplication for dz1
        let dz1 = [];
        for (let i = 0; i < m; i++) {
            let row = [];
            for (let j = 0; j < this.hidden_dim; j++) {
                row.push(da1[i][j] * this.sigmoid_derivative(this.a1[i][j]));
            }
            dz1.push(row);
        }

        // dW1 = (1/m) * X.T * dz1
        let X_T = X[0].map((_, colIndex) => X.map(row => row[colIndex]));
        let dW1 = this.matmul(X_T, dz1).map(row => row.map(val => val / m));

        // db1 = (1/m) * sum(dz1)
        let db1 = [];
        for (let j = 0; j < this.hidden_dim; j++) {
            let sum = 0;
            for (let i = 0; i < m; i++) {
                sum += dz1[i][j];
            }
            db1.push(sum / m);
        }

        // Store gradients for visualization
        this.last_gradients = { dW1, dW2 };

        // Update weights
        for (let i = 0; i < this.input_dim; i++) {
            for (let j = 0; j < this.hidden_dim; j++) {
                this.W1[i][j] -= lr * dW1[i][j];
            }
        }
        for (let j = 0; j < this.hidden_dim; j++) {
            this.b1[j] -= lr * db1[j];
        }
        for (let i = 0; i < this.hidden_dim; i++) {
            for (let j = 0; j < this.output_dim; j++) {
                this.W2[i][j] -= lr * dW2[i][j];
            }
        }
        for (let j = 0; j < this.output_dim; j++) {
            this.b2[j] -= lr * db2[j];
        }
    }
}

// Data: XOR
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [[0], [1], [1], [0]];

// App State
let nn = new SimpleNN(2, 4, 1);
let isTraining = false;
let epoch = 0;
let lossHistory = [];
let gradHistory = []; // { w1: val, w2: val }
let animId;
let lr = 0.1;

// UI Elements
const btnTrain = document.getElementById('btn-train');
const btnStep = document.getElementById('btn-step');
const btnReset = document.getElementById('btn-reset');
const lrSlider = document.getElementById('lr-slider');
const lrValue = document.getElementById('lr-value');
const stepInput = document.getElementById('step-input');
// Remove stepValue span update since input shows value, but user code had it. 
// Actually I didn't add the listener for step input to update text, simpler to just use input value.

const epochDisplay = document.getElementById('epoch-display');
const lossDisplay = document.getElementById('loss-display');

// Canvases
const netCanvas = document.getElementById('network-canvas');
const netCtx = netCanvas.getContext('2d');
const decCanvas = document.getElementById('decision-canvas');
const decCtx = decCanvas.getContext('2d');
const lossCanvas = document.getElementById('loss-canvas');
const lossCtx = lossCanvas.getContext('2d');
const gradCanvas = document.getElementById('grad-canvas');
const gradCtx = gradCanvas.getContext('2d');

// Event Listeners
btnTrain.addEventListener('click', () => {
    // If we are step training, stop it
    if (isStepTraining) {
        cancelAnimationFrame(animId);
        isStepTraining = false;
    }

    isTraining = !isTraining;
    btnTrain.textContent = isTraining ? "Pause Training" : "Start Training";
    if (isTraining) loop();
});

let isStepTraining = false;
let stepsRemaining = 0;

btnStep.addEventListener('click', () => {
    if (isTraining) {
        // Pause continuous training if running
        isTraining = false;
        btnTrain.textContent = "Start Training";
        cancelAnimationFrame(animId);
    }

    const steps = parseInt(stepInput.value) || 1;
    stepsRemaining = steps;
    isStepTraining = true;
    loop();
});

btnReset.addEventListener('click', () => {
    isTraining = false;
    isStepTraining = false;
    btnTrain.textContent = "Start Training";
    cancelAnimationFrame(animId);
    nn = new SimpleNN(2, 4, 1);
    epoch = 0;
    lossHistory = [];
    gradHistory = [];
    drawAll();
    epochDisplay.textContent = 0;
    lossDisplay.textContent = "0.0000";
});

lrSlider.addEventListener('input', (e) => {
    lr = parseFloat(e.target.value);
    lrValue.textContent = lr;
});

function calculateLoss() {
    const y_hat = nn.forward(X);
    let sum = 0;
    for (let i = 0; i < y.length; i++) {
        // Binary cross entropy
        const p = Math.max(Math.min(y_hat[i][0], 1 - 1e-9), 1e-9); // clamp
        sum += -(y[i][0] * Math.log(p) + (1 - y[i][0]) * Math.log(1 - p));
    }
    return sum / y.length;
}

function meanAbs(matrix) {
    let sum = 0;
    let count = 0;
    for (let row of matrix) {
        for (let val of row) {
            sum += Math.abs(val);
            count++;
        }
    }
    return sum / count;
}

function loop() {
    if (!isTraining && !isStepTraining) return;

    // Determine how many steps to run in this frame
    let stepsToRun = 50;

    if (isStepTraining) {
        if (stepsRemaining <= 0) {
            isStepTraining = false;
            cancelAnimationFrame(animId);
            return;
        }
        stepsToRun = Math.min(50, stepsRemaining);
    }

    for (let i = 0; i < stepsToRun; i++) {
        nn.forward(X);
        nn.backward(X, y, lr);
        epoch++;
        if (isStepTraining) stepsRemaining--;
    }

    const loss = calculateLoss();
    if (epoch % 10 === 0 || isStepTraining) {
        lossHistory.push(loss);
        if (lossHistory.length > 300) lossHistory.shift();

        // Track gradients
        if (nn.last_gradients) {
            const mg1 = meanAbs(nn.last_gradients.dW1);
            const mg2 = meanAbs(nn.last_gradients.dW2);
            gradHistory.push({ w1: mg1, w2: mg2 });
            if (gradHistory.length > 300) gradHistory.shift();
        }
    }

    epochDisplay.textContent = epoch;
    lossDisplay.textContent = loss.toFixed(4);

    drawAll();

    if (isTraining || (isStepTraining && stepsRemaining > 0)) {
        animId = requestAnimationFrame(loop);
    } else {
        cancelAnimationFrame(animId);
    }
}

// --------------------------------------------------------
// Visualization
// --------------------------------------------------------

function drawAll() {
    drawNetwork();
    drawDecisionBoundary();
    drawLoss();
    drawGradients();
}

function drawGradients() {
    const w = gradCanvas.width;
    const h = gradCanvas.height;
    gradCtx.clearRect(0, 0, w, h);

    if (gradHistory.length < 2) return;

    // We plot two lines: W1 grad (Purple), W2 grad (Yellow)
    const maxG = gradHistory.reduce((acc, curr) => Math.max(acc, curr.w1, curr.w2), 0.001);
    const step = w / gradHistory.length;

    // Helper to draw line
    function drawLine(color, key) {
        gradCtx.beginPath();
        gradCtx.strokeStyle = color;
        gradCtx.lineWidth = 2;
        gradCtx.moveTo(0, h - (gradHistory[0][key] / maxG) * (h - 20));

        for (let i = 1; i < gradHistory.length; i++) {
            const x = i * step;
            const y = h - (gradHistory[i][key] / maxG) * (h - 20);
            gradCtx.lineTo(x, y);
        }
        gradCtx.stroke();
    }

    drawLine('#a855f7', 'w1'); // Purple
    drawLine('#eab308', 'w2'); // Yellow

    // Legend
    gradCtx.font = "10px sans-serif";
    gradCtx.fillStyle = '#a855f7';
    gradCtx.fillText("Layer 1 Grad", 10, 15);
    gradCtx.fillStyle = '#eab308';
    gradCtx.fillText("Layer 2 Grad", 80, 15);
}


function drawNetwork() {
    netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);

    // Adjust layout for labels
    const layerX = [60, 200, 340];
    const nodeY = {
        0: [100, 200], // Input (2 nodes)
        1: [75, 125, 175, 225], // Hidden (4 nodes)
        2: [150] // Output (1 node)
    };

    const nodeRadius = 18;
    netCtx.font = "10px monospace";
    netCtx.textAlign = "center";
    netCtx.textBaseline = "middle";

    // Draw Weights (Lines)
    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 4; j++) {
            const w = nn.W1[i][j];
            const weightAbs = Math.abs(w);

            // Line
            netCtx.beginPath();
            netCtx.moveTo(layerX[0], nodeY[0][i]);
            netCtx.lineTo(layerX[1], nodeY[1][j]);
            netCtx.lineWidth = Math.min(weightAbs * 3, 6);
            netCtx.strokeStyle = w > 0 ? '#38bdf8' : '#f43f5e';
            netCtx.globalAlpha = Math.min(weightAbs + 0.2, 1);
            netCtx.stroke();

            // Label background
            const midX = (layerX[0] + layerX[1]) / 2;
            const midY = (nodeY[0][i] + nodeY[1][j]) / 2;

            // Shift label slightly based on index to avoid overlap
            const shiftY = (j - 1.5) * 5;

            netCtx.globalAlpha = 1.0;
            netCtx.fillStyle = 'rgba(15, 23, 42, 0.8)'; // Dark bg
            netCtx.fillRect(midX - 12, midY + shiftY - 6, 24, 12);

            // Label text
            netCtx.fillStyle = '#e2e8f0';
            netCtx.fillText(w.toFixed(2), midX, midY + shiftY);
        }
    }

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 1; j++) {
            const w = nn.W2[i][j];
            const weightAbs = Math.abs(w);

            // Line
            netCtx.beginPath();
            netCtx.moveTo(layerX[1], nodeY[1][i]);
            netCtx.lineTo(layerX[2], nodeY[2][j]);
            netCtx.lineWidth = Math.min(weightAbs * 3, 6);
            netCtx.strokeStyle = w > 0 ? '#38bdf8' : '#f43f5e';
            netCtx.globalAlpha = Math.min(weightAbs + 0.2, 1);
            netCtx.stroke();

            // Label
            const midX = (layerX[1] + layerX[2]) / 2;
            const midY = (nodeY[1][i] + nodeY[2][j]) / 2;

            netCtx.globalAlpha = 1.0;
            netCtx.fillStyle = 'rgba(15, 23, 42, 0.8)';
            netCtx.fillRect(midX - 12, midY - 6, 24, 12);

            netCtx.fillStyle = '#e2e8f0';
            netCtx.fillText(w.toFixed(2), midX, midY);
        }
    }
    netCtx.globalAlpha = 1.0;

    // Draw Nodes
    function drawNode(x, y, label, bias = null) {
        netCtx.beginPath();
        netCtx.arc(x, y, nodeRadius, 0, Math.PI * 2);
        netCtx.fillStyle = '#1e293b';
        netCtx.fill();
        netCtx.strokeStyle = '#94a3b8';
        netCtx.lineWidth = 2;
        netCtx.stroke();

        // Node Label (e.g., x1)
        netCtx.fillStyle = '#fff';
        netCtx.font = "bold 12px sans-serif";
        netCtx.fillText(label, x, y);

        // Bias Label
        if (bias !== null) {
            netCtx.font = "10px monospace";
            netCtx.fillStyle = '#fbbf24'; // Amber for bias
            netCtx.fillText(`b:${bias.toFixed(2)}`, x, y - nodeRadius - 8);
        }
    }

    // Inputs
    drawNode(layerX[0], nodeY[0][0], "x1");
    drawNode(layerX[0], nodeY[0][1], "x2");

    // Hidden (with biases)
    for (let i = 0; i < 4; i++) {
        drawNode(layerX[1], nodeY[1][i], "h" + (i + 1), nn.b1[i]);
    }

    // Output (with bias)
    drawNode(layerX[2], nodeY[2][0], "y", nn.b2[0]);
}

function drawDecisionBoundary() {
    // Generate heatmap
    const w = decCanvas.width;
    const h = decCanvas.height;
    const imageData = decCtx.createImageData(w, h);
    const data = imageData.data;

    // We do a coarse grid to save perf if needed, but 300x300 is fine for modern browsers
    // To speed up: compute prediction for every pixel is heavy. 
    // Optimization: resolution scale.
    const res = 4; // 1 = full res, 4 = lower res

    for (let y = 0; y < h; y += res) {
        for (let x = 0; x < w; x += res) {
            // Normalize to 0-1 range
            let nx = x / w;
            let ny = y / h; // Invert y usually? simple_NN didn't specify, assume 0,0 is top-left

            // Forward pass for this point
            // We need a quick forward pass without creating object overhead if possible
            // Reusing logic:

            // Layer 1
            let h_acts = [];
            for (let j = 0; j < 4; j++) {
                let sum = nn.b1[j] + nx * nn.W1[0][j] + ny * nn.W1[1][j];
                h_acts.push(1 / (1 + Math.exp(-sum)));
            }

            // Layer 2
            let sum = nn.b2[0];
            for (let j = 0; j < 4; j++) {
                sum += h_acts[j] * nn.W2[j][0];
            }
            let out = 1 / (1 + Math.exp(-sum));

            // Color: Blue (0) -> Purple -> Pink (1)
            // Lets do map 0 -> Black, 1 -> White/Blue
            const val = Math.floor(out * 255);

            // Fill block
            for (let dy = 0; dy < res; dy++) {
                for (let dx = 0; dx < res; dx++) {
                    if (y + dy >= h || x + dx >= w) continue;
                    const idx = ((y + dy) * w + (x + dx)) * 4;
                    // Gradient: Dark Blue to Bright Cyan
                    data[idx] = 15;     // R
                    data[idx + 1] = val;  // G
                    data[idx + 2] = Math.max(100, val); // B
                    data[idx + 3] = 255;  // Alpha
                }
            }
        }
    }
    decCtx.putImageData(imageData, 0, 0);

    // Draw XOR points
    decCtx.lineWidth = 2;
    decCtx.strokeStyle = 'white';

    const points = [
        { x: 0, y: 0, val: 0 },
        { x: 0, y: 1, val: 1 },
        { x: 1, y: 0, val: 1 },
        { x: 1, y: 1, val: 0 }
    ];

    points.forEach(p => {
        // Map 0,1 to canvas coords (with some padding)
        const padding = 40;
        const cx = padding + p.x * (w - 2 * padding);
        const cy = padding + p.y * (h - 2 * padding);

        decCtx.beginPath();
        decCtx.arc(cx, cy, 8, 0, Math.PI * 2);
        decCtx.fillStyle = p.val === 1 ? '#38bdf8' : '#f43f5e'; // Blue for 1, Red for 0
        decCtx.fill();
        decCtx.stroke();
    });
}

function drawLoss() {
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);

    if (lossHistory.length < 2) return;

    lossCtx.beginPath();
    lossCtx.strokeStyle = '#f472b6'; // Pink
    lossCtx.lineWidth = 2;

    const maxLoss = Math.max(...lossHistory, 1.0); // scale height
    const step = w / lossHistory.length;

    lossCtx.moveTo(0, h - (lossHistory[0] / maxLoss) * h);

    for (let i = 1; i < lossHistory.length; i++) {
        const x = i * step;
        const y = h - (lossHistory[i] / maxLoss) * (h - 20); // padding bottom
        lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();
}

// Initial Draw
drawAll();
