const sess = new onnx.InferenceSession();
const model_loading = sess.loadModel("./handwritten_flatten.onnx");

const CANVAS_SIZE = 300;
const INPUT_SIZE = 28;
const PIXEL_SIZE = CANVAS_SIZE / INPUT_SIZE;

const canvas = document.getElementById("canvas");
const clr_btn = document.getElementById("clr-btn");
const preds_container = document.getElementById("preds-container");
const softmax_checkbox = document.getElementById("softmax-checkbox");

canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;
const ctx = canvas.getContext("2d");
let is_mouse_down = false;

let input = new Array(INPUT_SIZE * INPUT_SIZE).fill(0);  

function clearCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = "lightgray";
    for (let i = 0; i < INPUT_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * PIXEL_SIZE, 0);
        ctx.lineTo(i * PIXEL_SIZE, CANVAS_SIZE);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * PIXEL_SIZE);
        ctx.lineTo(CANVAS_SIZE, i * PIXEL_SIZE);
        ctx.stroke();
    }
    input = new Array(INPUT_SIZE * INPUT_SIZE).fill(0);
    updateBars(new Array(10).fill(0));
}

function drawAt(x, y) {
    ctx.fillStyle = "#000000";
    ctx.fillRect(x, y, PIXEL_SIZE, PIXEL_SIZE);
    let i = Math.floor(y / PIXEL_SIZE);
    let j = Math.floor(x / PIXEL_SIZE);
    i = Math.min(Math.max(i, 0), INPUT_SIZE - 1);
    j = Math.min(Math.max(j, 0), INPUT_SIZE - 1);
    input[i * INPUT_SIZE + j] = 1;
    predict(input);
}

function updateBars(logit) {
    preds_container.innerHTML = "";
    let max = 0;
    let max_i = 0;

    for (let i=0; i<logit.length; i++) {
        const pred_label = document.createElement("p");
        pred_label.classList.add("pred-label");
        pred_label.textContent = i;

        const pred_bar = document.createElement("div");
        pred_bar.classList.add("pred-bar");
        pred_bar.style.width = `${logit[i] * 100}%`;

        const pred = document.createElement("div");
        pred.classList.add("pred");
        pred.appendChild(pred_label);
        pred.appendChild(pred_bar);

        preds_container.appendChild(pred);

        if (logit[i] > max) {
            max = logit[i];
            max_i = i;
        }
    }

    preds_container.innerHTML += `<p class="pred-label">Prediction: ${max_i}</p>`;
}

function predict(input) {
    const input_tensor = new onnx.Tensor(new Float32Array(input), "float32", [INPUT_SIZE * INPUT_SIZE]);
    sess.run([input_tensor]).then((output) => {
        const result = output.values().next().value.data;
        updateBars(result);
    });
}

function onDraw(e) {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const i = Math.floor(x / PIXEL_SIZE);
    const j = Math.floor(y / PIXEL_SIZE);

    drawAt(i * PIXEL_SIZE, j * PIXEL_SIZE);
}

function onMouseDown(e) {
    is_mouse_down = true;
}

function onMouseUp(e) {
    is_mouse_down = false;
}

model_loading.then(() => {
    canvas.addEventListener("click", onDraw);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseUp);
    canvas.addEventListener("mousemove", (e) => {
        if (!is_mouse_down) return;
        onDraw(e);
    });
    clearCanvas();
    clr_btn.addEventListener("click", clearCanvas);
});