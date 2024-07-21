const sess = new onnx.InferenceSession();
const model_loading = sess.loadModel("./handwritten_flatten.onnx");

const CANVAS_SIZE = 350;
const INPUT_SIZE = 28;
const PIXEL_SIZE = CANVAS_SIZE / INPUT_SIZE;

const canvas = document.getElementById("canvas");
const clr_btn = document.getElementById("clr-btn");

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
}

function drawAt(x, y) {
    ctx.fillStyle = "#000000";
    ctx.fillRect(x, y, PIXEL_SIZE, PIXEL_SIZE);
    let i = Math.floor(y / PIXEL_SIZE);
    let j = Math.floor(x / PIXEL_SIZE);
    input[i * INPUT_SIZE + j] = 1;
    predict(input);
}

function predict(input) {
    const input_tensor = new onnx.Tensor(new Float32Array(input), "float32", [INPUT_SIZE * INPUT_SIZE]);
    sess.run([input_tensor]).then((output) => {
        const result = output.values().next().value.data;
        const max = Math.max(...result);
        const index = result.indexOf(max);
        console.log("Predicted: ", index);  
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