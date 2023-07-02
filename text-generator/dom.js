


let imageContainerElem = document.getElementById('image-container');
let canvasElem = document.getElementById('canvas');
let progressbarElem = document.getElementById('progressbar');
let stepsSlideElem = document.getElementById('steps-slide');
let heightSlideElem = document.getElementById('height-slide');
let widthSlideElem = document.getElementById('width-slide');

let radioCpuElem = document.getElementById('radio-cpu');
let radioGpuElem = document.getElementById('radio-gpu');
let generateBnElem = document.getElementById('generate-bn');
let reloadModelBnElem = document.getElementById('reload-model-bn');

let stepsSlideDisplayElem = document.getElementById('steps-slide-display');
let heightSlideDisplayElem = document.getElementById('height-slide-display');
let widthSlideDisplayElem = document.getElementById('width-slide-display');

function advancedImageShow(image){
    image = tf.tensor(image);
    let [height, width] = image.shape;
    const MAX_SIZE = 300;
    if (height < width){
        height = height / width * MAX_SIZE;
        width = MAX_SIZE;
    } else {
        width = width / height * MAX_SIZE;
        height = MAX_SIZE;
    }
    image = tf.expandDims(image, 2);
    image = tf.image.resizeBilinear(image, [height, width]);
    image = tf.squeeze(image, 2);
    imshow(canvasElem, image);
}

generateBnElem.addEventListener('click', evt => {
    if (!globalObj.unetPrepared){
        alert("The model is not prepared.");
        return;
    }
    if (globalObj.isGenerating){
        return;
    }
    globalObj.isGenerating = true;
    progressbarElem.max = stepsSlideElem.value;
    progressbarElem.value = 0;
    globalObj.worker.postMessage({
        action: 'generate image',
        steps: stepsSlideElem.value,
        height: Number.parseInt(heightSlideElem.value),
        width: Number.parseInt(widthSlideElem.value),
    });
});

reloadModelBnElem.addEventListener('click', evt=>{
    globalObj.worker.postMessage({
        action: 'load model',
        path: 'text-gen-light-wo-trans.json'
    });
});

radioCpuElem.addEventListener('change', () => {
    globalObj.worker.postMessage({
        action: 'set backend',
        backend: 'cpu'
    });
});
radioGpuElem.addEventListener('change', () => {
    globalObj.worker.postMessage({
        action: 'set backend',
        backend: 'webgl'
    });
});

stepsSlideElem.addEventListener('change', () => stepsSlideDisplayElem.value = stepsSlideElem.value);
heightSlideElem.addEventListener('change', () => heightSlideDisplayElem.value = heightSlideElem.value);
widthSlideElem.addEventListener('change', () => widthSlideDisplayElem.value = widthSlideElem.value);

