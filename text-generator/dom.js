


let imageContainerElem = document.getElementById('image-container');
let canvasElem = document.getElementById('canvas');
let progressbarElem = document.getElementById('progressbar');
let stepsSlideElem = document.getElementById('steps-slide');
let heightSlideElem = document.getElementById('height-slide');
let widthSlideElem = document.getElementById('width-slide');
let generateBnElem = document.getElementById('generate-bn');

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
    progressbarElem.max = stepsSlideElem.value;
    progressbarElem.value = 0;
    globalObj.worker.postMessage({
        action: 'generate image',
        steps: stepsSlideElem.value,
        height: Number.parseInt(heightSlideElem.value),
        width: Number.parseInt(widthSlideElem.value),
    });
});

stepsSlideElem.addEventListener('change', () => stepsSlideDisplayElem.value = stepsSlideElem.value);
heightSlideElem.addEventListener('change', () => heightSlideDisplayElem.value = heightSlideElem.value);
widthSlideElem.addEventListener('change', () => widthSlideDisplayElem.value = widthSlideElem.value);

