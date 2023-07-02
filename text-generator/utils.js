
function load_js(src){
	let script = document.createElement("script");
	script.src = src;
	script.async = true;
	document.body.appendChild(script);
	let [ok, err] = [null, null]; 
	let promise = new Promise((resolve, reject) => {
		ok = resolve;
		err = reject;
	});
	script.onload = () => {
		ok();
	}
	return promise;
}

async function loadWeights(model, json){
	if (typeof(json) === 'string') {
		json = await fetch(json, {mode: 'no-cors'}).then(r => r.json());
	}
	let weights = json.map(e => tf.tensor(e[1], e[0]));
	model.setWeights(weights);
	return model;
}

function loadUNet(path, config){
    let unet = new UNet(config);
	let x = tf.randomNormal([1, 48, 48, 1]);
	let t = tf.tensor([999]);
	let eps = unet.apply([x, t]);
	unet.customInitializeWeights();
	return loadWeights(unet, path);
}

function imshow(canvas, tensor) {
	if(canvas === undefined){
    	canvas =  document.createElement("canvas");
		document.body.appendChild(canvas);
	}
    tensor = tf.clone(tensor);
    let min = tf.min(tensor, [0, 1], true);
    let max = tf.max(tensor, [0, 1], true);
    tensor = tensor.sub(min).div(max.sub(min));

    canvas.width = tensor.shape[0];
    canvas.height = tensor.shape[1];
    tf.browser.toPixels(tensor, canvas);
}
