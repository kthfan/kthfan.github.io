

if(document.readyState === "complete" || document.readyState === "interactive") {
	setTimeout(main, 1);
}else{
    document.addEventListener("DOMContentLoaded", main);
}

let globalObj = {};


async function main(){
	Promise.all([
		load_js("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js")
	])
	.then(() => load_js("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tfjs-vis.umd.min.js"))
	.then(() => load_js("modules.js"))
	.then(() => load_js("sampling.js"))
	.then(() => {
		// tf.setBackend('wasm');
		tf.setBackend('cpu');
		// tf.setBackend('webgl');

		
		// let x = tf.range(0, 1000);
		// let t = timestep_embedding(x, 512);
		// imshow(t);
		// tfvis.render.heatmap({ name: 'Heatmap', tab: 'Charts' }, {values: t.transpose().arraySync()});

		// let qkv = tf.randomNormal([1, 28, 28, 10*8*3]);
		// console.log(qkv_attention(qkv).shape)
		let lightConfig = {
			base_channels: 32,
			out_channels: 1,
			channels_mult: [1, 1, 2, 4],
			num_blocks: [2, 2, 2, 2],
			use_attentions: [false, true, true, false, false]
		};
		let lightwoTransConfig = {
			base_channels: 32,
			out_channels: 1,
			channels_mult: [1, 1, 2, 4],
			num_blocks: [2, 2, 2, 2],
			use_attentions: [false, false, false, false, false]
		};

		// let unet = new UNet(lightwoTransConfig);
		// let x = tf.input({shape: [48, 48, 1]});
		// let t = tf.input({shape: []});
		// let x = tf.randomNormal([1, 48, 48, 1]);
		// let t = tf.tensor([999]);
		// let eps = unet.apply([x, t]);
		// unet.customInitializeWeights();

		// window.unet = unet;
		
		// let unet_model = tf.model({
		// 	inputs: [x, t],
		// 	outputs: eps
		// });
		// window.unet_model = unet_model;
		// return loadWeights(unet, 'text-gen-light-wo-trans.json');

		
		
		// let x = tf.randomNormal([1, 32, 32, 1]);
		// let t = tf.tensor([1]);
		// let x_hat = unet.apply([x, t]);
	}).then(() => {
		// let x = tf.randomNormal([1, 48, 48, 1]);
		// let t = tf.tensor([999]);
		// let eps = unet.apply([x, t]);
		// eps.print();

		// let gen_img = sampleEulerAncestral(unet, tf.randomNormal([1, 32, 32, 1]), 1);
		// gen_img = gen_img.squeeze(0).squeeze(-1);
		// gen_img = gen_img.mul(0.1850).add(0.0438);
		// imshow(undefined, gen_img);
		// gen_img.print();
		globalObj.unetPrepared = false;
		globalObj.isGenerating = false;
		let worker = new Worker("worker.js");
		globalObj.worker = worker;
		
		reloadModelBnElem.click();

		globalObj.worker.onmessage = evt => {
			if(evt.data.action === "load model"){
				if (evt.data.status === 0){
					globalObj.unetPrepared = true;
					generateBnElem.disabled = false;
					globalObj.worker.postMessage({
						action: 'set backend',
						backend: radioCpuElem.checked ? 'cpu' : 'webgl'
					});
				} else alert(evt.data.message);
			}else if (evt.data.action === "generate image"){
				globalObj.isGenerating = false;
				if (evt.data.status === 0){
					advancedImageShow(evt.data.image);
				} else alert(evt.data.message);
			} else if (evt.data.action === "progress"){
				progressbarElem.value = (evt.data.currentStep + 1);
			}
		}
	});
}