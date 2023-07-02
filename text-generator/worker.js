
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js");
// importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tfjs-vis.umd.min.js");
importScripts('utils.js');
importScripts('modules.js');
importScripts('sampling.js');

let globalObj = {};

tf.setBackend('webgl');

onmessage = function(evt) {
	// tf.engine().startScope();
	
    if (evt.data.action === "load model"){
        let minConfig = {
			base_channels: 32,
			out_channels: 1,
			channels_mult: [1, 1, 2],
			num_blocks: [2, 2, 2],
			use_attentions: [false, false, false, false]
		};

		loadUNet(evt.data.path, minConfig).then(unet => {
            globalObj.unet = unet;
            postMessage({action: evt.data.action, message: "The model is loaded.", status: 0});
        }).catch(err => {
            postMessage({action: evt.data.action, message: "The model load failed. The error message is: " + String(err), status: 1});
        });
    } else if (evt.data.action === "set backend"){
        tf.setBackend(evt.data.backend);
        postMessage({action: evt.data.action, 
                     message: `The current backend is ${tf.getBackend()}.`, 
                     status: Number.parseInt(evt.data.backend != tf.getBackend())});
    } else if (evt.data.action === "generate image"){
        let callback = i => {
            postMessage({action: 'progress', 
                         status: 0,
                         currentStep: i});
        }
        let gen_img = sampleEulerAncestral(globalObj.unet, 
                                           tf.randomNormal([1, evt.data.height, evt.data.width, 1]),
                                           callback,
                                           evt.data.steps);
		gen_img = gen_img.squeeze(0).squeeze(-1);
		gen_img = gen_img.mul(0.1850).add(0.0438);
        postMessage({action: evt.data.action, 
                     message: `The image with ${gen_img.shape[0]}X${gen_img.shape[1]} pixels is generated.`, 
                     status: 0,
                     image: gen_img.arraySync()});
    }

	// tf.engine().endScope();
}


