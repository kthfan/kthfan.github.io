
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js");
// importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tfjs-vis.umd.min.js");
importScripts('utils.js');
importScripts('modules.js');
importScripts('sampling.js');

let globalObj = {};

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
        let genImg = tf.tidy(() => {
            let genImg = sampleEulerAncestral(globalObj.unet, 
                            tf.randomNormal([1, evt.data.height, evt.data.width, 1]),
                            callback,
                            evt.data.steps);
            genImg = genImg.squeeze(0).squeeze(-1);
            genImg = genImg.mul(0.1850).add(0.0438);
            genImg = tf.clipByValue(genImg, 0, 1);
            return genImg;
        });
        
        postMessage({action: evt.data.action, 
                     message: `The image with ${genImg.shape[0]}X${genImg.shape[1]} pixels is generated.`, 
                     status: 0,
                     image: genImg.arraySync()});
        genImg.dispose();
    }

	// tf.engine().endScope();
}


