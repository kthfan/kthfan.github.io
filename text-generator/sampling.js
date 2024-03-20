


function tToSigma(t, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	t = t.div(num_timesteps - 1);
	let sigma = tf.sqrt(tf.exp(t.pow(2).mul(beta_d / 2).add(t.mul(beta_min))).sub(1));
	return sigma;
}
function sigmaToT(sigma, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	let t = sigma.pow(2).add(1).log().mul(2 * beta_d).add(beta_min**2).sqrt().add(-beta_min).div(beta_d);
    t = t.mul(num_timesteps - 1);
	return t;
}

function ddpmDenoise(model, x, t, sigma){
	let c_out = - sigma;
	let c_in = 1 / (sigma ** 2 + 1. ** 2) ** 0.5;
	let s_in = tf.ones([x.shape[0]]);
	let eps = model.apply([x.mul(c_in), s_in.mul(t)]);
	let denoised = x.add(eps.mul(c_out));
	return denoised;
}

function getAncestralStep(sigma_from, sigma_to, eta=1.){
	if (!eta)
		return sigma_to, 0.;
	let sigma_up = Math.min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
	let sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
	return [sigma_down, sigma_up];
}

function sampleEulerAncestral(model, x, callback, steps=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015, eta=1.){
	return sampleEulerAncestralManual(model, x.mul(38.6546), 38.6546, callback, steps, num_timesteps, beta_d, beta_min, eta);
}

function sampleEulerAncestralManual(model, x, sigmaMax, callback, steps=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015, eta=1.){
	if(callback === undefined)
		callback = () => {};
	let tMax = sigmaToT(tf.tensor(sigmaMax), num_timesteps, beta_d, beta_min);
	tMax = Math.ceil(tMax.arraySync());
    steps = Math.ceil(steps / (num_timesteps - 1) * tMax);
	if (steps <= 0 || tMax - 1 <= 0) {
		return x;
	}
	
    let tList = tf.linspace(tMax - 1, 1, steps);
	let sigmas = tToSigma(tList, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	tList = tList.arraySync();
	for(let i=0; i < steps; i++){
		callback(i);
		x = tf.tidy(() => {
			let t = tList[i];
			let sigma = sigmas[i];
			let sigma_next = sigmas[i + 1];
			
			let denoised = ddpmDenoise(model, x, t, sigma);
			let [sigma_down, sigma_up] = getAncestralStep(sigma, sigma_next, eta);
			let d =  tf.sub(x, denoised).div(sigma);
			let dt = sigma_down - sigma;
			x = x.add(d.mul(dt));
			if(sigma_next > 0)
				x = x.add(tf.randomNormal(x.shape).mul(sigma_up));
			return x;
		});
	}
	return x
}

function lowPassFilter(x, D){
	let h = x.shape[1];
	let w = x.shape[2];
	x = applyGaussianBlur(x, 3, 1);
	x = tf.image.resizeBilinear(x, [Number.parseInt(h / D), Number.parseInt(w / D)]);
	x = tf.image.resizeBilinear(x, [h, w]);
	return x;
}
function sampleEulerAncestralWithILVR(model, x, ref, callback, steps=20, guideRatio=0.5, num_timesteps=1000, 
									  beta_d=14.617, beta_min=0.0015, eta=1., D=2){
	let t_list = tf.linspace(num_timesteps - 1, 1, steps);
	let sigmas = tToSigma(t_list, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	t_list = t_list.arraySync();

	x = x.mul(sigmas[0]);
	for(let i=0; i < steps; i++){
		callback(i);
		x = tf.tidy(() => {
			let t = t_list[i];
			let sigma = sigmas[i];
			let sigma_next = sigmas[i + 1];
			
			let denoised = ddpmDenoise(model, x, t, sigma);
			let [sigma_down, sigma_up] = getAncestralStep(sigma, sigma_next, eta);
			
			let d =  tf.sub(x, denoised).div(sigma);
			let dt = sigma_down - sigma;
			x = x.add(d.mul(dt));
			if(sigma_next > 0){
				x = x.add(tf.randomNormal(x.shape).mul((1 - guideRatio**0.5) * sigma_up));

				// ILVER
				let ref_t = ref.add(tf.randomNormal(x.shape).mul(sigma_next));
				let guide = lowPassFilter(ref_t, D).sub(lowPassFilter(x, D));
				guide = guide.div(computeVar(guide).add(1e-7).sqrt()).mul(guideRatio * sigma_up);
				x = x.add(guide);
			}
			
			return x;
		});
	}
	return x
}


function qSamplePair(x1, x2, n=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
    let sigmaMax = tToSigma(tf.tensor(num_timesteps - 1), num_timesteps, beta_d, beta_min).arraySync();
    // let sigmas = tf.linspace(0, sigmaMax, n + 1);
	let sigmas = tToSigma(tf.linspace(1, num_timesteps - 1, n - 1), num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([tf.zeros([1]), sigmas], 0)
	sigmas = sigmas.arraySync();
    let noise = tf.randomNormal(x1.shape);
	let x1List = [];
	let x2List = [];
    for(let i=0; i < n; i++){
        let sigma = sigmas[i];
        let alpha = 0.5 * sigma / sigmaMax;
        let _x1 = x1.mul(1 - alpha).add(x2.mul(alpha)).add(noise.mul(sigma));
        let _x2 = x1.mul(alpha).add(x2.mul(1 - alpha)).add(noise.mul(sigma));
        x1List.push(_x1);
        x2List.push(_x2);
	}
    x2List.reverse();
    return x1List.concat(x2List);
}

function sampleOdeManuel(model, x, sigmaMax, callback, steps=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	if(callback === undefined)
		callback = () => {};
	let tMax = sigmaToT(tf.tensor(sigmaMax), num_timesteps, beta_d, beta_min);
	tMax = Math.ceil(tMax.arraySync());
    steps = Math.ceil(steps / (num_timesteps - 1) * tMax);
	if (steps <= 0) {
		return x;
	}
	
    let tList = tf.linspace(tMax - 1, 1, steps);
    let sigmas = tToSigma(tList, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	tList = tList.arraySync();
    for (let i=0; i < steps; i++){
		x = tf.tidy(() => {
			return ddpmDenoise(model, x, tList[i], sigmas[i]);
		});
	}
    return x
}

function pSamplePair(model, xList, callback, steps=20, n=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	let yList = [];
    // let sigmaMax = tToSigma(tf.tensor(num_timesteps - 1), num_timesteps, beta_d, beta_min).arraySync();
    // let sigmas = tf.linspace(0, sigmaMax, n);
	let sigmas = tToSigma(tf.linspace(1, num_timesteps - 1, n - 1), num_timesteps, beta_d, beta_min);
    sigmas = tf.concat([tf.zeros([1]), sigmas], 0);
	sigmas = tf.concat([sigmas, sigmas.reverse(0)], 0);
	sigmas = sigmas.arraySync();
	
    for (let i=0; i < xList.length; i++){
		callback(i);
		let y = sampleEulerAncestralManual(model, xList[i], sigmas[i], undefined, steps, num_timesteps, beta_d, beta_min);
		yList.push(y);
	}
	
	return yList;
}