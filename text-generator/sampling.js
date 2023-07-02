


function tToSigma(t, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	t = t.div(num_timesteps - 1);
	let sigma = tf.sqrt(tf.exp(t.pow(2).mul(beta_d / 2).add(t.mul(beta_min))).sub(1));
	return sigma;
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
	let t_list = tf.linspace(num_timesteps - 1, 1, steps);
	let sigmas = tToSigma(t_list, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	t_list = t_list.arraySync();

	x = x.mul(sigmas[0]);
	for(i=0; i < steps; i++){
		callback(i);
		let t = t_list[i];
		let sigma = sigmas[i];
		let sigma_next = sigmas[i + 1];
		
		let denoised = ddpmDenoise(model, x, t, sigma);
		let [sigma_down, sigma_up] = getAncestralStep(sigma, sigma_next, eta);
		
		let d =  tf.sub(x, denoised).div(sigma);
		let dt = sigma_down - sigma;
		x = x.add(d.mul(dt));
		if(sigma_next > 0)
			x = x.add(tf.randomNormal(x.shape).mul(sigma_up));
	}
	return x
}


