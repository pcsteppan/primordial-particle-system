import { createShader, render } from "./lib.js";

const size = (type, numBytes) => { return { type, numBytes } }
const sizes = {
	f32: size('f32', 4),
	u32: size('u32', 4),
	i32: size('i32', 4),
	vec2: size('vec2', 8),
	vec4: size('vec4', 16),
};

console.log(sizes);

const uniform = (value, size) => { return { value, size } }
const uniforms = {
	rez: uniform(512, sizes.f32),
	time: uniform(0, sizes.f32),
	count: uniform(10000, sizes.u32),
	alpha: uniform(Math.PI, sizes.f32),
	beta: uniform(-0.023, sizes.f32),
	radius: uniform(24, sizes.f32),
	velocity: uniform(1, sizes.f32)
};

const deg_to_rad = deg => deg * (Math.PI / 180);
const presets = {
	default: {
		count: 10000,
		alpha: Math.PI,
		beta: -0.023,
		radius: 24,
		velocity: 1
	},
	// preset_1: {
	// 	alpha: deg_to_rad(-79.6),
	// 	beta: deg_to_rad(-0.8),
	// 	velocity: .3
	// }
}

let selectedPreset = {
	current: 'default'
}

const settings = {
	scale:
		(0.95 * Math.min(window.innerHeight, window.innerWidth)) / uniforms.rez.value,
	pixelWorkgroups: Math.ceil(uniforms.rez.value ** 2 / 256),
	agentWorkgroups: Math.ceil(uniforms.count.value / 256),
};

async function main() {
	if (!navigator.gpu) {
		throw Error("WebGPU not supported.");
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw Error("Couldn't request WebGPU adapter.");
	}

	const gpu = await adapter.requestDevice();

	const canvas = document.createElement("canvas");
	canvas.width = canvas.height = uniforms.rez.value * settings.scale;
	document.body.appendChild(canvas);
	const context = canvas.getContext("webgpu");
	const format = "bgra8unorm";
	context.configure({
		device: gpu,
		format: format,
		alphaMode: "premultiplied",
	});

	/////////////////////////
	// Set up memory resources
	const visibility = GPUShaderStage.COMPUTE;

	// Pixel buffer
	const pixelBuffer = gpu.createBuffer({
		size: uniforms.rez.value ** 2 * sizes.vec4.numBytes,
		usage: GPUBufferUsage.STORAGE,
	});
	const pixelBufferLayout = gpu.createBindGroupLayout({
		entries: [{ visibility, binding: 0, buffer: { type: "storage" } }],
	});
	const pixelBufferBindGroup = gpu.createBindGroup({
		layout: pixelBufferLayout,
		entries: [{ binding: 0, resource: { buffer: pixelBuffer } }],
	});

	// Uniform buffers
	const uniformBuffers = {};

	for (let [k, v] of Object.entries(uniforms)) {
		uniformBuffers[k] = gpu.createBuffer({
			size: v.size.numBytes,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
		});
	}

	const uniformsLayoutEntries = new Array(Object.keys(uniforms).length).fill(null).map((_, i) => {
		return {
			visibility, binding: i, buffer: { type: 'uniform' }
		}
	});
	const uniformsLayout = gpu.createBindGroupLayout({
		entries: uniformsLayoutEntries
	});

	const uniformBuffersBindGroup = gpu.createBindGroup({
		layout: uniformsLayout,
		entries: Array.from(Object.keys(uniforms)).map((k, i) => {
			return {
				binding: i,
				resource: {
					buffer: uniformBuffers[k]
				}
			}
		})
	});

	const writeUniforms = () => {
		for (let [k, v] of Object.entries(uniformBuffers)) {
			const newValue = uniforms[k].value;
			const newBufferValue = Array.isArray(newValue) ? newValue : [newValue];
			const newBuffer = uniforms[k].size.type === sizes.u32.type
				? new Uint32Array(newBufferValue)
				: new Float32Array(newBufferValue);
			gpu.queue.writeBuffer(v, 0, newBuffer);
		}
		settings.agentWorkgroups = Math.ceil(uniforms.count.value / 256);
	};

	writeUniforms();

	// Storage buffers
	const positionsBuffer = gpu.createBuffer({
		size: sizes.vec2.numBytes * uniforms.count.value,
		usage: GPUBufferUsage.STORAGE,
	});

	const headingsBuffer = gpu.createBuffer({
		size: sizes.vec2.numBytes * uniforms.count.value,
		usage: GPUBufferUsage.STORAGE,
	});

	const agentsLayout = gpu.createBindGroupLayout({
		entries: [
			{ visibility, binding: 0, buffer: { type: "storage" } },
			{ visibility, binding: 1, buffer: { type: "storage" } },
		],
	});

	const agentsBuffersBindGroup = gpu.createBindGroup({
		layout: agentsLayout,
		entries: [
			{ binding: 0, resource: { buffer: positionsBuffer } },
			{ binding: 1, resource: { buffer: headingsBuffer } },
		],
	});

	/////
	// Overall memory layout
	const layout = gpu.createPipelineLayout({
		bindGroupLayouts: [pixelBufferLayout, uniformsLayout, agentsLayout],
	});

	/////////////////////////
	// Set up code instructions
	const module = await createShader(gpu, "compute.wgsl");

	const resetPipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "reset" },
	});

	const simulatePipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "simulate" },
	});

	const fadePipeline = gpu.createComputePipeline({
		layout,
		compute: { module, entryPoint: "fade" },
	});

	/////////////////////////
	// RUN the reset shader function
	const reset = () => {
		const encoder = gpu.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(resetPipeline);
		pass.setBindGroup(0, pixelBufferBindGroup);
		pass.setBindGroup(1, uniformBuffersBindGroup);
		pass.setBindGroup(2, agentsBuffersBindGroup);
		pass.dispatchWorkgroups(settings.agentWorkgroups);
		pass.end();
		gpu.queue.submit([encoder.finish()]);
	};
	reset();

	/////////////////////////
	// RUN the sim compute function and render pixels
	const draw = () => {
		// Compute sim function
		const encoder = gpu.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setBindGroup(0, pixelBufferBindGroup);
		pass.setBindGroup(1, uniformBuffersBindGroup);
		pass.setBindGroup(2, agentsBuffersBindGroup);

		pass.setPipeline(fadePipeline);
		pass.dispatchWorkgroups(settings.pixelWorkgroups);

		pass.setPipeline(simulatePipeline);
		pass.dispatchWorkgroups(settings.agentWorkgroups);

		pass.end();

		// Render the pixels buffer to the canvas
		render(gpu, uniforms.rez.value, pixelBuffer, format, context, encoder);
		gpu.queue.submit([encoder.finish()]);
		gpu.queue.writeBuffer(uniformBuffers.time, 0, new Float32Array([uniforms.time.value++]));
		requestAnimationFrame(draw);
	};
	draw();

	let gui = new lil.GUI();
	gui.add(uniforms.alpha, 'value', 0, Math.PI)
		.name('alpha')
		.step(Math.PI / 128)
		.listen();
	gui.add(uniforms.beta, 'value', -Math.PI / 6, Math.PI / 6)
		.name('beta')
		.step(Math.PI / 128)
		.listen();
	gui.add(uniforms.radius, 'value', 0.0, 64)
		.name('radius')
		.listen();
	gui.add(uniforms.count, 'value', 0, 40000)
		.name('count')
		.listen();
	gui.add(uniforms.velocity, 'value', 0, 2)
		.name('velocity')
		.listen();
	gui.add(selectedPreset, 'current', Object.keys(presets)).onChange(preset => {
		selectedPreset.current = preset;
		const newSelectedPreset = presets[preset];
		for (let k in newSelectedPreset) {
			uniforms[k].value = newSelectedPreset[k];
		}
		writeUniforms();
	});
	gui.onChange(writeUniforms);
}
main();

