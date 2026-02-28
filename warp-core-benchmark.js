// Warp-Core Performance Benchmark v3.0 (Tiled INT4 GEMM + Correct RMSNorm)
import { createPipeline } from "./warp-core-pipeline.js";

export async function runBenchmark() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });

    const dim = 4096;
    const iterations = 50;
    const pipeline = await createPipeline(device);

    const inputBuffer = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const weightBuffer = device.createBuffer({ size: (dim * dim / 8) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputBuffer = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(paramBuffer, 0, new Uint32Array([dim]));
    device.queue.writeBuffer(paramBuffer, 4, new Float32Array([1e-5]));
    device.queue.writeBuffer(paramBuffer, 8, new Float32Array([1.0]));

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: { buffer: paramBuffer } }
        ]
    });

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dim);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    await device.queue.onSubmittedWorkDone();
    console.log("Total ms:", performance.now() - start);
}
