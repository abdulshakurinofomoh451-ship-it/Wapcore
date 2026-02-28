import shader from "./warp-core.wgsl";

export async function createPipeline(device) {
    return device.createComputePipeline({
        layout: "auto",
        compute: {
            module: device.createShaderModule({ code: shader }),
            entryPoint: "main"
        }
    });
}
