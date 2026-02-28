@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<u32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params : vec4<f32>; // dim, eps, scale, _

var<workgroup> tileInput : array<f32, 256>;
var<workgroup> tileWeight : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {

    let row = wg.x;
    let dim = u32(params.x);
    let tid = lid.x;
    if (row >= dim) { return; }

    var sum : f32 = 0.0;
    var sqsum : f32 = 0.0;

    for (var tile = 0u; tile < dim; tile += 256u) {
        let idx = tile + tid;
        if (idx < dim) {
            tileInput[tid] = input[idx];
            let packed = weights[(row * dim + idx) >> 3u];
            let shift = (idx & 7u) * 4u;
            tileWeight[tid] = f32((packed >> shift) & 0xFu) - 8.0;
        }
        workgroupBarrier();

        if (idx < dim) {
            let v = tileInput[tid] * tileWeight[tid];
            sum += v;
            sqsum += v * v;
        }
        workgroupBarrier();
    }

    // Reduce within workgroup
    for (var offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            sum += workgroupUniformLoad(&sum, tid + offset);
            sqsum += workgroupUniformLoad(&sqsum, tid + offset);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let rms = sqrt(sqsum / f32(dim) + params.y);
        output[row] = (sum / rms) * params.z;
    }
}
