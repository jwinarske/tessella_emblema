// Sixteen samples a fragment, at coordinates that move with the pixel so the cache is
// exercised rather than one texel being read back.
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var k = 0; k < 16; k = k + 1) {
        let uv = (pos.xy + vec2<f32>(f32(k) * 3.0, f32(k) * 7.0)) / 512.0;
        acc = acc + textureSampleLevel(tex, samp, uv, 0.0);
    }
    return acc / 16.0;
}
