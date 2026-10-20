// A dependent chain that cannot be folded: squaring is not affine, so no compiler can
// collapse it the way both of these fold `acc * k + c` repeated. This is the pair that
// measures ALU throughput rather than an optimizer.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var acc = pos.x * 0.001;
    for (var k = 0; k < 256; k = k + 1) {
        acc = acc * acc + 0.0001;
    }
    return vec4<f32>(fract(acc), 0.5, 0.75, 1.0);
}
