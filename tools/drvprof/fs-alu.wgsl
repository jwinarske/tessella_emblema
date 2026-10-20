// 256 dependent multiply-adds a fragment, seeded from the position so nothing folds.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var acc = pos.x * 0.001;
    for (var k = 0; k < 256; k = k + 1) {
        acc = acc * 1.0001 + 0.0001;
    }
    return vec4<f32>(fract(acc), 0.5, 0.75, 1.0);
}
