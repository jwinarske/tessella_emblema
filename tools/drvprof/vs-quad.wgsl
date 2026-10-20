// One oversized triangle covering the target, with no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var x = -1.0;
    var y = -1.0;
    if i == 1u { x = 3.0; }
    if i == 2u { y = 3.0; }
    return vec4<f32>(x, y, 0.0, 1.0);
}
