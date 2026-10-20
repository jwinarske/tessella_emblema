// Vertex work without fragment work: one multiply a vertex, and a position every vertex
// shares, so every triangle is degenerate and rasterizes nothing.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let t = f32(i) * 1e-9;
    return vec4<f32>(t, t, 0.0, 1.0);
}
