struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>, 
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var texture_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var sampler_diffuse: sampler;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // out.clip_position = vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // return vec4<f32>(in.uv.x, in.uv.y, 0.0, 1.0);
    return textureSample(texture_diffuse, sampler_diffuse, in.uv);

}