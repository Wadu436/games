use std::{
    sync::Arc,
    time::{self, Duration},
};

use cgmath::{InnerSpace, Point2, SquareMatrix, Vector2};
use rand::Rng;
use tracing::{debug, error, info, Level};
use tracing_subscriber::FmtSubscriber;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferUsages, PipelineLayoutDescriptor,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderStages, SurfaceConfiguration, VertexBufferLayout,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const BALL_SPEED: f32 = 2.0;
const PADDLE_SPEED: f32 = 1.25;

const PADDLE_HEIGHT: f32 = 0.5;
const PADDLE_WIDTH: f32 = 0.1;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    const ATTRIBS: &'static [wgpu::VertexAttribute] = &vertex_attr_array![0=> Float32x3];

    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBS,
        }
    }
}

struct Rectangle {
    pub width: f32,
    pub height: f32,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Rectangle {
    fn new(device: &wgpu::Device, width: f32, height: f32) -> Self {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                Vertex {
                    position: [-width / 2., -height / 2., 0.0],
                },
                Vertex {
                    position: [width / 2., -height / 2., 0.0],
                },
                Vertex {
                    position: [-width / 2., height / 2., 0.0],
                },
                Vertex {
                    position: [width / 2., height / 2., 0.0],
                },
            ]),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0, 1, 2, 2, 1, 3]),
            usage: BufferUsages::INDEX,
        });

        Self {
            width,
            height,
            index_buffer,
            vertex_buffer,
            num_indices: 6,
        }
    }
}

// #[rustfmt::skip]
// pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
//     1.0, 0.0, 0.0, 0.0,
//     0.0, 1.0, 0.0, 0.0,
//     0.0, 0.0, 0.5, 0.5,
//     0.0, 0.0, 0.0, 1.0,
// );

// Needed because the projection matrix from cgmath maps to -1.0 <= z <= 1.0, while wgpu uses 0.0 <= z <= 1.0 as its clipping coords
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

struct Camera {
    center: cgmath::Point2<f32>,
    aspect: f32,
}

impl Camera {
    fn new(center: cgmath::Point2<f32>, viewport_size: winit::dpi::PhysicalSize<u32>) -> Self {
        let aspect = viewport_size.width as f32 / viewport_size.height as f32;
        Self { center, aspect }
    }

    fn resize(&mut self, new_viewport_size: winit::dpi::PhysicalSize<u32>) {
        info!("Resizing camera!");
        self.aspect = new_viewport_size.width as f32 / new_viewport_size.height as f32;
    }

    // From world coordinates to clipping coordinates
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let (width, height) = if self.aspect >= 1.0 {
            (2.0 * self.aspect, 2.0)
        } else {
            (2.0, 2.0 / self.aspect)
        };

        info!(
            "Width: {}, Height: {}, Aspect ratio: {}, Intended Aspect ratio: {}",
            width,
            height,
            width / height,
            self.aspect
        );

        let proj = cgmath::ortho(
            self.center.x - (width / 2.0),
            self.center.x + (width / 2.0),
            self.center.y - (height / 2.0),
            self.center.y + (height / 2.0),
            0.0,
            1.0,
        );
        OPENGL_TO_WGPU_MATRIX * proj
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniform {
    translation: [f32; 2],
}

struct Model {
    pub translation: cgmath::Vector2<f32>,

    // Render stuff
    pub rectangle: Rectangle,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

impl Model {
    fn top(&self) -> f32 {
        self.translation.y - self.rectangle.height / 2.0
    }

    fn bottom(&self) -> f32 {
        self.translation.y + self.rectangle.height / 2.0
    }

    fn left(&self) -> f32 {
        self.translation.x - self.rectangle.width / 2.0
    }

    fn right(&self) -> f32 {
        self.translation.x + self.rectangle.width / 2.0
    }

    fn center(&self) -> Vector2<f32> {
        self.translation
    }

    fn new(
        device: &wgpu::Device,
        width: f32,
        height: f32,
        x: f32,
        y: f32,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let rectangle = Rectangle::new(device, width, height);

        let translation = cgmath::Vector2::new(x, y);

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[ModelUniform {
                translation: translation.into(),
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            translation,
            rectangle,
            bind_group,
            uniform_buffer,
        }
    }

    fn update_uniform(&self, queue: &wgpu::Queue) {
        let data = ModelUniform {
            translation: self.translation.into(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[data]));
    }

    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.rectangle.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.rectangle.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.rectangle.num_indices, 0, 0..1);
    }

    fn bind_group_layout_descriptor() -> wgpu::BindGroupLayoutDescriptor<'static> {
        wgpu::BindGroupLayoutDescriptor {
            label: Some("model_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        }
    }
}

struct GameObjects {
    ball: Model,
    paddle_1: Model,
    paddle_2: Model,
    walls: Vec<Model>,
}

impl GameObjects {
    fn new(device: &wgpu::Device, layout: &BindGroupLayout) -> Self {
        let ball = Model::new(device, 0.1, 0.1, 0.0, 0.0, layout);
        let paddle_1 = Model::new(device, PADDLE_WIDTH, PADDLE_HEIGHT, -0.9, 0.2, layout);
        let paddle_2 = Model::new(device, PADDLE_WIDTH, PADDLE_HEIGHT, 0.9, -0.6, layout);
        let wall_1 = Model::new(device, 2.0, 0.025, 0.0, 0.975, layout);
        let wall_2 = Model::new(device, 2.0, 0.025, 0.0, -0.975, layout);

        GameObjects {
            ball,
            paddle_1,
            paddle_2,
            walls: vec![wall_1, wall_2],
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct InputState {
    move_up: bool,
    move_down: bool,
    shoot: bool,
}

// What part of the game we're in. e.g. waiting to shoot, playing, etc
#[derive(Copy, Clone, Debug, Default)]
enum GamePhase {
    #[default]
    Ready,
    Playing,
}

struct GameState {
    surface: wgpu::Surface<'static>,
    surface_config: SurfaceConfiguration,
    render_pipeline: RenderPipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    last_render_time: time::Instant,
    average_fps: f32,
    frametimes: Vec<f32>,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    game_objects: GameObjects,
    last_frame_start: Option<time::Instant>,
    paddle_2_target: f32,
    input_state: InputState,
    game_phase: GamePhase,
    score: (u32, u32),
    ball_direction: cgmath::Vector2<f32>,
}

impl GameState {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Unable to acquire surface from window");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Unable to acquire an adapter");
        let adapter_info = adapter.get_info();
        debug!(
            "using adapter `{}` with backend `{}`",
            adapter_info.name,
            adapter_info.backend.to_str()
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Unable to acquire a device");
        debug!("Created device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            alpha_mode: surface_caps.alpha_modes[0],
            // present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::AutoVsync,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        debug!("Configured surface {:?}", surface);

        let camera = Camera::new(Point2::new(0., 0.), window.inner_size());

        let mut camera_uniform = CameraUniform::new();

        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera uniform"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load the shaders
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let model_bind_group_layout =
            device.create_bind_group_layout(&Model::bind_group_layout_descriptor());

        let game_objects = GameObjects::new(&device, &model_bind_group_layout);

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions {
                    ..Default::default()
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(surface_format.into())],
                compilation_options: wgpu::PipelineCompilationOptions {
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            surface_config,
            render_pipeline,
            device,
            queue,
            size,
            last_render_time: time::Instant::now(),
            average_fps: 0.,
            frametimes: Vec::new(),
            game_objects,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            last_frame_start: None,
            paddle_2_target: 0.0,
            input_state: InputState::default(),
            ball_direction: cgmath::Vector2 { x: -1.25, y: 1.25 },
            game_phase: GamePhase::default(),
            score: (0, 0),
        }
    }

    pub fn input(&mut self, event: &KeyEvent) {
        let pressed = event.state.is_pressed();
        match event.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) => self.input_state.move_up = pressed,
            PhysicalKey::Code(KeyCode::ArrowDown) => self.input_state.move_down = pressed,
            PhysicalKey::Code(KeyCode::Space) => self.input_state.shoot = pressed,
            _ => {}
        }
    }

    pub fn update(&mut self) {
        let current_frame_start = std::time::Instant::now();
        let dt = self
            .last_frame_start
            .map_or(Duration::ZERO, |t| current_frame_start.duration_since(t));
        let dt = dt.as_secs_f32();

        // Update the paddle position and ball position then write the uniform buffers again
        let paddle_1 = &mut self.game_objects.paddle_1;
        let paddle_2 = &mut self.game_objects.paddle_2;
        let ball = &mut self.game_objects.ball;

        // User input
        if self.input_state.move_up {
            paddle_1.translation.y = (paddle_1.translation.y + dt * PADDLE_SPEED)
                .min(0.95 - paddle_1.rectangle.height / 2.0);
        }
        if self.input_state.move_down {
            paddle_1.translation.y = (paddle_1.translation.y - dt * PADDLE_SPEED)
                .max(-0.95 + paddle_1.rectangle.height / 2.0);
        }

        match self.game_phase {
            GamePhase::Ready => {
                ball.translation =
                    paddle_1.translation + Vector2::unit_x() * (1.25 * paddle_1.rectangle.width);

                if self.input_state.shoot {
                    self.game_phase = GamePhase::Playing;
                    self.ball_direction = (Vector2::unit_x() + Vector2::unit_y()).normalize();
                }
            }
            GamePhase::Playing => {
                // Update the ball position
                ball.translation += self.ball_direction * BALL_SPEED * dt;

                // Check collision with wall
                if ball.translation.y >= (0.9625 - ball.rectangle.height / 2.0) {
                    // reflect off the wall
                    ball.translation.y +=
                        -2.0 * (ball.translation.y - (0.9625 - ball.rectangle.height / 2.0));
                    self.ball_direction.y *= -1.;
                }
                if ball.translation.y <= (-0.9625 + ball.rectangle.height / 2.0) {
                    ball.translation.y +=
                        -2.0 * (ball.translation.y - (-0.9625 + ball.rectangle.height / 2.0));
                    self.ball_direction.y *= -1.;
                }

                fn ball_reflection_direction(ball: &Model, wall: &Model) -> Vector2<f32> {
                    let relative_intersect = (2.0 * (ball.center().y - wall.center().y)
                        / wall.rectangle.height)
                        .min(1.0)
                        .max(-1.0);
                    // 75 degrees to -75 degrees
                    let angle = 75.0 / 180.0 * std::f32::consts::PI * relative_intersect;

                    if ball.center().x < wall.center().x {
                        Vector2::new(-f32::cos(angle), f32::sin(angle))
                    } else {
                        Vector2::new(f32::cos(angle), f32::sin(angle))
                    }
                }

                // Check collision with paddles
                // Left Paddle (paddle 1)
                if self.ball_direction.x <= 0.0
                    && ball.left() <= paddle_1.right()
                    && ball.right() >= paddle_1.left()
                    && ball.top() <= paddle_1.bottom()
                    && ball.bottom() >= paddle_1.top()
                {
                    // self.ball_direction.x = self.ball_direction.x.abs();
                    self.ball_direction = ball_reflection_direction(ball, paddle_1);
                }

                // Right paddle (paddle 2)
                if self.ball_direction.x >= 0.0
                    && ball.left() <= paddle_2.right()
                    && ball.right() >= paddle_2.left()
                    && ball.top() <= paddle_2.bottom()
                    && ball.bottom() >= paddle_2.top()
                {
                    self.ball_direction = ball_reflection_direction(ball, paddle_2);
                    let range = 0.9 * PADDLE_HEIGHT / 2.0;
                    self.paddle_2_target = rand::thread_rng().gen_range(-range..range);
                }

                // Check if the ball is behind the paddle
                if ball.right() < paddle_1.left() - 0.1 {
                    self.game_phase = GamePhase::Ready;
                    self.score.1 += 1;
                    info!("Score: {} - {}", self.score.0, self.score.1);
                }
                if ball.left() > paddle_2.right() + 0.1 {
                    self.game_phase = GamePhase::Ready;
                    self.score.0 += 1;
                    info!("Score: {} - {}", self.score.0, self.score.1);
                }

                // update paddle 2
                if ball.translation.x > 0.0 && self.ball_direction.x > 0.0 {
                    // Move the paddle to the ball
                    let delta_ball_paddle = ball.translation.y - paddle_2.translation.y + 0.1;
                    if delta_ball_paddle > 0.0 {
                        // Up
                        paddle_2.translation.y += (dt * PADDLE_SPEED).min(delta_ball_paddle);
                        if paddle_2.translation.y >= (0.95 - paddle_2.rectangle.height / 2.0) {
                            paddle_2.translation.y += -2.0
                                * (paddle_2.translation.y
                                    - (0.95 - paddle_2.rectangle.height / 2.0));
                        }
                    } else {
                        // Down
                        paddle_2.translation.y += (dt * -PADDLE_SPEED).max(delta_ball_paddle);
                        if paddle_2.translation.y <= -0.95 + paddle_2.rectangle.height / 2.0 {
                            paddle_2.translation.y += -2.0
                                * (paddle_2.translation.y
                                    - (-0.95 + paddle_2.rectangle.height / 2.0));
                        }
                    }
                }
            }
        }

        paddle_1.update_uniform(&self.queue);
        paddle_2.update_uniform(&self.queue);
        ball.update_uniform(&self.queue);

        self.last_frame_start = Some(current_frame_start);
    }

    pub fn resize(&mut self) {
        info!("Resizing the window");
        // Recreate the swapchain
        self.surface_config.width = self.size.width;
        self.surface_config.height = self.size.height;
        self.surface.configure(&self.device, &self.surface_config);

        // Update the camera and camera uniform
        self.camera.resize(self.size);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn set_size(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Draw ball
            self.game_objects.ball.render(&mut render_pass);
            self.game_objects.paddle_1.render(&mut render_pass);
            self.game_objects.paddle_2.render(&mut render_pass);
            self.game_objects
                .walls
                .iter()
                .for_each(|wall| wall.render(&mut render_pass));
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        let current_time = time::Instant::now();

        let frametime = (current_time - self.last_render_time).as_secs_f32();
        self.frametimes.push(frametime);
        let total_frametime = self.frametimes.iter().sum::<f32>();
        if total_frametime > 0.25 {
            self.average_fps = self.frametimes.len() as f32 / total_frametime;
            self.frametimes.clear();
        }
        // info!(
        //     "frametime: {:.2} ms ({:.0} fps)",
        //     1000. * frametime,
        //     self.average_fps
        // );

        self.last_render_time = current_time;

        Ok(())
    }
}

#[derive(Default)]
enum AppState {
    #[default]
    Uninitialized,
    Initialized {
        window: Arc<Window>,
        game_state: GameState,
    },
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppState::Uninitialized = self {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Pong")
                            .with_visible(false)
                            // setting min and max size to 1024x768 forces i3 to make the window floating
                            .with_min_inner_size(PhysicalSize {
                                width: 1024,
                                height: 768,
                            })
                            .with_max_inner_size(PhysicalSize {
                                width: 1024,
                                height: 768,
                            }),
                    )
                    .unwrap(),
            );

            let game_state = pollster::block_on(GameState::new(window.clone()));

            window.set_visible(true);

            *self = AppState::Initialized { window, game_state };
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let AppState::Initialized { game_state, window } = self {
            if window_id != window.id() {
                return;
            }

            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(size) => game_state.set_size(size),
                WindowEvent::KeyboardInput { event, .. } => {
                    game_state.input(&event);
                }
                WindowEvent::RedrawRequested => {
                    game_state.update();
                    match game_state.render() {
                        Ok(()) => (),
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            game_state.resize()
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            error!("Out of memory; exiting");
                            event_loop.exit();
                        }
                        Err(e) => error!("error while rendering: {:?}", e),
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppState::Initialized { window, .. } = self {
            window.request_redraw();
        }
    }
}

pub fn run() -> eyre::Result<()> {
    // Initialize tracing
    FmtSubscriber::builder().with_max_level(Level::INFO).init();

    // Build window, keep it hidden untill we're ready to start rendering
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = AppState::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
