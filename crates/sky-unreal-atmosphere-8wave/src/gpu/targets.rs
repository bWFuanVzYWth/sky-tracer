//! 渲染目标。
//!
//! 目标按 resize 生命周期分组，不按 pass 所有权分组。pass 专属资源仍由对应
//! renderer 结构持有。

/// 主 scene radiance / post target 格式。
///
/// PT 接入后 radiance 与 accumulation 需要足够大的动态范围，主 scene 边界统一
/// 使用 32-bit float，避免中间 target 截断结果。
pub const SCENE_RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// 深度目标格式。
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// 非零渲染目标尺寸。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonZeroRenderSize {
    width: u32,
    height: u32,
}

impl NonZeroRenderSize {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Option<Self> {
        if width > 0 && height > 0 {
            Some(Self { width, height })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// resize 绑定的 scene targets。
#[derive(Debug)]
pub struct RenderTargets {
    size: NonZeroRenderSize,
    _scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    _post_texture: wgpu::Texture,
    post_view: wgpu::TextureView,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl RenderTargets {
    #[must_use]
    pub fn new(device: &wgpu::Device, size: NonZeroRenderSize) -> Self {
        let (scene_texture, scene_view) = color_target(
            device,
            size,
            SCENE_RADIANCE_FORMAT,
            "ca_render.scene_radiance",
        );
        let (post_texture, post_view) =
            color_target(device, size, SCENE_RADIANCE_FORMAT, "ca_render.scene_post");
        let (depth_texture, depth_view) = depth_target(device, size);
        Self {
            size,
            _scene_texture: scene_texture,
            scene_view,
            _post_texture: post_texture,
            post_view,
            _depth_texture: depth_texture,
            depth_view,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: NonZeroRenderSize) {
        if self.size == size {
            return;
        }
        *self = Self::new(device, size);
    }

    #[must_use]
    pub const fn size(&self) -> NonZeroRenderSize {
        self.size
    }

    #[must_use]
    pub const fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    #[must_use]
    pub const fn post_view(&self) -> &wgpu::TextureView {
        &self.post_view
    }

    #[must_use]
    pub const fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }
}

fn color_target(
    device: &wgpu::Device,
    size: NonZeroRenderSize,
    format: wgpu::TextureFormat,
    label: &'static str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    });
    (texture, view)
}

fn depth_target(
    device: &wgpu::Device,
    size: NonZeroRenderSize,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ca_render.depth"),
        size: extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ca_render.depth.view"),
        ..Default::default()
    });
    (texture, view)
}

const fn extent(size: NonZeroRenderSize) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.width(),
        height: size.height(),
        depth_or_array_layers: 1,
    }
}
