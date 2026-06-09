pub struct Gpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

impl<'a> Gpu<'a> {
    #[must_use]
    pub const fn borrowed(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        Self { device, queue }
    }

    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        self.device
    }

    #[must_use]
    pub const fn queue(&self) -> &wgpu::Queue {
        self.queue
    }
}
