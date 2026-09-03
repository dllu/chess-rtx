#![allow(clippy::borrow_as_ptr, clippy::ptr_as_ptr, clippy::ref_as_ptr)]

use std::{
    ffi::{CStr, CString},
    io::Cursor,
    mem::size_of,
    ptr,
};

use anyhow::{anyhow, bail, Context, Result};
use ash::{khr, vk, Device, Entry, Instance};
use bytemuck::{Pod, Zeroable};

use crate::{
    material::{BoardStyle, GpuMaterial, RenderSettings},
    scene::{ChessScene, Vertex},
};

const RAY_GROUP_COUNT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    camera_position: [f32; 4],
    camera_forward: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    /// xyz: position, w: radiant intensity
    light_position: [f32; 4],
    /// exposure, light radius, samples, maximum bounces
    render: [f32; 4],
    /// reflections, refractions, caustics, soft shadows
    effects: [f32; 4],
    /// width, height, frame index, board style
    resolution: [f32; 4],
}

struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_family: u32,
    queue: vk::Queue,
    acceleration_structure: khr::acceleration_structure::Device,
    ray_tracing_pipeline: khr::ray_tracing_pipeline::Device,
    rt_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    device_name: String,
}

impl VulkanContext {
    fn new() -> Result<Self> {
        // SAFETY: Vulkan's loader is opened for the lifetime of this context.
        let entry = unsafe { Entry::load() }.context("could not load libvulkan")?;
        let application_name = CString::new("Chess RTX")?;
        let engine_name = CString::new("Chess RTX")?;
        let application_info = vk::ApplicationInfo {
            p_application_name: application_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };
        let instance_info = vk::InstanceCreateInfo {
            p_application_info: &application_info,
            ..Default::default()
        };
        // SAFETY: all pointers in instance_info remain alive for the call.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("vkCreateInstance failed")?;

        let required_extensions = [
            vk::KHR_ACCELERATION_STRUCTURE_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
            vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
            vk::KHR_SPIRV_1_4_NAME,
            vk::KHR_SHADER_FLOAT_CONTROLS_NAME,
        ];

        // SAFETY: instance is valid and remains so while the returned handles are inspected.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("could not enumerate Vulkan devices")?;
        let mut candidates = Vec::new();
        for physical_device in physical_devices {
            // SAFETY: physical_device came from this instance.
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }
                    .unwrap_or_default();
            let supports_extensions = required_extensions.iter().all(|required| {
                extensions.iter().any(|available| {
                    // SAFETY: Vulkan guarantees extension_name is NUL-terminated.
                    unsafe { CStr::from_ptr(available.extension_name.as_ptr()) == *required }
                })
            });
            if !supports_extensions {
                continue;
            }

            let mut buffer_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut ray_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
            let mut acceleration = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            ray_pipeline.p_next = (&mut buffer_address as *mut _) as *mut _;
            acceleration.p_next = (&mut ray_pipeline as *mut _) as *mut _;
            let mut features = vk::PhysicalDeviceFeatures2 {
                p_next: (&mut acceleration as *mut _) as *mut _,
                ..Default::default()
            };
            // SAFETY: the pNext chain is valid and writable for the duration of the call.
            unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
            if acceleration.acceleration_structure == vk::FALSE
                || ray_pipeline.ray_tracing_pipeline == vk::FALSE
                || buffer_address.buffer_device_address == vk::FALSE
            {
                continue;
            }

            // SAFETY: physical_device came from this instance.
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let Some(queue_family) = queue_families.iter().position(|properties| {
                properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) else {
                continue;
            };
            // SAFETY: physical_device came from this instance.
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 10_000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1_000,
                _ => 0,
            } + properties.limits.max_compute_shared_memory_size;
            candidates.push((score, physical_device, queue_family as u32));
        }

        candidates.sort_unstable_by_key(|candidate| candidate.0);
        let Some((_, physical_device, queue_family)) = candidates.pop() else {
            // SAFETY: no children were created from the instance.
            unsafe { instance.destroy_instance(None) };
            bail!(
                "no Vulkan GPU supports VK_KHR_ray_tracing_pipeline, \
                 VK_KHR_acceleration_structure, and buffer device addresses"
            );
        };

        let queue_priority = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };
        let extension_names: Vec<*const i8> = required_extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect();
        let mut buffer_address = vk::PhysicalDeviceBufferDeviceAddressFeatures {
            buffer_device_address: vk::TRUE,
            ..Default::default()
        };
        let mut ray_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
            ray_tracing_pipeline: vk::TRUE,
            ..Default::default()
        };
        let mut acceleration = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
            acceleration_structure: vk::TRUE,
            ..Default::default()
        };
        ray_pipeline.p_next = (&mut buffer_address as *mut _) as *mut _;
        acceleration.p_next = (&mut ray_pipeline as *mut _) as *mut _;
        let device_info = vk::DeviceCreateInfo {
            p_next: (&mut acceleration as *mut _) as *const _,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            ..Default::default()
        };
        // SAFETY: all create-info pointers and the feature chain are valid for the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed for the ray-tracing device")?;
        // SAFETY: queue zero was requested from queue_family.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let acceleration_structure = khr::acceleration_structure::Device::new(&instance, &device);
        let ray_tracing_pipeline = khr::ray_tracing_pipeline::Device::new(&instance, &device);
        let mut rt_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties2 = vk::PhysicalDeviceProperties2 {
            p_next: (&mut rt_properties as *mut _) as *mut _,
            ..Default::default()
        };
        // SAFETY: the properties pNext chain is valid and writable.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        // SAFETY: Vulkan guarantees device_name is NUL-terminated.
        let device_name = unsafe { CStr::from_ptr(properties2.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        log::info!(
            "RTX device: {device_name}; shader-group handle={} bytes, max recursion={}",
            rt_properties.shader_group_handle_size,
            rt_properties.max_ray_recursion_depth
        );

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue_family,
            queue,
            acceleration_structure,
            ray_tracing_pipeline,
            rt_properties,
            device_name,
        })
    }

    fn find_memory_type(
        &self,
        allowed_types: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        // SAFETY: physical_device belongs to this instance.
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..properties.memory_type_count)
            .find(|&index| {
                allowed_types & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or_else(|| anyhow!("no Vulkan memory type supports {required:?}"))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: all child objects are destroyed by RayTracer before this context is dropped.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    address: vk::DeviceAddress,
    size: vk::DeviceSize,
}

#[derive(Clone, Copy)]
struct AccelerationResource {
    handle: vk::AccelerationStructureKHR,
    storage: BufferResource,
}

#[derive(Clone, Copy)]
struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// A native Vulkan renderer using `VK_KHR_ray_tracing_pipeline`.
pub struct RayTracer {
    context: VulkanContext,
    command_pool: vk::CommandPool,
    vertex_buffer: BufferResource,
    index_buffer: BufferResource,
    triangle_material_buffer: BufferResource,
    material_buffer: BufferResource,
    uniform_buffer: BufferResource,
    readback_buffer: BufferResource,
    shader_binding_table: BufferResource,
    blas: AccelerationResource,
    tlas: AccelerationResource,
    output: ImageResource,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sbt_regions: [vk::StridedDeviceAddressRegionKHR; 4],
    width: u32,
    height: u32,
    frame_index: u32,
    pipeline_recursion_depth: u32,
}

impl RayTracer {
    /// Creates the Vulkan device, scene acceleration structures, and ray pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if the scene is invalid, a required Vulkan extension is absent, or any
    /// Vulkan resource cannot be created.
    pub fn new(width: u32, height: u32, scene: &ChessScene) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("render dimensions must be non-zero");
        }
        if !scene.validate() {
            bail!("scene contains invalid geometry");
        }

        let context = VulkanContext::new()?;
        let command_pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: context.queue_family,
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            ..Default::default()
        };
        // SAFETY: context.device is valid and the queue family exists.
        let command_pool = unsafe { context.device.create_command_pool(&command_pool_info, None) }
            .context("could not create the Vulkan command pool")?;

        let vertex_buffer = create_buffer(
            &context,
            byte_len(&scene.vertices),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        upload(&context, vertex_buffer, 0, &scene.vertices)?;
        let index_buffer = create_buffer(
            &context,
            byte_len(&scene.indices),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        upload(&context, index_buffer, 0, &scene.indices)?;
        let triangle_material_buffer = create_buffer(
            &context,
            byte_len(&scene.triangle_materials),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        upload(
            &context,
            triangle_material_buffer,
            0,
            &scene.triangle_materials,
        )?;
        let material_buffer = create_buffer(
            &context,
            (size_of::<GpuMaterial>() * 6) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let uniform_buffer = create_buffer(
            &context,
            size_of::<FrameUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let readback_buffer = create_buffer(
            &context,
            u64::from(width) * u64::from(height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let blas = build_blas(
            &context,
            command_pool,
            vertex_buffer,
            scene.vertices.len() as u32,
            index_buffer,
            scene.indices.len() as u32,
        )?;
        let tlas = build_tlas(&context, command_pool, blas)?;
        let output = create_output_image(&context, command_pool, width, height)?;

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: descriptor_pool_sizes.len() as u32,
            p_pool_sizes: descriptor_pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: descriptor pool sizes are valid and live for the call.
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .context("could not create the descriptor pool")?;

        let raygen = vk::ShaderStageFlags::RAYGEN_KHR;
        let closest_hit = vk::ShaderStageFlags::CLOSEST_HIT_KHR;
        let bindings = [
            descriptor_binding(
                0,
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                raygen | closest_hit,
            ),
            descriptor_binding(1, vk::DescriptorType::STORAGE_IMAGE, raygen),
            descriptor_binding(2, vk::DescriptorType::UNIFORM_BUFFER, raygen | closest_hit),
            descriptor_binding(3, vk::DescriptorType::STORAGE_BUFFER, closest_hit),
            descriptor_binding(4, vk::DescriptorType::STORAGE_BUFFER, closest_hit),
            descriptor_binding(5, vk::DescriptorType::STORAGE_BUFFER, closest_hit),
            descriptor_binding(6, vk::DescriptorType::STORAGE_BUFFER, closest_hit),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        // SAFETY: bindings are valid and live for the call.
        let descriptor_set_layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&layout_info, None)
        }
        .context("could not create the descriptor set layout")?;
        let set_layouts = [descriptor_set_layout];
        let set_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        // SAFETY: pool and set layout are valid.
        let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&set_alloc_info) }
            .context("could not allocate the ray-tracing descriptor set")?[0];

        update_descriptors(
            &context,
            descriptor_set,
            tlas,
            output,
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            triangle_material_buffer,
            material_buffer,
        );

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        // SAFETY: the descriptor set layout is valid.
        let pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .context("could not create the ray-tracing pipeline layout")?;
        let pipeline_recursion_depth = context.rt_properties.max_ray_recursion_depth.min(10);
        if pipeline_recursion_depth < 3 {
            bail!("the selected GPU does not support recursive ray tracing");
        }
        let pipeline = create_pipeline(&context, pipeline_layout, pipeline_recursion_depth)?;
        let (shader_binding_table, sbt_regions) = create_shader_binding_table(&context, pipeline)?;

        Ok(Self {
            context,
            command_pool,
            vertex_buffer,
            index_buffer,
            triangle_material_buffer,
            material_buffer,
            uniform_buffer,
            readback_buffer,
            shader_binding_table,
            blas,
            tlas,
            output,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            sbt_regions,
            width,
            height,
            frame_index: 0,
            pipeline_recursion_depth,
        })
    }

    /// Recreates only the resolution-dependent render resources.
    ///
    /// The ray-tracing pipeline and acceleration structures are intentionally retained, so an
    /// interactive window resize does not rebuild the scene geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if the new output image or readback buffer cannot be allocated.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            bail!("render dimensions must be non-zero");
        }
        if width == self.width && height == self.height {
            return Ok(());
        }

        let new_readback = create_buffer(
            &self.context,
            u64::from(width) * u64::from(height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let new_output = match create_output_image(&self.context, self.command_pool, width, height)
        {
            Ok(output) => output,
            Err(error) => {
                // SAFETY: this newly allocated buffer has never been submitted to the GPU.
                unsafe { destroy_buffer(&self.context.device, new_readback) };
                return Err(error);
            }
        };

        // Render submission is synchronous today, but explicitly waiting here also keeps this
        // method correct if submission becomes asynchronous later.
        if let Err(error) = unsafe { self.context.device.device_wait_idle() } {
            // SAFETY: both replacement resources are unused and belong to this device.
            unsafe {
                destroy_image(&self.context.device, new_output);
                destroy_buffer(&self.context.device, new_readback);
            }
            bail!("could not idle the Vulkan device before resize: {error:?}");
        }

        update_descriptors(
            &self.context,
            self.descriptor_set,
            self.tlas,
            new_output,
            self.uniform_buffer,
            self.vertex_buffer,
            self.index_buffer,
            self.triangle_material_buffer,
            self.material_buffer,
        );
        let old_output = std::mem::replace(&mut self.output, new_output);
        let old_readback = std::mem::replace(&mut self.readback_buffer, new_readback);
        self.width = width;
        self.height = height;
        self.frame_index = 0;
        log::info!("resized RTX output to {width}x{height}");

        // SAFETY: the device is idle and the descriptor now points at the replacement image.
        unsafe {
            destroy_image(&self.context.device, old_output);
            destroy_buffer(&self.context.device, old_readback);
        }
        Ok(())
    }

    /// Traces and reads back one RGBA8 frame with the supplied camera and materials.
    ///
    /// # Errors
    ///
    /// Returns an error if an upload, queue submission, ray dispatch, or readback fails.
    pub fn render(&mut self, settings: &RenderSettings) -> Result<Vec<u8>> {
        let (position, forward, right, up) = settings.camera_basis();
        let uniform = FrameUniform {
            camera_position: position.extend(1.0).to_array(),
            camera_forward: forward.extend(0.0).to_array(),
            camera_right: right.extend(0.0).to_array(),
            camera_up: up.extend(0.0).to_array(),
            light_position: [-4.8, 8.5, 5.2, 85.0],
            render: [
                settings.exposure,
                settings.light_radius,
                settings.samples.clamp(1, 16) as f32,
                settings
                    .max_bounces
                    .clamp(1, self.pipeline_recursion_depth - 2) as f32,
            ],
            effects: [
                f32::from(settings.reflections),
                f32::from(settings.refractions),
                f32::from(settings.caustics),
                f32::from(settings.soft_shadows),
            ],
            resolution: [
                self.width as f32,
                self.height as f32,
                self.frame_index as f32,
                f32::from(settings.board_style == BoardStyle::BrushedMetal),
            ],
        };
        upload(&self.context, self.uniform_buffer, 0, &[uniform])?;
        upload(
            &self.context,
            self.material_buffer,
            0,
            &settings.materials(),
        )?;

        submit_immediate(&self.context, self.command_pool, |command_buffer| {
            // SAFETY: all bound objects are valid, compatible, and kept alive until queue idle.
            unsafe {
                self.context.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline,
                );
                self.context.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.context.ray_tracing_pipeline.cmd_trace_rays(
                    command_buffer,
                    &self.sbt_regions[0],
                    &self.sbt_regions[1],
                    &self.sbt_regions[2],
                    &self.sbt_regions[3],
                    self.width,
                    self.height,
                    1,
                );

                let range = color_subresource_range();
                let to_transfer = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image: self.output.image,
                    subresource_range: range,
                    ..Default::default()
                };
                self.context.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                let copy = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    },
                };
                self.context.device.cmd_copy_image_to_buffer(
                    command_buffer,
                    self.output.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback_buffer.buffer,
                    &[copy],
                );
                let to_host = vk::BufferMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::HOST_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: self.readback_buffer.buffer,
                    offset: 0,
                    size: self.readback_buffer.size,
                    ..Default::default()
                };
                self.context.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[to_host],
                    &[],
                );
                let to_general = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    image: self.output.image,
                    subresource_range: range,
                    ..Default::default()
                };
                self.context.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
        })?;

        let byte_count = (u64::from(self.width) * u64::from(self.height) * 4) as usize;
        // SAFETY: readback memory is HOST_VISIBLE, the queue is idle, and byte_count is in range.
        let mapped = unsafe {
            self.context.device.map_memory(
                self.readback_buffer.memory,
                0,
                byte_count as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("could not map the rendered image")?;
        // SAFETY: mapped points to at least byte_count initialized bytes after the copy.
        let pixels =
            unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), byte_count) }.to_vec();
        // SAFETY: mapped is the active mapping of this allocation.
        unsafe {
            self.context
                .device
                .unmap_memory(self.readback_buffer.memory);
        };
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(pixels)
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.context.device_name
    }

    #[must_use]
    pub const fn dimensions(&self) -> [usize; 2] {
        [self.width as usize, self.height as usize]
    }
}

impl Drop for RayTracer {
    fn drop(&mut self) {
        // SAFETY: the device owns every handle below; waiting makes all resources idle before
        // destruction. Each handle is destroyed exactly once in dependency order.
        unsafe {
            let _ = self.context.device.device_wait_idle();
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            destroy_image(&self.context.device, self.output);
            self.context
                .acceleration_structure
                .destroy_acceleration_structure(self.tlas.handle, None);
            destroy_buffer(&self.context.device, self.tlas.storage);
            self.context
                .acceleration_structure
                .destroy_acceleration_structure(self.blas.handle, None);
            destroy_buffer(&self.context.device, self.blas.storage);
            for buffer in [
                self.shader_binding_table,
                self.readback_buffer,
                self.uniform_buffer,
                self.material_buffer,
                self.triangle_material_buffer,
                self.index_buffer,
                self.vertex_buffer,
            ] {
                destroy_buffer(&self.context.device, buffer);
            }
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

fn byte_len<T>(slice: &[T]) -> u64 {
    std::mem::size_of_val(slice) as u64
}

fn descriptor_binding(
    binding: u32,
    descriptor_type: vk::DescriptorType,
    stage_flags: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding {
        binding,
        descriptor_type,
        descriptor_count: 1,
        stage_flags,
        ..Default::default()
    }
}

fn create_buffer(
    context: &VulkanContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_properties: vk::MemoryPropertyFlags,
) -> Result<BufferResource> {
    let size = size.max(4);
    let buffer_info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    // SAFETY: buffer_info contains no borrowed pointer fields.
    let buffer = unsafe { context.device.create_buffer(&buffer_info, None) }
        .context("vkCreateBuffer failed")?;
    // SAFETY: buffer is valid and was created on this device.
    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
    let memory_type = context.find_memory_type(requirements.memory_type_bits, memory_properties)?;
    let needs_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let mut allocate_flags = vk::MemoryAllocateFlagsInfo {
        flags: if needs_address {
            vk::MemoryAllocateFlags::DEVICE_ADDRESS
        } else {
            vk::MemoryAllocateFlags::empty()
        },
        ..Default::default()
    };
    let allocation_info = vk::MemoryAllocateInfo {
        p_next: if needs_address {
            (&mut allocate_flags as *mut _) as *const _
        } else {
            ptr::null()
        },
        allocation_size: requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };
    // SAFETY: allocation_info and its optional pNext are valid for the call.
    let memory = unsafe { context.device.allocate_memory(&allocation_info, None) }
        .context("vkAllocateMemory failed for a buffer")?;
    // SAFETY: memory satisfies this buffer's requirements and offset zero is aligned.
    unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }
        .context("vkBindBufferMemory failed")?;
    let address = if needs_address {
        let address_info = vk::BufferDeviceAddressInfo {
            buffer,
            ..Default::default()
        };
        // SAFETY: buffer was created with SHADER_DEVICE_ADDRESS and suitable memory.
        unsafe { context.device.get_buffer_device_address(&address_info) }
    } else {
        0
    };
    Ok(BufferResource {
        buffer,
        memory,
        address,
        size,
    })
}

fn upload<T: Copy>(
    context: &VulkanContext,
    resource: BufferResource,
    byte_offset: u64,
    values: &[T],
) -> Result<()> {
    let byte_count = byte_len(values);
    if byte_offset + byte_count > resource.size {
        bail!("buffer upload exceeds allocation");
    }
    // SAFETY: callers only upload into HOST_VISIBLE allocations and the range is checked above.
    let mapped_size = byte_offset + byte_count;
    let mapped = unsafe {
        context
            .device
            .map_memory(resource.memory, 0, mapped_size, vk::MemoryMapFlags::empty())
    }
    .context("could not map a Vulkan upload buffer")?;
    // SAFETY: mapped has byte_count writable bytes and values has exactly byte_count readable bytes.
    unsafe {
        ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>().add(byte_offset as usize),
            byte_count as usize,
        );
        context.device.unmap_memory(resource.memory);
    }
    Ok(())
}

unsafe fn destroy_buffer(device: &Device, resource: BufferResource) {
    // SAFETY: caller guarantees the resource belongs to device and is no longer in use.
    unsafe {
        device.destroy_buffer(resource.buffer, None);
        device.free_memory(resource.memory, None);
    }
}

unsafe fn destroy_image(device: &Device, resource: ImageResource) {
    // SAFETY: caller guarantees the resource belongs to device and is no longer in use.
    unsafe {
        device.destroy_image_view(resource.view, None);
        device.destroy_image(resource.image, None);
        device.free_memory(resource.memory, None);
    }
}

fn submit_immediate(
    context: &VulkanContext,
    command_pool: vk::CommandPool,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<()> {
    let allocation_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    // SAFETY: command_pool belongs to the device.
    let command_buffer = unsafe { context.device.allocate_command_buffers(&allocation_info) }
        .context("could not allocate a Vulkan command buffer")?[0];
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    // SAFETY: command_buffer is newly allocated and in the initial state.
    unsafe {
        context
            .device
            .begin_command_buffer(command_buffer, &begin_info)
    }
    .context("could not begin a Vulkan command buffer")?;
    record(command_buffer);
    // SAFETY: the callback records complete, valid commands.
    unsafe { context.device.end_command_buffer(command_buffer) }
        .context("could not end a Vulkan command buffer")?;
    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: command_buffers.as_ptr(),
        ..Default::default()
    };
    // SAFETY: command buffer remains alive until queue_wait_idle returns.
    unsafe {
        context
            .device
            .queue_submit(context.queue, &[submit], vk::Fence::null())
    }
    .context("Vulkan queue submission failed")?;
    // SAFETY: queue belongs to the device.
    unsafe { context.device.queue_wait_idle(context.queue) }
        .context("Vulkan queue did not become idle")?;
    // SAFETY: execution is complete, so the command buffer can be freed.
    unsafe {
        context
            .device
            .free_command_buffers(command_pool, &command_buffers);
    };
    Ok(())
}

fn build_blas(
    context: &VulkanContext,
    command_pool: vk::CommandPool,
    vertices: BufferResource,
    vertex_count: u32,
    indices: BufferResource,
    index_count: u32,
) -> Result<AccelerationResource> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR {
        vertex_format: vk::Format::R32G32B32_SFLOAT,
        vertex_data: vk::DeviceOrHostAddressConstKHR {
            device_address: vertices.address,
        },
        vertex_stride: size_of::<Vertex>() as u64,
        max_vertex: vertex_count.saturating_sub(1),
        index_type: vk::IndexType::UINT32,
        index_data: vk::DeviceOrHostAddressConstKHR {
            device_address: indices.address,
        },
        ..Default::default()
    };
    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::TRIANGLES,
        geometry: vk::AccelerationStructureGeometryDataKHR { triangles },
        flags: vk::GeometryFlagsKHR::OPAQUE,
        ..Default::default()
    };
    let geometry_list = [geometry];
    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: geometry_list.as_ptr(),
        ..Default::default()
    };
    let primitive_count = index_count / 3;
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: build_info describes one valid triangles geometry.
    unsafe {
        context
            .acceleration_structure
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[primitive_count],
                &mut sizes,
            );
    }
    let storage = create_buffer(
        context,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let create_info = vk::AccelerationStructureCreateInfoKHR {
        buffer: storage.buffer,
        size: sizes.acceleration_structure_size,
        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        ..Default::default()
    };
    // SAFETY: storage buffer has the correct usage and sufficient size.
    let handle = unsafe {
        context
            .acceleration_structure
            .create_acceleration_structure(&create_info, None)
    }
    .context("could not create the bottom-level acceleration structure")?;
    let scratch = create_buffer(
        context,
        sizes.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    build_info.dst_acceleration_structure = handle;
    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch.address,
    };
    let range = vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    };
    submit_immediate(context, command_pool, |command_buffer| {
        // SAFETY: geometry buffers, scratch buffer, destination AS, and range remain valid.
        unsafe {
            context
                .acceleration_structure
                .cmd_build_acceleration_structures(command_buffer, &[build_info], &[&[range]]);
        }
    })?;
    // SAFETY: the queue is idle and the scratch buffer is no longer needed.
    unsafe { destroy_buffer(&context.device, scratch) };
    log::info!(
        "BLAS built: {vertex_count} vertices, {} triangles",
        index_count / 3
    );
    Ok(AccelerationResource { handle, storage })
}

fn build_tlas(
    context: &VulkanContext,
    command_pool: vk::CommandPool,
    blas: AccelerationResource,
) -> Result<AccelerationResource> {
    let address_info = vk::AccelerationStructureDeviceAddressInfoKHR {
        acceleration_structure: blas.handle,
        ..Default::default()
    };
    // SAFETY: BLAS is built and valid.
    let blas_address = unsafe {
        context
            .acceleration_structure
            .get_acceleration_structure_device_address(&address_info)
    };
    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    };
    let instances = create_buffer(
        context,
        size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    upload(context, instances, 0, &[instance])?;
    let instance_data = vk::AccelerationStructureGeometryInstancesDataKHR {
        data: vk::DeviceOrHostAddressConstKHR {
            device_address: instances.address,
        },
        ..Default::default()
    };
    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: instance_data,
        },
        ..Default::default()
    };
    let geometries = [geometry];
    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: geometries.as_ptr(),
        ..Default::default()
    };
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: build_info describes one valid instance geometry.
    unsafe {
        context
            .acceleration_structure
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[1],
                &mut sizes,
            );
    }
    let storage = create_buffer(
        context,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let create_info = vk::AccelerationStructureCreateInfoKHR {
        buffer: storage.buffer,
        size: sizes.acceleration_structure_size,
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        ..Default::default()
    };
    // SAFETY: storage is a sufficiently large AS storage buffer.
    let handle = unsafe {
        context
            .acceleration_structure
            .create_acceleration_structure(&create_info, None)
    }
    .context("could not create the top-level acceleration structure")?;
    let scratch = create_buffer(
        context,
        sizes.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    build_info.dst_acceleration_structure = handle;
    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch.address,
    };
    let range = vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count: 1,
        ..Default::default()
    };
    submit_immediate(context, command_pool, |command_buffer| {
        // SAFETY: BLAS, instance and scratch data all remain alive for this build.
        unsafe {
            context
                .acceleration_structure
                .cmd_build_acceleration_structures(command_buffer, &[build_info], &[&[range]]);
        }
    })?;
    // SAFETY: the queue is idle and these build inputs are no longer needed.
    unsafe {
        destroy_buffer(&context.device, scratch);
        destroy_buffer(&context.device, instances);
    }
    Ok(AccelerationResource { handle, storage })
}

fn create_output_image(
    context: &VulkanContext,
    command_pool: vk::CommandPool,
    width: u32,
    height: u32,
) -> Result<ImageResource> {
    let image_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    // SAFETY: image_info is self-contained and valid.
    let image = unsafe { context.device.create_image(&image_info, None) }
        .context("could not create the RTX output image")?;
    // SAFETY: image belongs to the device.
    let requirements = unsafe { context.device.get_image_memory_requirements(image) };
    let memory_type = context.find_memory_type(
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let allocation_info = vk::MemoryAllocateInfo {
        allocation_size: requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };
    // SAFETY: allocation parameters satisfy the image requirements.
    let memory = unsafe { context.device.allocate_memory(&allocation_info, None) }
        .context("could not allocate the RTX output image")?;
    // SAFETY: offset zero satisfies the returned alignment requirement.
    unsafe { context.device.bind_image_memory(image, memory, 0) }
        .context("could not bind the RTX output image")?;
    let view_info = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        subresource_range: color_subresource_range(),
        ..Default::default()
    };
    // SAFETY: view_info addresses the image's only color subresource.
    let view = unsafe { context.device.create_image_view(&view_info, None) }
        .context("could not create the RTX output image view")?;
    submit_immediate(context, command_pool, |command_buffer| {
        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            image,
            subresource_range: color_subresource_range(),
            ..Default::default()
        };
        // SAFETY: this is the first use of the image and the range is valid.
        unsafe {
            context.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    })?;
    Ok(ImageResource {
        image,
        memory,
        view,
    })
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_descriptors(
    context: &VulkanContext,
    descriptor_set: vk::DescriptorSet,
    tlas: AccelerationResource,
    output: ImageResource,
    uniform: BufferResource,
    vertices: BufferResource,
    indices: BufferResource,
    triangle_materials: BufferResource,
    materials: BufferResource,
) {
    let acceleration_structures = [tlas.handle];
    let mut tlas_info = vk::WriteDescriptorSetAccelerationStructureKHR {
        acceleration_structure_count: 1,
        p_acceleration_structures: acceleration_structures.as_ptr(),
        ..Default::default()
    };
    let image_info = [vk::DescriptorImageInfo {
        image_view: output.view,
        image_layout: vk::ImageLayout::GENERAL,
        ..Default::default()
    }];
    let uniform_info = [buffer_descriptor(uniform)];
    let vertex_info = [buffer_descriptor(vertices)];
    let index_info = [buffer_descriptor(indices)];
    let triangle_material_info = [buffer_descriptor(triangle_materials)];
    let material_info = [buffer_descriptor(materials)];
    let writes = [
        vk::WriteDescriptorSet {
            p_next: (&mut tlas_info as *mut _) as *const _,
            dst_set: descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            ..Default::default()
        },
        image_descriptor(descriptor_set, 1, &image_info),
        buffer_write(
            descriptor_set,
            2,
            vk::DescriptorType::UNIFORM_BUFFER,
            &uniform_info,
        ),
        buffer_write(
            descriptor_set,
            3,
            vk::DescriptorType::STORAGE_BUFFER,
            &vertex_info,
        ),
        buffer_write(
            descriptor_set,
            4,
            vk::DescriptorType::STORAGE_BUFFER,
            &index_info,
        ),
        buffer_write(
            descriptor_set,
            5,
            vk::DescriptorType::STORAGE_BUFFER,
            &triangle_material_info,
        ),
        buffer_write(
            descriptor_set,
            6,
            vk::DescriptorType::STORAGE_BUFFER,
            &material_info,
        ),
    ];
    // SAFETY: descriptor resources and backing info arrays remain valid for this immediate call.
    unsafe { context.device.update_descriptor_sets(&writes, &[]) };
}

fn buffer_descriptor(resource: BufferResource) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo {
        buffer: resource.buffer,
        offset: 0,
        range: resource.size,
    }
}

fn buffer_write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    info: &[vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet {
        dst_set: set,
        dst_binding: binding,
        descriptor_count: 1,
        descriptor_type: ty,
        p_buffer_info: info.as_ptr(),
        ..Default::default()
    }
}

fn image_descriptor(
    set: vk::DescriptorSet,
    binding: u32,
    info: &[vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet {
        dst_set: set,
        dst_binding: binding,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
        p_image_info: info.as_ptr(),
        ..Default::default()
    }
}

fn create_pipeline(
    context: &VulkanContext,
    layout: vk::PipelineLayout,
    recursion_depth: u32,
) -> Result<vk::Pipeline> {
    let binaries = [
        include_bytes!(concat!(env!("OUT_DIR"), "/raygen.rgen.spv")).as_slice(),
        include_bytes!(concat!(env!("OUT_DIR"), "/miss.rmiss.spv")).as_slice(),
        include_bytes!(concat!(env!("OUT_DIR"), "/shadow.rmiss.spv")).as_slice(),
        include_bytes!(concat!(env!("OUT_DIR"), "/closesthit.rchit.spv")).as_slice(),
    ];
    let mut modules = Vec::with_capacity(binaries.len());
    for bytes in binaries {
        let words = ash::util::read_spv(&mut Cursor::new(bytes))
            .context("compiled ray shader did not contain valid SPIR-V")?;
        let module_info = vk::ShaderModuleCreateInfo {
            code_size: words.len() * size_of::<u32>(),
            p_code: words.as_ptr(),
            ..Default::default()
        };
        // SAFETY: words contains SPIR-V emitted for Vulkan 1.2 by shaderc.
        modules.push(
            unsafe { context.device.create_shader_module(&module_info, None) }
                .context("could not create a ray-tracing shader module")?,
        );
    }
    let entry = c"main";
    let stages = [
        shader_stage(vk::ShaderStageFlags::RAYGEN_KHR, modules[0], entry),
        shader_stage(vk::ShaderStageFlags::MISS_KHR, modules[1], entry),
        shader_stage(vk::ShaderStageFlags::MISS_KHR, modules[2], entry),
        shader_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, modules[3], entry),
    ];
    let unused = vk::SHADER_UNUSED_KHR;
    let groups = [
        shader_group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 0, unused),
        shader_group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 1, unused),
        shader_group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 2, unused),
        shader_group(
            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
            unused,
            3,
        ),
    ];
    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR {
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        group_count: groups.len() as u32,
        p_groups: groups.as_ptr(),
        max_pipeline_ray_recursion_depth: recursion_depth,
        layout,
        ..Default::default()
    };
    // SAFETY: stage modules, groups, layout, and all borrowed arrays are valid for the call.
    let result = unsafe {
        context.ray_tracing_pipeline.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        )
    };
    // SAFETY: shader modules are no longer needed after pipeline creation, including on failure.
    unsafe {
        for module in modules {
            context.device.destroy_shader_module(module, None);
        }
    }
    result
        .map(|pipelines| pipelines[0])
        .map_err(|(_, error)| anyhow!("vkCreateRayTracingPipelinesKHR failed: {error:?}"))
}

fn shader_stage(
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    entry: &CStr,
) -> vk::PipelineShaderStageCreateInfo<'_> {
    vk::PipelineShaderStageCreateInfo {
        stage,
        module,
        p_name: entry.as_ptr(),
        ..Default::default()
    }
}

fn shader_group(
    ty: vk::RayTracingShaderGroupTypeKHR,
    general_shader: u32,
    closest_hit_shader: u32,
) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    vk::RayTracingShaderGroupCreateInfoKHR {
        ty,
        general_shader,
        closest_hit_shader,
        any_hit_shader: vk::SHADER_UNUSED_KHR,
        intersection_shader: vk::SHADER_UNUSED_KHR,
        ..Default::default()
    }
}

fn create_shader_binding_table(
    context: &VulkanContext,
    pipeline: vk::Pipeline,
) -> Result<(BufferResource, [vk::StridedDeviceAddressRegionKHR; 4])> {
    let properties = &context.rt_properties;
    let handle_size = u64::from(properties.shader_group_handle_size);
    let handle_alignment = u64::from(properties.shader_group_handle_alignment);
    let base_alignment = u64::from(properties.shader_group_base_alignment);
    let handle_stride = align_up(handle_size, handle_alignment);
    let raygen_stride = align_up(handle_stride, base_alignment);
    let raygen_offset = 0;
    let miss_offset = raygen_stride;
    let miss_size = align_up(handle_stride * 2, base_alignment);
    let hit_offset = miss_offset + miss_size;
    let hit_size = handle_stride;
    let logical_size = hit_offset + hit_size;
    let allocation_size = logical_size + base_alignment;
    let buffer = create_buffer(
        context,
        allocation_size,
        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let base_padding = align_up(buffer.address, base_alignment) - buffer.address;

    // SAFETY: pipeline has RAY_GROUP_COUNT groups and the output size matches handle size.
    let handles = unsafe {
        context
            .ray_tracing_pipeline
            .get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                RAY_GROUP_COUNT,
                (handle_size * u64::from(RAY_GROUP_COUNT)) as usize,
            )
    }
    .context("could not obtain ray-tracing shader group handles")?;
    let mut table = vec![0_u8; logical_size as usize];
    let offsets = [
        raygen_offset,
        miss_offset,
        miss_offset + handle_stride,
        hit_offset,
    ];
    for (group, &offset) in offsets.iter().enumerate() {
        let source = group * handle_size as usize;
        table[offset as usize..offset as usize + handle_size as usize]
            .copy_from_slice(&handles[source..source + handle_size as usize]);
    }
    upload(context, buffer, base_padding, &table)?;
    let address = buffer.address + base_padding;
    let regions = [
        vk::StridedDeviceAddressRegionKHR {
            device_address: address + raygen_offset,
            stride: raygen_stride,
            size: raygen_stride,
        },
        vk::StridedDeviceAddressRegionKHR {
            device_address: address + miss_offset,
            stride: handle_stride,
            size: handle_stride * 2,
        },
        vk::StridedDeviceAddressRegionKHR {
            device_address: address + hit_offset,
            stride: handle_stride,
            size: hit_size,
        },
        vk::StridedDeviceAddressRegionKHR::default(),
    ];
    Ok((buffer, regions))
}

const fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}
