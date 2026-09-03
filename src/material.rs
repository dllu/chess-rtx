use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    /// roughness, metallic, transmission, index of refraction
    pub surface: [f32; 4],
    /// anisotropy, caustic strength, clear-coat, reserved
    pub optics: [f32; 4],
}

impl GpuMaterial {
    #[must_use]
    pub const fn new(
        base_color: [f32; 4],
        roughness: f32,
        metallic: f32,
        transmission: f32,
        ior: f32,
        anisotropy: f32,
        caustics: f32,
        clear_coat: f32,
    ) -> Self {
        Self {
            base_color,
            surface: [roughness, metallic, transmission, ior],
            optics: [anisotropy, caustics, clear_coat, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceStyle {
    Ceramic,
    Metal,
    FrostedGlass,
}

impl PieceStyle {
    pub const ALL: [Self; 3] = [Self::Ceramic, Self::Metal, Self::FrostedGlass];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ceramic => "Ceramic",
            Self::Metal => "Anisotropic metal",
            Self::FrostedGlass => "Frosted glass",
        }
    }

    #[must_use]
    pub fn material(self, light: bool) -> GpuMaterial {
        match (self, light) {
            (Self::Ceramic, true) => GpuMaterial::new(
                [0.94, 0.91, 0.82, 1.0],
                0.16,
                0.02,
                0.0,
                1.48,
                0.0,
                0.0,
                0.55,
            ),
            (Self::Ceramic, false) => GpuMaterial::new(
                [0.045, 0.055, 0.072, 1.0],
                0.12,
                0.08,
                0.0,
                1.5,
                0.0,
                0.0,
                0.7,
            ),
            (Self::Metal, true) => GpuMaterial::new(
                [0.82, 0.63, 0.24, 1.0],
                0.14,
                0.96,
                0.0,
                1.5,
                0.72,
                0.0,
                0.35,
            ),
            (Self::Metal, false) => GpuMaterial::new(
                [0.12, 0.17, 0.22, 1.0],
                0.1,
                0.98,
                0.0,
                1.5,
                -0.62,
                0.0,
                0.25,
            ),
            (Self::FrostedGlass, true) => GpuMaterial::new(
                [0.77, 0.91, 0.97, 1.0],
                0.13,
                0.0,
                0.91,
                1.52,
                0.18,
                1.0,
                0.2,
            ),
            (Self::FrostedGlass, false) => GpuMaterial::new(
                [0.19, 0.31, 0.42, 1.0],
                0.19,
                0.03,
                0.88,
                1.58,
                -0.2,
                1.0,
                0.15,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoardStyle {
    Marble,
    BrushedMetal,
}

impl BoardStyle {
    pub const ALL: [Self; 2] = [Self::Marble, Self::BrushedMetal];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Marble => "Polished marble",
            Self::BrushedMetal => "Brushed metal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub light_material: GpuMaterial,
    pub dark_material: GpuMaterial,
    pub light_style: PieceStyle,
    pub dark_style: PieceStyle,
    pub board_style: BoardStyle,
    pub light_square: [f32; 3],
    pub dark_square: [f32; 3],
    pub board_roughness: f32,
    pub board_metallic: f32,
    pub reflections: bool,
    pub refractions: bool,
    pub caustics: bool,
    pub soft_shadows: bool,
    pub exposure: f32,
    pub light_radius: f32,
    pub samples: u32,
    pub max_bounces: u32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub camera_target_height: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            light_material: PieceStyle::Ceramic.material(true),
            dark_material: PieceStyle::Metal.material(false),
            light_style: PieceStyle::Ceramic,
            dark_style: PieceStyle::Metal,
            board_style: BoardStyle::Marble,
            light_square: [0.72, 0.69, 0.62],
            dark_square: [0.095, 0.15, 0.16],
            board_roughness: 0.075,
            board_metallic: 0.05,
            reflections: true,
            refractions: true,
            caustics: true,
            soft_shadows: true,
            exposure: 1.15,
            light_radius: 1.25,
            samples: 2,
            max_bounces: 4,
            camera_yaw: 38.0_f32.to_radians(),
            camera_pitch: 27.0_f32.to_radians(),
            camera_distance: 12.8,
            camera_target_height: 0.65,
        }
    }
}

impl RenderSettings {
    pub fn set_light_style(&mut self, style: PieceStyle) {
        self.light_style = style;
        self.light_material = style.material(true);
    }

    pub fn set_dark_style(&mut self, style: PieceStyle) {
        self.dark_style = style;
        self.dark_material = style.material(false);
    }

    pub fn set_board_style(&mut self, style: BoardStyle) {
        self.board_style = style;
        match style {
            BoardStyle::Marble => {
                self.light_square = [0.72, 0.69, 0.62];
                self.dark_square = [0.095, 0.15, 0.16];
                self.board_roughness = 0.075;
                self.board_metallic = 0.05;
            }
            BoardStyle::BrushedMetal => {
                self.light_square = [0.53, 0.56, 0.59];
                self.dark_square = [0.075, 0.09, 0.115];
                self.board_roughness = 0.19;
                self.board_metallic = 0.9;
            }
        }
    }

    #[must_use]
    pub fn materials(&self) -> [GpuMaterial; 6] {
        let board_anisotropy = if self.board_style == BoardStyle::BrushedMetal {
            0.78
        } else {
            0.0
        };
        let board_metallic = self.board_metallic;
        [
            self.light_material,
            self.dark_material,
            GpuMaterial::new(
                [
                    self.light_square[0],
                    self.light_square[1],
                    self.light_square[2],
                    1.0,
                ],
                self.board_roughness,
                board_metallic,
                0.0,
                1.5,
                board_anisotropy,
                0.0,
                0.22,
            ),
            GpuMaterial::new(
                [
                    self.dark_square[0],
                    self.dark_square[1],
                    self.dark_square[2],
                    1.0,
                ],
                self.board_roughness,
                board_metallic,
                0.0,
                1.5,
                -board_anisotropy,
                0.0,
                0.22,
            ),
            GpuMaterial::new(
                [0.12, 0.085, 0.048, 1.0],
                0.15,
                0.7,
                0.0,
                1.5,
                0.42,
                0.0,
                0.5,
            ),
            GpuMaterial::new(
                [0.025, 0.028, 0.034, 1.0],
                0.32,
                0.12,
                0.0,
                1.5,
                0.0,
                0.0,
                0.1,
            ),
        ]
    }

    #[must_use]
    pub fn camera_basis(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let target = Vec3::new(0.0, self.camera_target_height, 0.0);
        let orbit = Vec3::new(
            self.camera_pitch.cos() * self.camera_yaw.sin(),
            self.camera_pitch.sin(),
            self.camera_pitch.cos() * self.camera_yaw.cos(),
        );
        let position = target + orbit * self.camera_distance;
        let forward = (target - position).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        (position, forward, right, up)
    }
}
