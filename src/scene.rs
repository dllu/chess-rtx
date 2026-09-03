use bytemuck::{Pod, Zeroable};
use chess::{Board, Color, File, Piece, Rank, Square};
use glam::{Mat3, Vec2, Vec3};

use crate::assets::{self, PieceMesh};

pub const MATERIAL_LIGHT_PIECE: u32 = 0;
pub const MATERIAL_DARK_PIECE: u32 = 1;
pub const MATERIAL_LIGHT_SQUARE: u32 = 2;
pub const MATERIAL_DARK_SQUARE: u32 = 3;
pub const MATERIAL_BORDER: u32 = 4;
pub const MATERIAL_TABLE: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct ChessScene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub triangle_materials: Vec<u32>,
    pub detailed_piece_meshes: usize,
}

impl ChessScene {
    #[must_use]
    pub fn from_board(board: &Board) -> Self {
        let mut builder = SceneBuilder::default();
        builder.add_box(
            Vec3::new(0.0, -0.07, 0.0),
            Vec3::new(8.8, 0.28, 8.8),
            MATERIAL_BORDER,
        );

        for rank in 0..8 {
            for file in 0..8 {
                let material = if (rank + file) % 2 == 0 {
                    MATERIAL_DARK_SQUARE
                } else {
                    MATERIAL_LIGHT_SQUARE
                };
                builder.add_box(
                    board_position(file, rank, 0.105),
                    Vec3::new(0.985, 0.075, 0.985),
                    material,
                );
            }
        }

        builder.add_box(
            Vec3::new(0.0, -0.72, 0.0),
            Vec3::new(18.0, 0.95, 14.0),
            MATERIAL_TABLE,
        );

        for rank in 0..8 {
            for file in 0..8 {
                let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
                let (Some(piece), Some(color)) = (board.piece_on(square), board.color_on(square))
                else {
                    continue;
                };
                let position = board_position(file, rank, 0.15);
                let material = if color == Color::White {
                    MATERIAL_LIGHT_PIECE
                } else {
                    MATERIAL_DARK_PIECE
                };
                builder.add_piece(piece, color, position, material);
            }
        }

        Self {
            vertices: builder.vertices,
            indices: builder.indices,
            triangle_materials: builder.triangle_materials,
            detailed_piece_meshes: assets::loaded_piece_mesh_count(),
        }
    }

    #[must_use]
    pub const fn geometry_label(&self) -> &'static str {
        match self.detailed_piece_meshes {
            0 => "procedural pieces",
            6 => "CC Staunton pieces",
            _ => "mixed piece assets",
        }
    }

    #[must_use]
    pub fn validate(&self) -> bool {
        !self.vertices.is_empty()
            && self.indices.len().is_multiple_of(3)
            && self.triangle_materials.len() == self.indices.len() / 3
            && self
                .indices
                .iter()
                .all(|&index| index < self.vertices.len() as u32)
    }
}

/// Maps chess coordinates into the scene's right-handed coordinate system.
///
/// White looks toward increasing Z. Files therefore run from positive X (a-file)
/// to negative X (h-file), which makes a through h appear left-to-right from
/// White's side with the renderer's camera basis.
fn board_position(file: usize, rank: usize, height: f32) -> Vec3 {
    Vec3::new(3.5 - file as f32, height, rank as f32 - 3.5)
}

#[derive(Default)]
struct SceneBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    triangle_materials: Vec<u32>,
}

impl SceneBuilder {
    fn vertex(position: Vec3, normal: Vec3) -> Vertex {
        Vertex {
            position: [position.x, position.y, position.z, 1.0],
            normal: [normal.x, normal.y, normal.z, 0.0],
        }
    }

    fn add_triangle(&mut self, a: Vertex, b: Vertex, c: Vertex, material: u32) {
        let base = self.vertices.len() as u32;
        self.vertices.extend([a, b, c]);
        self.indices.extend([base, base + 1, base + 2]);
        self.triangle_materials.push(material);
    }

    fn add_quad(&mut self, points: [Vec3; 4], normal: Vec3, material: u32) {
        self.add_triangle(
            Self::vertex(points[0], normal),
            Self::vertex(points[1], normal),
            Self::vertex(points[2], normal),
            material,
        );
        self.add_triangle(
            Self::vertex(points[0], normal),
            Self::vertex(points[2], normal),
            Self::vertex(points[3], normal),
            material,
        );
    }

    fn add_box(&mut self, center: Vec3, size: Vec3, material: u32) {
        let half = size * 0.5;
        let p = |x: f32, y: f32, z: f32| center + Vec3::new(x, y, z) * half;
        self.add_quad(
            [
                p(-1.0, 1.0, -1.0),
                p(-1.0, 1.0, 1.0),
                p(1.0, 1.0, 1.0),
                p(1.0, 1.0, -1.0),
            ],
            Vec3::Y,
            material,
        );
        self.add_quad(
            [
                p(-1.0, -1.0, 1.0),
                p(-1.0, -1.0, -1.0),
                p(1.0, -1.0, -1.0),
                p(1.0, -1.0, 1.0),
            ],
            Vec3::NEG_Y,
            material,
        );
        self.add_quad(
            [
                p(-1.0, -1.0, 1.0),
                p(1.0, -1.0, 1.0),
                p(1.0, 1.0, 1.0),
                p(-1.0, 1.0, 1.0),
            ],
            Vec3::Z,
            material,
        );
        self.add_quad(
            [
                p(1.0, -1.0, -1.0),
                p(-1.0, -1.0, -1.0),
                p(-1.0, 1.0, -1.0),
                p(1.0, 1.0, -1.0),
            ],
            Vec3::NEG_Z,
            material,
        );
        self.add_quad(
            [
                p(1.0, -1.0, 1.0),
                p(1.0, -1.0, -1.0),
                p(1.0, 1.0, -1.0),
                p(1.0, 1.0, 1.0),
            ],
            Vec3::X,
            material,
        );
        self.add_quad(
            [
                p(-1.0, -1.0, -1.0),
                p(-1.0, -1.0, 1.0),
                p(-1.0, 1.0, 1.0),
                p(-1.0, 1.0, -1.0),
            ],
            Vec3::NEG_X,
            material,
        );
    }

    fn add_piece(&mut self, piece: Piece, color: Color, origin: Vec3, material: u32) {
        if let Some(mesh) = assets::piece_mesh(piece) {
            self.add_piece_mesh(piece, mesh, color, origin, material);
            return;
        }

        let base = [
            Vec2::new(0.38, 0.0),
            Vec2::new(0.42, 0.06),
            Vec2::new(0.40, 0.14),
            Vec2::new(0.30, 0.22),
            Vec2::new(0.26, 0.34),
            Vec2::new(0.29, 0.42),
        ];
        self.add_lathe(&base, origin, material, 24);

        match piece {
            Piece::Pawn => {
                let profile = [
                    Vec2::new(0.24, 0.36),
                    Vec2::new(0.20, 0.56),
                    Vec2::new(0.14, 0.82),
                    Vec2::new(0.22, 0.93),
                    Vec2::new(0.24, 1.02),
                ];
                self.add_lathe(&profile, origin, material, 24);
                self.add_ellipsoid(
                    origin + Vec3::new(0.0, 1.23, 0.0),
                    Vec3::splat(0.245),
                    material,
                    20,
                    12,
                    Mat3::IDENTITY,
                );
            }
            Piece::Rook => {
                let profile = [
                    Vec2::new(0.27, 0.36),
                    Vec2::new(0.25, 0.69),
                    Vec2::new(0.29, 1.03),
                    Vec2::new(0.36, 1.12),
                    Vec2::new(0.36, 1.37),
                ];
                self.add_lathe(&profile, origin, material, 24);
                for (x, z) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                    self.add_box(
                        origin + Vec3::new(x, 1.43, z),
                        Vec3::new(0.20, 0.24, 0.20),
                        material,
                    );
                }
            }
            Piece::Knight => self.add_knight(origin, color, material),
            Piece::Bishop => {
                let profile = [
                    Vec2::new(0.27, 0.36),
                    Vec2::new(0.22, 0.68),
                    Vec2::new(0.17, 0.92),
                    Vec2::new(0.25, 1.02),
                    Vec2::new(0.18, 1.13),
                    Vec2::new(0.10, 1.42),
                    Vec2::new(0.0, 1.58),
                ];
                self.add_lathe(&profile, origin, material, 24);
                self.add_ellipsoid(
                    origin + Vec3::new(0.0, 1.34, 0.0),
                    Vec3::new(0.18, 0.28, 0.18),
                    material,
                    20,
                    12,
                    Mat3::IDENTITY,
                );
            }
            Piece::Queen => {
                let profile = [
                    Vec2::new(0.27, 0.36),
                    Vec2::new(0.22, 0.76),
                    Vec2::new(0.17, 1.12),
                    Vec2::new(0.30, 1.25),
                    Vec2::new(0.25, 1.43),
                    Vec2::new(0.14, 1.56),
                    Vec2::new(0.0, 1.67),
                ];
                self.add_lathe(&profile, origin, material, 28);
                self.add_ellipsoid(
                    origin + Vec3::new(0.0, 1.73, 0.0),
                    Vec3::splat(0.13),
                    material,
                    18,
                    10,
                    Mat3::IDENTITY,
                );
            }
            Piece::King => {
                let profile = [
                    Vec2::new(0.28, 0.36),
                    Vec2::new(0.23, 0.78),
                    Vec2::new(0.18, 1.18),
                    Vec2::new(0.31, 1.34),
                    Vec2::new(0.22, 1.51),
                    Vec2::new(0.11, 1.62),
                ];
                self.add_lathe(&profile, origin, material, 28);
                self.add_box(
                    origin + Vec3::new(0.0, 1.82, 0.0),
                    Vec3::new(0.13, 0.43, 0.13),
                    material,
                );
                self.add_box(
                    origin + Vec3::new(0.0, 1.88, 0.0),
                    Vec3::new(0.40, 0.13, 0.13),
                    material,
                );
            }
        }
    }

    fn add_piece_mesh(
        &mut self,
        piece: Piece,
        mesh: &PieceMesh,
        color: Color,
        origin: Vec3,
        material: u32,
    ) {
        let yaw = imported_piece_yaw(piece, color);
        let rotation = Mat3::from_rotation_y(yaw);
        let base = self.vertices.len() as u32;
        self.vertices.extend(mesh.vertices.iter().map(|vertex| {
            let position = Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
            let normal = Vec3::new(vertex.normal[0], vertex.normal[1], vertex.normal[2]);
            Self::vertex(
                origin + rotation * position,
                (rotation * normal).normalize_or_zero(),
            )
        }));
        self.indices
            .extend(mesh.indices.iter().map(|index| base + index));
        self.triangle_materials.resize(
            self.triangle_materials.len() + mesh.indices.len() / 3,
            material,
        );
    }

    fn add_lathe(&mut self, profile: &[Vec2], origin: Vec3, material: u32, segments: u32) {
        let start = self.vertices.len() as u32;
        for (row, point) in profile.iter().enumerate() {
            let before = profile[row.saturating_sub(1)];
            let after = profile[(row + 1).min(profile.len() - 1)];
            let slope = (after.x - before.x) / (after.y - before.y).abs().max(0.0001);
            for segment in 0..=segments {
                let angle = segment as f32 / segments as f32 * std::f32::consts::TAU;
                let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                let normal = (radial + Vec3::Y * -slope).normalize();
                let position =
                    origin + Vec3::new(point.x * angle.cos(), point.y, point.x * angle.sin());
                self.vertices.push(Self::vertex(position, normal));
            }
        }
        let stride = segments + 1;
        for row in 0..profile.len() as u32 - 1 {
            for segment in 0..segments {
                let a = start + row * stride + segment;
                let b = a + stride;
                self.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                self.triangle_materials.extend([material, material]);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_ellipsoid(
        &mut self,
        center: Vec3,
        radii: Vec3,
        material: u32,
        slices: u32,
        stacks: u32,
        rotation: Mat3,
    ) {
        let start = self.vertices.len() as u32;
        for stack in 0..=stacks {
            let v = stack as f32 / stacks as f32;
            let phi = v * std::f32::consts::PI;
            for slice in 0..=slices {
                let theta = slice as f32 / slices as f32 * std::f32::consts::TAU;
                let unit = Vec3::new(theta.cos() * phi.sin(), phi.cos(), theta.sin() * phi.sin());
                let local = unit * radii;
                let normal = rotation * (unit / radii).normalize();
                self.vertices
                    .push(Self::vertex(center + rotation * local, normal));
            }
        }
        let stride = slices + 1;
        for stack in 0..stacks {
            for slice in 0..slices {
                let a = start + stack * stride + slice;
                let b = a + stride;
                self.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                self.triangle_materials.extend([material, material]);
            }
        }
    }

    fn add_knight(&mut self, origin: Vec3, color: Color, material: u32) {
        let direction = if color == Color::White { 1.0 } else { -1.0 };
        let profile = [
            Vec2::new(-0.31, 0.42),
            Vec2::new(0.28, 0.42),
            Vec2::new(0.30, 0.70),
            Vec2::new(0.18, 1.02),
            Vec2::new(0.32, 1.30),
            Vec2::new(0.18, 1.50),
            Vec2::new(-0.05, 1.57),
            Vec2::new(-0.23, 1.43),
            Vec2::new(-0.18, 1.19),
            Vec2::new(-0.37, 0.93),
            Vec2::new(-0.34, 0.62),
        ];
        let transform = |point: Vec3| origin + Vec3::new(point.x, point.y, point.z * direction);
        let half_width = 0.16;
        let front_start = self.vertices.len() as u32;
        for point in profile {
            self.vertices.push(Self::vertex(
                transform(Vec3::new(-half_width, point.y, point.x)),
                Vec3::new(-1.0, 0.0, 0.0),
            ));
        }
        for index in 1..profile.len() as u32 - 1 {
            self.indices
                .extend([front_start, front_start + index + 1, front_start + index]);
            self.triangle_materials.push(material);
        }
        let back_start = self.vertices.len() as u32;
        for point in profile {
            self.vertices.push(Self::vertex(
                transform(Vec3::new(half_width, point.y, point.x)),
                Vec3::X,
            ));
        }
        for index in 1..profile.len() as u32 - 1 {
            self.indices
                .extend([back_start, back_start + index, back_start + index + 1]);
            self.triangle_materials.push(material);
        }
        for index in 0..profile.len() {
            let next = (index + 1) % profile.len();
            let p0 = transform(Vec3::new(-half_width, profile[index].y, profile[index].x));
            let p1 = transform(Vec3::new(half_width, profile[index].y, profile[index].x));
            let p2 = transform(Vec3::new(half_width, profile[next].y, profile[next].x));
            let p3 = transform(Vec3::new(-half_width, profile[next].y, profile[next].x));
            let normal = (p1 - p0).cross(p3 - p0).normalize_or_zero();
            self.add_quad([p0, p1, p2, p3], normal, material);
        }
        self.add_ellipsoid(
            origin + Vec3::new(0.0, 1.35, 0.18 * direction),
            Vec3::new(0.22, 0.28, 0.32),
            material,
            18,
            10,
            Mat3::IDENTITY,
        );
    }
}

fn imported_piece_yaw(piece: Piece, color: Color) -> f32 {
    // Jeyhun1985's knight faces +X in the downloaded glTF. Rotate it
    // onto the board's rank axis; White faces +Z and Black faces -Z.
    let source_correction = if piece == Piece::Knight {
        -std::f32::consts::FRAC_PI_2
    } else {
        0.0
    };
    source_correction
        + if color == Color::White {
            0.0
        } else {
            std::f32::consts::PI
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::RenderSettings;

    #[test]
    fn starting_scene_is_well_formed() {
        let scene = ChessScene::from_board(&Board::default());
        assert!(scene.validate());
        assert!(scene.vertices.len() > 10_000);
        assert_eq!(scene.triangle_materials.len(), scene.indices.len() / 3);
    }

    #[test]
    fn scene_tracks_board_piece_count() {
        let full = ChessScene::from_board(&Board::default());
        let sparse: Board = "8/8/8/8/8/8/4K3/7k w - - 0 1".parse().unwrap();
        let sparse = ChessScene::from_board(&sparse);
        assert!(sparse.vertices.len() < full.vertices.len());
    }

    #[test]
    fn files_project_left_to_right_from_whites_side() {
        let mut settings = RenderSettings {
            camera_yaw: std::f32::consts::PI,
            ..RenderSettings::default()
        };
        settings.camera_pitch = 0.35;
        let (camera, _, right, _) = settings.camera_basis();
        let a1_screen_x = (board_position(0, 0, 0.0) - camera).dot(right);
        let h1_screen_x = (board_position(7, 0, 0.0) - camera).dot(right);

        assert!(a1_screen_x < h1_screen_x);
    }

    #[test]
    fn imported_knights_face_the_opposing_side() {
        let source_forward = Vec3::X;
        let white_forward =
            Mat3::from_rotation_y(imported_piece_yaw(Piece::Knight, Color::White)) * source_forward;
        let black_forward =
            Mat3::from_rotation_y(imported_piece_yaw(Piece::Knight, Color::Black)) * source_forward;

        assert!(white_forward.dot(Vec3::Z) > 0.999);
        assert!(black_forward.dot(Vec3::NEG_Z) > 0.999);
    }
}
