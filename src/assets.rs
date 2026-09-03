use std::{
    fs,
    io::{Cursor, Read, Write},
    mem::size_of,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{bail, ensure, Context, Result};
use chess::Piece;
use glam::{Mat3, Mat4, Vec3};
use meshopt::{simplify::SimplifyOptions, VertexDataAdapter};

use crate::scene::Vertex;

const MESH_MAGIC: &[u8; 8] = b"CHRTXM01";
const MAX_VERTICES: usize = 2_000_000;
const MAX_INDICES: usize = 6_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceAssetKind {
    Pawn,
    Rook,
    Knight,
    Bishop,
    Queen,
    King,
}

impl PieceAssetKind {
    pub const ALL: [Self; 6] = [
        Self::Pawn,
        Self::Rook,
        Self::Knight,
        Self::Bishop,
        Self::Queen,
        Self::King,
    ];

    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Pawn => "pawn",
            Self::Rook => "rook",
            Self::Knight => "knight",
            Self::Bishop => "bishop",
            Self::Queen => "queen",
            Self::King => "king",
        }
    }

    #[must_use]
    pub const fn piece(self) -> Piece {
        match self {
            Self::Pawn => Piece::Pawn,
            Self::Rook => Piece::Rook,
            Self::Knight => Piece::Knight,
            Self::Bishop => Piece::Bishop,
            Self::Queen => Piece::Queen,
            Self::King => Piece::King,
        }
    }

    #[must_use]
    const fn target_height(self) -> f32 {
        match self {
            Self::Pawn => 1.42,
            Self::Rook => 1.58,
            Self::Knight => 1.72,
            Self::Bishop => 1.76,
            Self::Queen => 1.94,
            Self::King => 2.08,
        }
    }

    #[must_use]
    const fn target_triangles(self) -> usize {
        match self {
            Self::Pawn => 22_000,
            Self::Rook | Self::Bishop => 32_000,
            Self::Knight => 56_000,
            Self::Queen => 40_000,
            Self::King => 44_000,
        }
    }
}

impl TryFrom<&str> for PieceAssetKind {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "pawn" => Ok(Self::Pawn),
            "rook" => Ok(Self::Rook),
            "knight" => Ok(Self::Knight),
            "bishop" => Ok(Self::Bishop),
            "queen" => Ok(Self::Queen),
            "king" => Ok(Self::King),
            _ => bail!("unknown chess piece {value:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PieceMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl PieceMesh {
    fn validate(&self) -> bool {
        !self.vertices.is_empty()
            && !self.indices.is_empty()
            && self.indices.len().is_multiple_of(3)
            && self
                .indices
                .iter()
                .all(|&index| index < self.vertices.len() as u32)
            && self.vertices.iter().all(|vertex| {
                vertex
                    .position
                    .iter()
                    .all(|component| component.is_finite())
                    && vertex.normal.iter().all(|component| component.is_finite())
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PreparationStats {
    pub source_triangles: usize,
    pub output_triangles: usize,
    pub output_vertices: usize,
    pub geometric_error: f32,
}

#[derive(Default)]
struct PieceMeshLibrary {
    pawn: Option<PieceMesh>,
    rook: Option<PieceMesh>,
    knight: Option<PieceMesh>,
    bishop: Option<PieceMesh>,
    queen: Option<PieceMesh>,
    king: Option<PieceMesh>,
}

impl PieceMeshLibrary {
    fn load() -> Self {
        let mut library = Self::default();
        for kind in PieceAssetKind::ALL {
            let path = asset_root().join(format!("{}.mesh", kind.slug()));
            if !path.exists() {
                log::debug!("{} is absent; using procedural fallback", path.display());
                continue;
            }
            match read_mesh(&path) {
                Ok(mesh) => {
                    log::info!(
                        "loaded {}: {} vertices, {} triangles",
                        path.display(),
                        mesh.vertices.len(),
                        mesh.indices.len() / 3
                    );
                    *library.slot_mut(kind.piece()) = Some(mesh);
                }
                Err(error) => {
                    log::warn!("could not load {}: {error:#}", path.display());
                }
            }
        }
        library
    }

    fn get(&self, piece: Piece) -> Option<&PieceMesh> {
        match piece {
            Piece::Pawn => self.pawn.as_ref(),
            Piece::Rook => self.rook.as_ref(),
            Piece::Knight => self.knight.as_ref(),
            Piece::Bishop => self.bishop.as_ref(),
            Piece::Queen => self.queen.as_ref(),
            Piece::King => self.king.as_ref(),
        }
    }

    fn slot_mut(&mut self, piece: Piece) -> &mut Option<PieceMesh> {
        match piece {
            Piece::Pawn => &mut self.pawn,
            Piece::Rook => &mut self.rook,
            Piece::Knight => &mut self.knight,
            Piece::Bishop => &mut self.bishop,
            Piece::Queen => &mut self.queen,
            Piece::King => &mut self.king,
        }
    }

    fn loaded_count(&self) -> usize {
        PieceAssetKind::ALL
            .iter()
            .filter(|kind| self.get(kind.piece()).is_some())
            .count()
    }
}

static PIECE_MESHES: OnceLock<PieceMeshLibrary> = OnceLock::new();

#[must_use]
pub fn piece_mesh(piece: Piece) -> Option<&'static PieceMesh> {
    PIECE_MESHES.get_or_init(PieceMeshLibrary::load).get(piece)
}

#[must_use]
pub fn loaded_piece_mesh_count() -> usize {
    PIECE_MESHES
        .get_or_init(PieceMeshLibrary::load)
        .loaded_count()
}

fn asset_root() -> PathBuf {
    std::env::var_os("CHESS_RTX_ASSET_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/staunton"),
        PathBuf::from,
    )
}

/// Imports a glTF model, normalizes it for one board square, simplifies it, and
/// writes the renderer-native piece format.
///
/// # Errors
///
/// Returns an error for an unreadable or unsupported glTF, invalid geometry, or
/// a failure to write the output mesh.
pub fn prepare_gltf(input: &Path, output: &Path, kind: PieceAssetKind) -> Result<PreparationStats> {
    let mut mesh = import_gltf(input)?;
    normalize(&mut mesh, kind)?;
    let source_triangles = mesh.indices.len() / 3;
    let mut geometric_error = 0.0;
    if source_triangles > kind.target_triangles() {
        mesh = simplify_mesh(&mesh, kind.target_triangles(), &mut geometric_error)?;
    } else {
        mesh = compact_mesh(&mesh, &mesh.indices)?;
    }
    fill_missing_normals(&mut mesh);
    ensure!(mesh.validate(), "prepared mesh is invalid");
    write_mesh(output, &mesh)?;
    Ok(PreparationStats {
        source_triangles,
        output_triangles: mesh.indices.len() / 3,
        output_vertices: mesh.vertices.len(),
        geometric_error,
    })
}

fn import_gltf(path: &Path) -> Result<PieceMesh> {
    let (document, buffers, _) =
        gltf::import(path).with_context(|| format!("could not import {}", path.display()))?;
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF contains no scene")?;
    let mut mesh = PieceMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for node in scene.nodes() {
        import_node(&node, Mat4::IDENTITY, &buffers, &mut mesh)?;
    }
    ensure!(mesh.validate(), "glTF contains no valid triangle geometry");
    Ok(mesh)
}

fn import_node(
    node: &gltf::Node<'_>,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    destination: &mut PieceMesh,
) -> Result<()> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let transform = parent_transform * local;
    let linear = Mat3::from_mat4(transform);
    let normal_transform = if linear.determinant().abs() > f32::EPSILON {
        linear.inverse().transpose()
    } else {
        Mat3::IDENTITY
    };

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            ensure!(
                primitive.mode() == gltf::mesh::Mode::Triangles,
                "only triangle-list glTF primitives are supported"
            );
            let reader = primitive.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
            let positions = reader
                .read_positions()
                .context("glTF primitive has no positions")?
                .collect::<Vec<_>>();
            let normals = reader
                .read_normals()
                .map(std::iter::Iterator::collect::<Vec<_>>);
            if let Some(normals) = &normals {
                ensure!(
                    normals.len() == positions.len(),
                    "glTF normal and position counts differ"
                );
            }
            let base = destination.vertices.len() as u32;
            for (index, position) in positions.iter().enumerate() {
                let position = transform.transform_point3(Vec3::from_array(*position));
                let normal = normals.as_ref().map_or(Vec3::ZERO, |values| {
                    normal_transform
                        .mul_vec3(Vec3::from_array(values[index]))
                        .normalize_or_zero()
                });
                destination.vertices.push(Vertex {
                    position: position.extend(1.0).to_array(),
                    normal: normal.extend(0.0).to_array(),
                });
            }
            if let Some(indices) = reader.read_indices() {
                destination
                    .indices
                    .extend(indices.into_u32().map(|index| base + index));
            } else {
                destination
                    .indices
                    .extend((0..positions.len() as u32).map(|index| base + index));
            }
        }
    }
    for child in node.children() {
        import_node(&child, transform, buffers, destination)?;
    }
    Ok(())
}

fn normalize(mesh: &mut PieceMesh, kind: PieceAssetKind) -> Result<()> {
    let (mut minimum, mut maximum) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for vertex in &mesh.vertices {
        let position = Vec3::from_array(vertex.position[..3].try_into().unwrap());
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    let size = maximum - minimum;
    ensure!(size.y > f32::EPSILON, "piece mesh has no vertical extent");
    let horizontal_extent = size.x.max(size.z);
    let height_scale = kind.target_height() / size.y;
    let width_scale = if horizontal_extent > f32::EPSILON {
        0.88 / horizontal_extent
    } else {
        height_scale
    };
    let scale = height_scale.min(width_scale);
    let center = Vec3::new(
        (minimum.x + maximum.x) * 0.5,
        minimum.y,
        (minimum.z + maximum.z) * 0.5,
    );
    for vertex in &mut mesh.vertices {
        let position = Vec3::from_array(vertex.position[..3].try_into().unwrap());
        vertex.position = ((position - center) * scale).extend(1.0).to_array();
    }
    Ok(())
}

fn simplify_mesh(
    mesh: &PieceMesh,
    target_triangles: usize,
    geometric_error: &mut f32,
) -> Result<PieceMesh> {
    let bytes = bytemuck::cast_slice(&mesh.vertices);
    let adapter = VertexDataAdapter::new(bytes, size_of::<Vertex>(), 0)
        .context("could not describe piece vertices to meshoptimizer")?;
    let normals = mesh
        .vertices
        .iter()
        .flat_map(|vertex| vertex.normal[..3].iter().copied())
        .collect::<Vec<_>>();
    let locked = vec![false; mesh.vertices.len()];
    let simplified = meshopt::simplify::simplify_with_attributes_and_locks(
        &mesh.indices,
        &adapter,
        &normals,
        &[0.15, 0.15, 0.15],
        size_of::<[f32; 3]>(),
        &locked,
        target_triangles * 3,
        0.003,
        // Preserve disconnected ornamental components such as the king's cross
        // and queen's crown; permissive collapses are sufficient for these
        // dense printable meshes.
        SimplifyOptions::Permissive,
        Some(geometric_error),
    );
    ensure!(
        !simplified.is_empty(),
        "mesh simplification removed all triangles"
    );
    compact_mesh(mesh, &simplified)
}

fn compact_mesh(source: &PieceMesh, indices: &[u32]) -> Result<PieceMesh> {
    let mut remap = vec![u32::MAX; source.vertices.len()];
    let mut vertices = Vec::new();
    let mut compact_indices = Vec::with_capacity(indices.len());
    for &source_index in indices {
        let slot = remap
            .get_mut(source_index as usize)
            .context("piece index lies outside its vertex buffer")?;
        if *slot == u32::MAX {
            *slot = vertices.len() as u32;
            vertices.push(source.vertices[source_index as usize]);
        }
        compact_indices.push(*slot);
    }
    Ok(PieceMesh {
        vertices,
        indices: compact_indices,
    })
}

fn fill_missing_normals(mesh: &mut PieceMesh) {
    let mut accumulated = vec![Vec3::ZERO; mesh.vertices.len()];
    for triangle in mesh.indices.chunks_exact(3) {
        let a = Vec3::from_array(
            mesh.vertices[triangle[0] as usize].position[..3]
                .try_into()
                .unwrap(),
        );
        let b = Vec3::from_array(
            mesh.vertices[triangle[1] as usize].position[..3]
                .try_into()
                .unwrap(),
        );
        let c = Vec3::from_array(
            mesh.vertices[triangle[2] as usize].position[..3]
                .try_into()
                .unwrap(),
        );
        let face = (b - a).cross(c - a);
        for &index in triangle {
            accumulated[index as usize] += face;
        }
    }
    for (vertex, generated) in mesh.vertices.iter_mut().zip(accumulated) {
        let normal = Vec3::from_array(vertex.normal[..3].try_into().unwrap());
        if normal.length_squared() < 0.25 {
            vertex.normal = generated.normalize_or_zero().extend(0.0).to_array();
        }
    }
}

fn write_mesh(path: &Path, mesh: &PieceMesh) -> Result<()> {
    ensure!(mesh.validate(), "refusing to write an invalid piece mesh");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let mut file =
        fs::File::create(path).with_context(|| format!("could not create {}", path.display()))?;
    file.write_all(MESH_MAGIC)?;
    file.write_all(&(mesh.vertices.len() as u32).to_le_bytes())?;
    file.write_all(&(mesh.indices.len() as u32).to_le_bytes())?;
    for vertex in &mesh.vertices {
        for component in vertex.position.into_iter().chain(vertex.normal) {
            file.write_all(&component.to_le_bytes())?;
        }
    }
    for index in &mesh.indices {
        file.write_all(&index.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn read_mesh(path: &Path) -> Result<PieceMesh> {
    let bytes = fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
    decode_mesh(&bytes)
}

fn decode_mesh(bytes: &[u8]) -> Result<PieceMesh> {
    let mut reader = Cursor::new(bytes);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == MESH_MAGIC, "invalid Chess RTX mesh signature");
    let vertex_count = read_u32(&mut reader)? as usize;
    let index_count = read_u32(&mut reader)? as usize;
    ensure!(
        vertex_count <= MAX_VERTICES,
        "piece mesh has too many vertices"
    );
    ensure!(
        index_count <= MAX_INDICES,
        "piece mesh has too many indices"
    );
    ensure!(
        index_count.is_multiple_of(3),
        "piece indices are not triangles"
    );

    let expected = 16_usize
        .checked_add(
            vertex_count
                .checked_mul(size_of::<Vertex>())
                .context("mesh size overflow")?,
        )
        .and_then(|size| size.checked_add(index_count.checked_mul(size_of::<u32>())?))
        .context("mesh size overflow")?;
    ensure!(bytes.len() == expected, "piece mesh byte length is invalid");

    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut values = [0.0_f32; 8];
        for value in &mut values {
            *value = read_f32(&mut reader)?;
        }
        vertices.push(Vertex {
            position: values[..4].try_into().unwrap(),
            normal: values[4..].try_into().unwrap(),
        });
    }
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(read_u32(&mut reader)?);
    }
    let mesh = PieceMesh { vertices, indices };
    ensure!(mesh.validate(), "piece mesh geometry is invalid");
    Ok(mesh)
}

fn read_u32(reader: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut Cursor<&[u8]>) -> Result<f32> {
    Ok(f32::from_bits(read_u32(reader)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_mesh_codec_round_trips() {
        let mesh = PieceMesh {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.0, 0.0, 1.0],
                    normal: [0.0, 1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 0.0, 0.0, 1.0],
                    normal: [0.0, 1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [0.0, 0.0, 1.0, 1.0],
                    normal: [0.0, 1.0, 0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 2],
        };
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pawn.mesh");
        write_mesh(&path, &mesh).unwrap();
        let decoded = read_mesh(&path).unwrap();

        assert_eq!(decoded.indices, mesh.indices);
        assert_eq!(decoded.vertices.len(), mesh.vertices.len());
        assert_eq!(
            decoded.vertices[1].position.map(f32::to_bits),
            mesh.vertices[1].position.map(f32::to_bits)
        );
    }
}
