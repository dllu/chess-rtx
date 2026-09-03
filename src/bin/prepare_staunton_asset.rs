use std::path::PathBuf;

use anyhow::Result;
use chess_rtx::assets::{prepare_gltf, PieceAssetKind};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Convert a downloaded Staunton glTF into a compact Chess RTX mesh")]
struct Args {
    #[arg(long)]
    piece: String,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let piece = PieceAssetKind::try_from(args.piece.as_str())?;
    let stats = prepare_gltf(&args.input, &args.output, piece)?;
    println!(
        "Prepared {}: {} -> {} triangles, {} vertices, relative error {:.6}",
        piece.slug(),
        stats.source_triangles,
        stats.output_triangles,
        stats.output_vertices,
        stats.geometric_error
    );
    Ok(())
}
