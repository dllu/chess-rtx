use std::path::PathBuf;

use anyhow::{Context, Result};
use chess::Board;
use chess_rtx::{
    app::ChessRtxApp,
    material::{BoardStyle, PieceStyle, RenderSettings},
    rt::RayTracer,
    scene::ChessScene,
};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PieceStyleArg {
    Ceramic,
    Metal,
    Glass,
}

impl From<PieceStyleArg> for PieceStyle {
    fn from(value: PieceStyleArg) -> Self {
        match value {
            PieceStyleArg::Ceramic => Self::Ceramic,
            PieceStyleArg::Metal => Self::Metal,
            PieceStyleArg::Glass => Self::FrostedGlass,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BoardStyleArg {
    Marble,
    Metal,
}

impl From<BoardStyleArg> for BoardStyle {
    fn from(value: BoardStyleArg) -> Self {
        match value {
            BoardStyleArg::Marble => Self::Marble,
            BoardStyleArg::Metal => Self::BrushedMetal,
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Render one frame without opening a window.
    #[arg(long, value_name = "PNG")]
    headless: Option<PathBuf>,

    /// Output width for headless rendering.
    #[arg(long, default_value_t = 960)]
    width: u32,

    /// Output height for headless rendering.
    #[arg(long, default_value_t = 640)]
    height: u32,

    /// Rays per pixel (1-16).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=16))]
    samples: u32,

    /// Progressive passes to average in a headless render (1-65536).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=65_536))]
    passes: u32,

    /// Optional Forsyth-Edwards Notation position.
    #[arg(long)]
    fen: Option<String>,

    /// Initial material for White's pieces.
    #[arg(long, value_enum, default_value = "ceramic")]
    light_style: PieceStyleArg,

    /// Initial material for Black's pieces.
    #[arg(long, value_enum, default_value = "metal")]
    dark_style: PieceStyleArg,

    /// Initial board material.
    #[arg(long, value_enum, default_value = "marble")]
    board_style: BoardStyleArg,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("chess_rtx=info,wgpu_core=warn"),
    )
    .format_timestamp_millis()
    .init();

    let args = Args::parse();
    let board = args
        .fen
        .as_deref()
        .map(str::parse::<Board>)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid FEN position: {error:?}"))?
        .unwrap_or_default();

    let mut settings = RenderSettings {
        samples: args.samples,
        ..Default::default()
    };
    settings.set_light_style(args.light_style.into());
    settings.set_dark_style(args.dark_style.into());
    settings.set_board_style(args.board_style.into());

    if let Some(path) = args.headless {
        let settings = RenderSettings {
            samples: args.samples,
            ..settings
        };
        let scene = ChessScene::from_board(&board);
        let geometry = scene.geometry_label();
        let mut renderer = RayTracer::new(args.width, args.height, &scene)
            .context("failed to initialize the Vulkan ray-tracing renderer")?;
        let first_pass = renderer
            .render(&settings)
            .context("hardware ray-traced frame failed")?;
        let mut accumulated = first_pass
            .iter()
            .map(|sample| u64::from(*sample))
            .collect::<Vec<_>>();
        for pass in 1..args.passes {
            let pass_pixels = renderer
                .render(&settings)
                .with_context(|| format!("hardware ray-traced pass {} failed", pass + 1))?;
            for (sum, sample) in accumulated.iter_mut().zip(pass_pixels) {
                *sum = sum.saturating_add(u64::from(sample));
            }
        }
        let pixels = accumulated
            .into_iter()
            .map(|sum| (sum / u64::from(args.passes)) as u8)
            .collect::<Vec<_>>();
        image::save_buffer(
            &path,
            &pixels,
            args.width,
            args.height,
            image::ColorType::Rgba8,
        )
        .with_context(|| format!("failed to save {}", path.display()))?;
        println!(
            "Rendered {}x{} with VK_KHR_ray_tracing_pipeline ({geometry}, {} rays/pixel) on {} -> {}",
            args.width,
            args.height,
            u64::from(args.samples) * u64::from(args.passes),
            renderer.device_name(),
            path.display()
        );
        return Ok(());
    }

    let mut wgpu_setup = egui_wgpu::WgpuSetupCreateNew::default();
    wgpu_setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    wgpu_setup.power_preference = wgpu::PowerPreference::HighPerformance;

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        hardware_acceleration: eframe::HardwareAcceleration::Required,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Chess RTX")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1050.0, 680.0]),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: wgpu_setup.into(),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Chess RTX",
        native_options,
        Box::new(move |creation_context| {
            Ok(Box::new(ChessRtxApp::new(
                creation_context,
                board,
                args.width,
                args.height,
                settings,
            )))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}
