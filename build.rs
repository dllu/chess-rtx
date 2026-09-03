use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result};

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=shaders/raygen.rgen");
    println!("cargo:rerun-if-changed=shaders/closesthit.rchit");
    println!("cargo:rerun-if-changed=shaders/miss.rmiss");
    println!("cargo:rerun-if-changed=shaders/shadow.rmiss");

    let output = PathBuf::from(env::var_os("OUT_DIR").context("OUT_DIR is not set")?);
    let shaders = [
        ("raygen.rgen", shaderc::ShaderKind::RayGeneration),
        ("closesthit.rchit", shaderc::ShaderKind::ClosestHit),
        ("miss.rmiss", shaderc::ShaderKind::Miss),
        ("shadow.rmiss", shaderc::ShaderKind::Miss),
    ];

    let compiler = shaderc::Compiler::new().context("could not initialize shaderc")?;
    let mut options =
        shaderc::CompileOptions::new().context("could not initialize shaderc options")?;
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_4);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_generate_debug_info();

    for (name, kind) in shaders {
        let path = PathBuf::from("shaders").join(name);
        let source = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let artifact = compiler
            .compile_into_spirv(&source, kind, name, "main", Some(&options))
            .with_context(|| format!("failed to compile {name}"))?;
        fs::write(output.join(format!("{name}.spv")), artifact.as_binary_u8())
            .with_context(|| format!("failed to write compiled {name}"))?;
    }
    Ok(())
}
