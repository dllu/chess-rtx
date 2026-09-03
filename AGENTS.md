# Chess RTX

This project implements a real time ray traced photorealistic chess game.

- Use Rust with wgpu and egui
- Use Vulkan Ray Tracing via VK_KHR_ray_tracing_pipeline
- This machine is equipped with an NVIDIA GPU
- We can use high quality 3D assets for official Staunton pieces with a free license, e.g.:
  - Rook: https://sketchfab.com/3d-models/rook-staunton-full-size-chess-set-39a02aa72ed64856a68f039db0626538
  - Knight: https://sketchfab.com/3d-models/knight-staunton-full-size-chess-set-dffe461e0def4f0a84bfe8551e58105f
  - King: https://sketchfab.com/3d-models/king-staunton-full-size-chess-set-cc297467241f42e1a384c7e5d0369143
  - Pawn: https://sketchfab.com/3d-models/pawn-staunton-full-size-chess-set-ba992ea7fa094032a521c474e9aea09f
  - Bishop: https://sketchfab.com/3d-models/bishop-staunton-full-size-chess-set-945dd86e3d6b4db2ad66a29a87ee9ef2
- Style and material guidance:
  - Configurable styles for chess pieces including:
    - Frosted glass
    - Metal
    - Ceramic
    - Maybe just have a UI letting the user come up with new materials and tweak various settings in addition to the presets for the above
  - Board should have independently configurable appearance, including:
    - Marble appearance with strong sharp reflections, though the board colors should still be clearly visible
    - Metallic
- The ray tracing demo should include:
  - Ray traced reflections
  - Anisotropic reflections
  - Refraction
  - Caustics
- Optional: UCI compatible interface to use any chess engine
  - You can download Stockfish to test it out

In addition to the interactive demo, you can implement an offscreen renderer to test it out rapidly without spawning windows (as this workstation is being used for other work simultaneously).
