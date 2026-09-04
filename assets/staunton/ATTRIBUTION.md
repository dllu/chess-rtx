# Staunton chess-piece assets

The bundled high-detail piece geometry is by
[Jeyhun1985](https://sketchfab.com/Jeyhun1985) and is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The `.mesh` adaptations in this directory remain under that license and are not covered by the
project's software license.

| Piece | Source |
| --- | --- |
| Pawn | [Pawn - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/pawn-staunton-full-size-chess-set-ba992ea7fa094032a521c474e9aea09f) |
| Rook | [Rook - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/rook-staunton-full-size-chess-set-39a02aa72ed64856a68f039db0626538) |
| Knight | [Knight - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/knight-staunton-full-size-chess-set-dffe461e0def4f0a84bfe8551e58105f) |
| Bishop | [Bishop - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/bishop-staunton-full-size-chess-set-945dd86e3d6b4db2ad66a29a87ee9ef2) |
| Queen | [Queen - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/queen-staunton-full-size-chess-set-2f47e64f6f994c41bff0d58d42bd4475) |
| King | [King - Staunton Full Size Chess Set](https://sketchfab.com/3d-models/king-staunton-full-size-chess-set-cc297467241f42e1a384c7e5d0369143) |

## Changes made for Chess RTX

The Sketchfab glTF node transforms are baked into the geometry. Each piece is centered, grounded,
uniformly scaled for one board square, and simplified with meshoptimizer to a ray-tracing-friendly
triangle budget. Original textures and materials are not used; the user-selectable Chess RTX
materials are applied instead. These changes make the generated `.mesh` files adaptations of the
original models.
