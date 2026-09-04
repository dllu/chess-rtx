// The SVG artwork embedded by this module is licensed separately under the
// BSD 3-Clause license. See assets/chess-pieces-2d/ATTRIBUTION.md and LICENSE.

use chess::{Color, Piece};
use eframe::egui::{self, ImageSource};

pub(crate) fn image_source(piece: Piece, color: Color) -> ImageSource<'static> {
    match (color, piece) {
        (Color::White, Piece::Pawn) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_plt45.svg")
        }
        (Color::White, Piece::Knight) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_nlt45.svg")
        }
        (Color::White, Piece::Bishop) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_blt45.svg")
        }
        (Color::White, Piece::Rook) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_rlt45.svg")
        }
        (Color::White, Piece::Queen) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_qlt45.svg")
        }
        (Color::White, Piece::King) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_klt45.svg")
        }
        (Color::Black, Piece::Pawn) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_pdt45.svg")
        }
        (Color::Black, Piece::Knight) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_ndt45.svg")
        }
        (Color::Black, Piece::Bishop) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_bdt45.svg")
        }
        (Color::Black, Piece::Rook) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_rdt45.svg")
        }
        (Color::Black, Piece::Queen) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_qdt45.svg")
        }
        (Color::Black, Piece::King) => {
            egui::include_image!("../assets/chess-pieces-2d/Chess_kdt45.svg")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{load::SizeHint, Context, TextureOptions};

    #[test]
    fn every_board_piece_svg_loads_at_hidpi_resolution() {
        let context = Context::default();
        egui_extras::install_image_loaders(&context);

        for color in [Color::White, Color::Black] {
            for piece in [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                let loaded = image_source(piece, color)
                    .load(
                        &context,
                        TextureOptions::LINEAR,
                        SizeHint::Size {
                            width: 90,
                            height: 90,
                            maintain_aspect_ratio: false,
                        },
                    )
                    .expect("embedded board-piece SVG should load");
                assert!(loaded.is_ready());
            }
        }
    }
}
