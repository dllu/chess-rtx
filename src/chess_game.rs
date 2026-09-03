use chess::{Board, BoardStatus, ChessMove, Color, MoveGen, Piece, Square};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClickResult {
    Unchanged,
    SelectionChanged,
    Moved,
}

pub struct ChessGame {
    board: Board,
    initial_board: Board,
    selected: Option<Square>,
    history: Vec<Board>,
    move_labels: Vec<String>,
}

impl ChessGame {
    #[must_use]
    pub fn new(board: Board) -> Self {
        Self {
            board,
            initial_board: board,
            selected: None,
            history: Vec::new(),
            move_labels: Vec::new(),
        }
    }

    #[must_use]
    pub const fn board(&self) -> &Board {
        &self.board
    }

    #[must_use]
    pub const fn selected(&self) -> Option<Square> {
        self.selected
    }

    #[must_use]
    pub fn move_labels(&self) -> &[String] {
        &self.move_labels
    }

    #[must_use]
    pub fn legal_destinations(&self) -> Vec<Square> {
        let Some(source) = self.selected else {
            return Vec::new();
        };
        MoveGen::new_legal(&self.board)
            .filter(|chess_move| chess_move.get_source() == source)
            .map(|chess_move| chess_move.get_dest())
            .collect()
    }

    pub fn click(&mut self, square: Square) -> ClickResult {
        if self.board.status() != BoardStatus::Ongoing {
            return ClickResult::Unchanged;
        }

        if let Some(source) = self.selected {
            let legal_move = MoveGen::new_legal(&self.board)
                .filter(|chess_move| {
                    chess_move.get_source() == source && chess_move.get_dest() == square
                })
                .max_by_key(|chess_move| {
                    usize::from(chess_move.get_promotion() == Some(Piece::Queen))
                });

            if let Some(chess_move) = legal_move {
                let label = move_label(&self.board, chess_move);
                self.history.push(self.board);
                self.board = self.board.make_move_new(chess_move);
                self.move_labels.push(label);
                self.selected = None;
                return ClickResult::Moved;
            }
        }

        if self.board.color_on(square) == Some(self.board.side_to_move()) {
            self.selected = Some(square);
            ClickResult::SelectionChanged
        } else if self.selected.take().is_some() {
            ClickResult::SelectionChanged
        } else {
            ClickResult::Unchanged
        }
    }

    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.history.pop() else {
            return false;
        };
        self.board = previous;
        self.move_labels.pop();
        self.selected = None;
        true
    }

    pub fn reset(&mut self) {
        self.board = self.initial_board;
        self.selected = None;
        self.history.clear();
        self.move_labels.clear();
    }

    pub fn load_position(&mut self, board: Board) {
        self.board = board;
        self.selected = None;
        self.history.clear();
        self.move_labels.clear();
    }

    #[must_use]
    pub fn status_label(&self) -> String {
        match self.board.status() {
            BoardStatus::Ongoing if *self.board.checkers() != chess::EMPTY => {
                format!("{} to move — check", color_name(self.board.side_to_move()))
            }
            BoardStatus::Ongoing => format!("{} to move", color_name(self.board.side_to_move())),
            BoardStatus::Stalemate => "Draw — stalemate".to_owned(),
            BoardStatus::Checkmate => format!(
                "Checkmate — {} wins",
                color_name(!self.board.side_to_move())
            ),
        }
    }
}

#[must_use]
pub const fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "White",
        Color::Black => "Black",
    }
}

fn move_label(board: &Board, chess_move: ChessMove) -> String {
    let piece = board
        .piece_on(chess_move.get_source())
        .unwrap_or(Piece::Pawn);
    let capture = board.piece_on(chess_move.get_dest()).is_some();
    let promotion = chess_move
        .get_promotion()
        .map_or(String::new(), |promoted| {
            format!("={}", promoted.to_string(Color::White))
        });
    format!(
        "{}{}{}{}",
        piece.to_string(Color::White),
        chess_move.get_source(),
        if capture { "×" } else { "–" },
        format_args!("{}{promotion}", chess_move.get_dest())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_moves_only_along_legal_moves() {
        let mut game = ChessGame::new(Board::default());
        assert_eq!(game.click(Square::E2), ClickResult::SelectionChanged);
        assert!(game.legal_destinations().contains(&Square::E4));
        assert_eq!(game.click(Square::E5), ClickResult::SelectionChanged);
        assert_eq!(game.board().piece_on(Square::E2), Some(Piece::Pawn));
        assert_eq!(game.click(Square::E2), ClickResult::SelectionChanged);
        assert_eq!(game.click(Square::E4), ClickResult::Moved);
        assert_eq!(game.board().piece_on(Square::E4), Some(Piece::Pawn));
        assert_eq!(game.board().side_to_move(), Color::Black);
    }

    #[test]
    fn undo_restores_position() {
        let mut game = ChessGame::new(Board::default());
        game.click(Square::G1);
        game.click(Square::F3);
        assert!(game.undo());
        assert_eq!(game.board().piece_on(Square::G1), Some(Piece::Knight));
        assert_eq!(game.board().side_to_move(), Color::White);
    }

    #[test]
    fn loading_position_clears_interaction_history() {
        let mut game = ChessGame::new(Board::default());
        game.click(Square::E2);
        game.click(Square::E4);
        let position: Board = "8/8/8/8/8/8/4K3/7k w - - 0 1".parse().unwrap();

        game.load_position(position);

        assert_eq!(game.board(), &position);
        assert_eq!(game.selected(), None);
        assert!(game.move_labels().is_empty());
        assert!(!game.undo());
    }
}
