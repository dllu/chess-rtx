use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use chess::{Board, Color, File, Piece, Rank, Square};
use eframe::egui::{
    self, Align2, Color32, FontId, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions,
    Vec2,
};

use crate::{
    chess_game::{ChessGame, ClickResult},
    material::{BoardStyle, GpuMaterial, PieceStyle, RenderSettings},
    rt::RayTracer,
    scene::ChessScene,
};

const DEFAULT_ACCUMULATION_PASS_LIMIT: u32 = 4_096;
const MAX_ACCUMULATION_PASS_LIMIT: u32 = 65_536;
const VIEWPORT_RESIZE_SETTLE_TIME: Duration = Duration::from_millis(120);
const MIN_INTERACTIVE_RENDER_DIMENSION: u32 = 64;
const MAX_INTERACTIVE_RENDER_DIMENSION: u32 = 8_192;

#[derive(Clone, Copy)]
struct PendingViewportResize {
    dimensions: [u32; 2],
    stable_since: Instant,
}

pub struct ChessRtxApp {
    game: ChessGame,
    renderer: Option<RayTracer>,
    texture: Option<TextureHandle>,
    pixels: Vec<u8>,
    accumulation: Vec<u64>,
    accumulation_passes: u64,
    accumulation_pass_limit: u32,
    continuous_accumulation: bool,
    accumulation_paused: bool,
    settings: RenderSettings,
    width: u32,
    height: u32,
    pending_viewport_resize: Option<PendingViewportResize>,
    failed_viewport_resize: Option<[u32; 2]>,
    render_requested: bool,
    scene_rebuild_requested: bool,
    piece_geometry: &'static str,
    status: String,
    error: Option<String>,
    fen_input: String,
    fen_error: Option<String>,
    last_render_ms: f32,
    save_counter: u32,
}

impl ChessRtxApp {
    #[must_use]
    pub fn new(
        creation_context: &eframe::CreationContext<'_>,
        board: Board,
        width: u32,
        height: u32,
        settings: RenderSettings,
    ) -> Self {
        configure_style(&creation_context.egui_ctx);
        let mut app = Self {
            game: ChessGame::new(board),
            renderer: None,
            texture: None,
            pixels: Vec::new(),
            accumulation: Vec::new(),
            accumulation_passes: 0,
            accumulation_pass_limit: DEFAULT_ACCUMULATION_PASS_LIMIT,
            continuous_accumulation: false,
            accumulation_paused: false,
            settings,
            width,
            height,
            pending_viewport_resize: None,
            failed_viewport_resize: None,
            render_requested: true,
            scene_rebuild_requested: true,
            piece_geometry: "loading piece geometry",
            status: "Initializing RTX pipeline…".to_owned(),
            error: None,
            fen_input: board.to_string(),
            fen_error: None,
            last_render_ms: 0.0,
            save_counter: 0,
        };
        app.render_if_needed(&creation_context.egui_ctx);
        app
    }

    fn rebuild_renderer(&mut self) {
        self.renderer = None;
        self.accumulation_paused = false;
        self.reset_accumulation();
        let scene = ChessScene::from_board(self.game.board());
        let geometry = scene.geometry_label();
        self.piece_geometry = geometry;
        match RayTracer::new(self.width, self.height, &scene) {
            Ok(renderer) => {
                self.status = format!("RTX ready · {geometry} · {}", renderer.device_name());
                self.renderer = Some(renderer);
                self.error = None;
                self.render_requested = true;
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                "RTX initialization failed".clone_into(&mut self.status);
            }
        }
        self.scene_rebuild_requested = false;
    }

    fn render_if_needed(&mut self, context: &egui::Context) {
        if self.scene_rebuild_requested {
            self.rebuild_renderer();
        }
        if !self.render_requested || self.accumulation_paused {
            return;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let started = Instant::now();
        match renderer.render(&self.settings) {
            Ok(pass_pixels) => {
                self.last_render_ms = started.elapsed().as_secs_f32() * 1000.0;
                if self.accumulation.len() != pass_pixels.len() {
                    self.accumulation = vec![0; pass_pixels.len()];
                    self.accumulation_passes = 0;
                }
                for (sum, sample) in self.accumulation.iter_mut().zip(&pass_pixels) {
                    *sum = sum.saturating_add(u64::from(*sample));
                }
                self.accumulation_passes = self.accumulation_passes.saturating_add(1);
                self.pixels.resize(pass_pixels.len(), 0);
                for (pixel, sum) in self.pixels.iter_mut().zip(&self.accumulation) {
                    *pixel = (sum / self.accumulation_passes) as u8;
                }
                let image =
                    egui::ColorImage::from_rgba_unmultiplied(renderer.dimensions(), &self.pixels);
                if let Some(texture) = self.texture.as_mut() {
                    texture.set(image, TextureOptions::LINEAR);
                } else {
                    self.texture = Some(context.load_texture(
                        "chess-rtx-output",
                        image,
                        TextureOptions::LINEAR,
                    ));
                }
                self.error = None;
                let target = if self.continuous_accumulation {
                    "∞".to_owned()
                } else {
                    self.accumulation_pass_limit.to_string()
                };
                self.status = format!(
                    "{} rays/pixel · pass {}/{} · {:.1} ms/pass · {} · {}",
                    u64::from(self.settings.samples).saturating_mul(self.accumulation_passes),
                    self.accumulation_passes,
                    target,
                    self.last_render_ms,
                    self.piece_geometry,
                    renderer.device_name()
                );
                context.request_repaint();
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                "Ray dispatch failed".clone_into(&mut self.status);
            }
        }
        self.update_render_schedule(context);
    }

    fn mark_render_changed(&mut self, changed: bool) {
        if changed {
            self.reset_accumulation();
        }
    }

    fn reset_accumulation(&mut self) {
        self.accumulation.fill(0);
        self.accumulation_passes = 0;
        self.render_requested = true;
    }

    fn accumulation_has_budget(&self) -> bool {
        self.continuous_accumulation
            || self.accumulation_passes < u64::from(self.accumulation_pass_limit)
    }

    fn update_render_schedule(&mut self, context: &egui::Context) {
        self.render_requested = self.error.is_none()
            && self.renderer.is_some()
            && !self.accumulation_paused
            && self.accumulation_has_budget();
        if self.render_requested {
            context.request_repaint();
        }
    }

    fn track_viewport_size(&mut self, context: &egui::Context, logical_size: Vec2) {
        let dimensions = interactive_render_dimensions(logical_size, context.pixels_per_point());
        if dimensions == [self.width, self.height]
            || self.failed_viewport_resize == Some(dimensions)
        {
            self.pending_viewport_resize = None;
            return;
        }

        let now = Instant::now();
        let Some(pending) = self.pending_viewport_resize else {
            self.pending_viewport_resize = Some(PendingViewportResize {
                dimensions,
                stable_since: now,
            });
            context.request_repaint_after(VIEWPORT_RESIZE_SETTLE_TIME);
            return;
        };
        if pending.dimensions != dimensions {
            self.failed_viewport_resize = None;
            self.pending_viewport_resize = Some(PendingViewportResize {
                dimensions,
                stable_since: now,
            });
            context.request_repaint_after(VIEWPORT_RESIZE_SETTLE_TIME);
            return;
        }

        let elapsed = now.saturating_duration_since(pending.stable_since);
        if elapsed < VIEWPORT_RESIZE_SETTLE_TIME {
            context.request_repaint_after(VIEWPORT_RESIZE_SETTLE_TIME - elapsed);
            return;
        }
        self.pending_viewport_resize = None;
        self.resize_render_target(dimensions);
    }

    fn resize_render_target(&mut self, dimensions: [u32; 2]) {
        let [width, height] = dimensions;
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match renderer.resize(width, height) {
            Ok(()) => {
                self.width = width;
                self.height = height;
                self.failed_viewport_resize = None;
                self.accumulation.clear();
                self.pixels.clear();
                self.reset_accumulation();
                self.status = format!("Resized RTX viewport to {width} × {height}");
            }
            Err(error) => {
                self.failed_viewport_resize = Some(dimensions);
                self.status = format!(
                    "Keeping {} × {} viewport; {width} × {height} resize failed: {error:#}",
                    self.width, self.height
                );
            }
        }
    }

    fn sync_fen_from_board(&mut self) {
        self.fen_input = self.game.board().to_string();
        self.fen_error = None;
    }

    fn load_fen(&mut self) {
        match self.fen_input.trim().parse::<Board>() {
            Ok(board) => {
                self.game.load_position(board);
                self.sync_fen_from_board();
                self.accumulation_paused = false;
                self.scene_rebuild_requested = true;
                "Loaded FEN position".clone_into(&mut self.status);
            }
            Err(error) => {
                self.fen_error = Some(format!("Invalid FEN: {error:?}"));
            }
        }
    }

    fn top_bar(&mut self, context: &egui::Context) {
        egui::TopBottomPanel::top("top_bar")
            .exact_height(48.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(13, 16, 22))
                    .inner_margin(egui::Margin::symmetric(14, 8)),
            )
            .show(context, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(
                        RichText::new("CHESS RTX")
                            .size(21.0)
                            .strong()
                            .color(Color32::from_rgb(232, 195, 112)),
                    );
                    ui.add_space(12.0);
                    ui.label(RichText::new(&self.status).color(Color32::from_gray(170)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Save PNG").clicked() {
                            self.save_current_frame();
                        }
                        if ui.button("Render").clicked() {
                            self.accumulation_paused = false;
                            self.reset_accumulation();
                            self.update_render_schedule(context);
                        }
                        if self.accumulation_paused {
                            if ui.button("Resume").clicked() {
                                self.accumulation_paused = false;
                                self.update_render_schedule(context);
                            }
                        } else if self.render_requested && ui.button("Pause").clicked() {
                            self.accumulation_paused = true;
                            self.render_requested = false;
                        }
                    });
                });
            });
    }

    fn side_panel(&mut self, context: &egui::Context) {
        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(322.0)
            .frame(
                egui::Frame::side_top_panel(&context.style()).inner_margin(egui::Margin::same(14)),
            )
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Game");
                    ui.label(
                        RichText::new(self.game.status_label())
                            .color(Color32::from_rgb(232, 195, 112)),
                    );
                    self.chess_board(ui);
                    ui.horizontal(|ui| {
                        if ui.button("New game").clicked() {
                            self.game.reset();
                            self.sync_fen_from_board();
                            self.scene_rebuild_requested = true;
                        }
                        if ui
                            .add_enabled(
                                !self.game.move_labels().is_empty(),
                                egui::Button::new("Undo"),
                            )
                            .clicked()
                            && self.game.undo()
                        {
                            self.sync_fen_from_board();
                            self.scene_rebuild_requested = true;
                        }
                    });
                    if !self.game.move_labels().is_empty() {
                        let moves = self
                            .game
                            .move_labels()
                            .iter()
                            .enumerate()
                            .map(|(index, chess_move)| {
                                if index % 2 == 0 {
                                    format!("{}. {chess_move}", index / 2 + 1)
                                } else {
                                    chess_move.clone()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("   ");
                        ui.label(RichText::new(moves).small().color(Color32::from_gray(150)));
                    }

                    ui.add_space(8.0);
                    ui.label(RichText::new("FEN position").strong());
                    let fen_response = ui.add(
                        egui::TextEdit::singleline(&mut self.fen_input)
                            .hint_text("Paste a FEN position…")
                            .desired_width(f32::INFINITY),
                    );
                    let load_on_enter = fen_response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    let mut load_clicked = false;
                    ui.horizontal(|ui| {
                        if ui.button("Paste").clicked() {
                            self.fen_input.clear();
                            fen_response.request_focus();
                            ui.ctx()
                                .send_viewport_cmd(egui::ViewportCommand::RequestPaste);
                        }
                        load_clicked = ui.button("Load FEN").clicked();
                    });
                    if load_clicked || load_on_enter {
                        self.load_fen();
                    }
                    ui.label(
                        RichText::new("Use Paste or Ctrl/Cmd+V, then press Enter or Load FEN.")
                            .small()
                            .color(Color32::from_gray(135)),
                    );
                    if let Some(error) = &self.fen_error {
                        ui.label(
                            RichText::new(error)
                                .small()
                                .color(Color32::from_rgb(238, 106, 116)),
                        );
                    }

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Piece materials");
                    let mut changed = false;
                    changed |= piece_material_controls(
                        ui,
                        "Light pieces",
                        true,
                        &mut self.settings.light_style,
                        &mut self.settings.light_material,
                    );
                    ui.add_space(6.0);
                    changed |= piece_material_controls(
                        ui,
                        "Dark pieces",
                        false,
                        &mut self.settings.dark_style,
                        &mut self.settings.dark_material,
                    );

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Board");
                    let old_board_style = self.settings.board_style;
                    egui::ComboBox::from_id_salt("board-style")
                        .selected_text(self.settings.board_style.label())
                        .show_ui(ui, |ui| {
                            for style in BoardStyle::ALL {
                                ui.selectable_value(
                                    &mut self.settings.board_style,
                                    style,
                                    style.label(),
                                );
                            }
                        });
                    if old_board_style != self.settings.board_style {
                        self.settings.set_board_style(self.settings.board_style);
                        changed = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Light square");
                        changed |= ui
                            .color_edit_button_rgb(&mut self.settings.light_square)
                            .changed();
                        ui.label("Dark square");
                        changed |= ui
                            .color_edit_button_rgb(&mut self.settings.dark_square)
                            .changed();
                    });
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.board_roughness, 0.015..=0.8)
                                .text("Roughness"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.board_metallic, 0.0..=1.0)
                                .text("Metallic"),
                        )
                        .changed();

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Optics");
                    changed |= ui
                        .checkbox(&mut self.settings.reflections, "Recursive reflections")
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.settings.refractions, "Dielectric refraction")
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.settings.caustics, "Caustic focus")
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.settings.soft_shadows, "Soft ray-traced shadows")
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.samples, 1..=16)
                                .text("Rays / pixel"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.max_bounces, 1..=7)
                                .text("Bounces"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.exposure, 0.35..=2.5)
                                .text("Exposure"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.light_radius, 0.0..=2.5)
                                .text("Light radius"),
                        )
                        .changed();
                    self.mark_render_changed(changed);

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Progressive accumulation");
                    let mut schedule_changed = ui
                        .checkbox(
                            &mut self.continuous_accumulation,
                            "Continuous — accumulate until paused",
                        )
                        .changed();
                    schedule_changed |= ui
                        .add_enabled(
                            !self.continuous_accumulation,
                            egui::Slider::new(
                                &mut self.accumulation_pass_limit,
                                8..=MAX_ACCUMULATION_PASS_LIMIT,
                            )
                            .logarithmic(true)
                            .text("Pass limit"),
                        )
                        .changed();
                    if schedule_changed {
                        self.update_render_schedule(context);
                    }
                    if self.continuous_accumulation {
                        ui.label(
                            RichText::new(format!(
                                "{} passes accumulated; use Pause in the top bar to stop.",
                                self.accumulation_passes
                            ))
                            .small()
                            .color(Color32::from_gray(145)),
                        );
                    } else {
                        let target_rays = u64::from(self.settings.samples)
                            * u64::from(self.accumulation_pass_limit);
                        ui.label(
                            RichText::new(format!(
                                "{} / {} passes · up to {target_rays} rays/pixel",
                                self.accumulation_passes, self.accumulation_pass_limit
                            ))
                            .small()
                            .color(Color32::from_gray(145)),
                        );
                    }

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(6.0);
                    piece_asset_credits(ui);
                });
            });
    }

    fn chess_board(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_width().min(292.0);
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
        let cell = size / 8.0;
        let painter = ui.painter_at(rect);
        let legal = self.game.legal_destinations();
        for screen_rank in 0..8 {
            let rank = 7 - screen_rank;
            for file in 0..8 {
                let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
                let min = rect.min + Vec2::new(file as f32 * cell, screen_rank as f32 * cell);
                let square_rect = Rect::from_min_size(min, Vec2::splat(cell));
                let light = (rank + file) % 2 == 1;
                let base_color = if light {
                    Color32::from_rgb(174, 167, 148)
                } else {
                    Color32::from_rgb(43, 72, 75)
                };
                painter.rect_filled(square_rect, 0.0, base_color);
                if self.game.selected() == Some(square) {
                    painter.rect_stroke(
                        square_rect.shrink(1.5),
                        0.0,
                        Stroke::new(3.0, Color32::from_rgb(244, 198, 92)),
                        egui::StrokeKind::Inside,
                    );
                } else if legal.contains(&square) {
                    painter.circle_filled(
                        square_rect.center(),
                        cell * 0.12,
                        Color32::from_rgba_unmultiplied(244, 198, 92, 205),
                    );
                }
                if let (Some(piece), Some(color)) = (
                    self.game.board().piece_on(square),
                    self.game.board().color_on(square),
                ) {
                    let disk = if color == Color::White {
                        Color32::from_rgb(226, 220, 201)
                    } else {
                        Color32::from_rgb(24, 29, 38)
                    };
                    let text_color = if color == Color::White {
                        Color32::from_rgb(37, 42, 48)
                    } else {
                        Color32::from_rgb(226, 220, 201)
                    };
                    painter.circle_filled(square_rect.center(), cell * 0.34, disk);
                    painter.text(
                        square_rect.center(),
                        Align2::CENTER_CENTER,
                        piece_letter(piece),
                        FontId::proportional(cell * 0.48),
                        text_color,
                    );
                }
            }
        }
        if response.clicked() {
            if let Some(position) = response.interact_pointer_pos() {
                let file = ((position.x - rect.left()) / cell).floor() as usize;
                let screen_rank = ((position.y - rect.top()) / cell).floor() as usize;
                if file < 8 && screen_rank < 8 {
                    let rank = 7 - screen_rank;
                    let square =
                        Square::make_square(Rank::from_index(rank), File::from_index(file));
                    if self.game.click(square) == ClickResult::Moved {
                        self.sync_fen_from_board();
                        self.scene_rebuild_requested = true;
                    }
                }
            }
        }
    }

    fn central_view(&mut self, context: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&context.style()).fill(Color32::from_rgb(8, 10, 14)))
            .show(context, |ui| {
                if let Some(error) = &self.error {
                    egui::Frame::group(ui.style())
                        .fill(Color32::from_rgb(52, 20, 24))
                        .show(ui, |ui| {
                            ui.heading("Vulkan ray tracing unavailable");
                            ui.label(error);
                            ui.label("The UI itself is running through wgpu/Vulkan.");
                        });
                    return;
                }
                let Some(texture_id) = self.texture.as_ref().map(TextureHandle::id) else {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                        ui.label("Building acceleration structures…");
                    });
                    return;
                };
                let available = ui.available_size();
                let aspect = self.width as f32 / self.height as f32;
                let mut image_size = Vec2::new(available.x, available.x / aspect);
                if image_size.y > available.y {
                    image_size = Vec2::new(available.y * aspect, available.y);
                }
                self.track_viewport_size(context, image_size);
                let response = ui.add(
                    egui::Image::new((texture_id, image_size))
                        .sense(Sense::click_and_drag())
                        .corner_radius(4.0),
                );
                if response.dragged() {
                    let delta = context.input(|input| input.pointer.delta());
                    self.settings.camera_yaw -= delta.x * 0.006;
                    self.settings.camera_pitch =
                        (self.settings.camera_pitch + delta.y * 0.004).clamp(0.12, 1.25);
                    self.reset_accumulation();
                }
                if response.hovered() {
                    let scroll = context.input(|input| input.raw_scroll_delta.y);
                    if scroll.abs() > 0.01 {
                        self.settings.camera_distance = (self.settings.camera_distance
                            * (-scroll * 0.0015).exp())
                        .clamp(8.2, 22.0);
                        self.reset_accumulation();
                    }
                }
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("drag to orbit  ·  scroll to zoom")
                            .small()
                            .color(Color32::from_gray(115)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} × {}", self.width, self.height))
                                .small()
                                .color(Color32::from_gray(115)),
                        );
                    });
                });
            });
    }

    fn save_current_frame(&mut self) {
        if self.pixels.is_empty() {
            self.error = Some("There is no completed frame to save yet".to_owned());
            return;
        }
        self.save_counter = self.save_counter.wrapping_add(1);
        let directory = PathBuf::from("renders");
        if let Err(error) = fs::create_dir_all(&directory) {
            self.error = Some(format!("Could not create renders/: {error}"));
            return;
        }
        let path = directory.join(format!("chess-rtx-{:03}.png", self.save_counter));
        match image::save_buffer(
            &path,
            &self.pixels,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        ) {
            Ok(()) => self.status = format!("Saved {}", path.display()),
            Err(error) => self.error = Some(format!("Could not save {}: {error}", path.display())),
        }
    }
}

fn interactive_render_dimensions(logical_size: Vec2, pixels_per_point: f32) -> [u32; 2] {
    let scale = if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let dimension = |points: f32| {
        (points.max(1.0) * scale).round().clamp(
            MIN_INTERACTIVE_RENDER_DIMENSION as f32,
            MAX_INTERACTIVE_RENDER_DIMENSION as f32,
        ) as u32
    };
    [dimension(logical_size.x), dimension(logical_size.y)]
}

impl eframe::App for ChessRtxApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.top_bar(context);
        self.side_panel(context);
        self.central_view(context);
        self.render_if_needed(context);
    }
}

fn piece_asset_credits(ui: &mut egui::Ui) {
    ui.collapsing("Piece asset credits", |ui| {
        ui.label("High-detail Staunton models by Jeyhun1985, provided via Sketchfab:");
        for (piece, url) in [
            (
                "Pawn",
                "https://sketchfab.com/3d-models/pawn-staunton-full-size-chess-set-ba992ea7fa094032a521c474e9aea09f",
            ),
            (
                "Rook",
                "https://sketchfab.com/3d-models/rook-staunton-full-size-chess-set-39a02aa72ed64856a68f039db0626538",
            ),
            (
                "Knight",
                "https://sketchfab.com/3d-models/knight-staunton-full-size-chess-set-dffe461e0def4f0a84bfe8551e58105f",
            ),
            (
                "Bishop",
                "https://sketchfab.com/3d-models/bishop-staunton-full-size-chess-set-945dd86e3d6b4db2ad66a29a87ee9ef2",
            ),
            (
                "Queen",
                "https://sketchfab.com/3d-models/queen-staunton-full-size-chess-set-2f47e64f6f994c41bff0d58d42bd4475",
            ),
            (
                "King",
                "https://sketchfab.com/3d-models/king-staunton-full-size-chess-set-cc297467241f42e1a384c7e5d0369143",
            ),
        ] {
            ui.hyperlink_to(piece, url);
        }
        ui.hyperlink_to(
            "Creative Commons Attribution 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        );
    });
}

fn piece_material_controls(
    ui: &mut egui::Ui,
    title: &str,
    light: bool,
    style: &mut PieceStyle,
    material: &mut GpuMaterial,
) -> bool {
    let mut changed = false;
    ui.label(RichText::new(title).strong());
    let old_style = *style;
    egui::ComboBox::from_id_salt(format!("piece-style-{title}"))
        .selected_text(style.label())
        .show_ui(ui, |ui| {
            for preset in PieceStyle::ALL {
                ui.selectable_value(style, preset, preset.label());
            }
        });
    if old_style != *style {
        *material = style.material(light);
        changed = true;
    }
    changed |= ui
        .color_edit_button_rgba_unmultiplied(&mut material.base_color)
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.surface[0], 0.015..=0.8).text("Roughness"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.surface[1], 0.0..=1.0).text("Metallic"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.surface[2], 0.0..=1.0).text("Transmission"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.surface[3], 1.01..=2.4).text("IOR"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.optics[0], -0.95..=0.95).text("Anisotropy"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut material.optics[1], 0.0..=2.0).text("Caustics"))
        .changed();
    changed
}

const fn piece_letter(piece: Piece) -> &'static str {
    match piece {
        Piece::Pawn => "P",
        Piece::Knight => "N",
        Piece::Bishop => "B",
        Piece::Rook => "R",
        Piece::Queen => "Q",
        Piece::King => "K",
    }
}

fn configure_style(context: &egui::Context) {
    let mut style = (*context.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = Color32::from_rgb(17, 20, 27);
    style.visuals.window_fill = Color32::from_rgb(17, 20, 27);
    style.visuals.extreme_bg_color = Color32::from_rgb(9, 11, 16);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(28, 32, 41);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(42, 48, 60);
    style.spacing.item_spacing = Vec2::new(8.0, 7.0);
    context.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_resolution_matches_physical_viewport_pixels() {
        assert_eq!(
            interactive_render_dimensions(Vec2::new(1_200.0, 700.0), 1.0),
            [1_200, 700]
        );
        assert_eq!(
            interactive_render_dimensions(Vec2::new(800.0, 450.0), 2.0),
            [1_600, 900]
        );
    }

    #[test]
    fn interactive_resolution_is_safely_bounded() {
        assert_eq!(
            interactive_render_dimensions(Vec2::new(0.0, f32::INFINITY), f32::NAN),
            [
                MIN_INTERACTIVE_RENDER_DIMENSION,
                MAX_INTERACTIVE_RENDER_DIMENSION
            ]
        );
    }
}
