use std::{
    sync::{
        mpsc::{self, Receiver, TryRecvError},
        Arc, Condvar, Mutex, MutexGuard,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use chess::Board;
use eframe::egui;

use crate::{material::RenderSettings, rt::RayTracer, scene::ChessScene};

#[derive(Clone)]
pub(crate) struct RenderRequest {
    pub generation: u64,
    pub scene_revision: u64,
    pub board: Board,
    pub dimensions: [u32; 2],
    pub settings: RenderSettings,
}

pub(crate) struct RenderedFrame {
    pub generation: u64,
    pub scene_revision: u64,
    pub dimensions: [usize; 2],
    pub pixels: Vec<u8>,
    pub samples_per_pixel: u32,
    pub accumulated_samples_per_pixel: u64,
    pub elapsed_milliseconds: f32,
    pub device_name: String,
    pub geometry_label: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderFailureStage {
    Initialization,
    Resize,
    Dispatch,
}

pub(crate) enum RenderEvent {
    Frame(RenderedFrame),
    Failed {
        generation: u64,
        stage: RenderFailureStage,
        message: String,
    },
}

#[derive(Default)]
struct WorkerControl {
    pending: Option<RenderRequest>,
    shutdown: bool,
}

/// Owns the Vulkan ray tracer on a background thread.
///
/// There is only ever one pending request. Replacing that request lets rapid UI changes coalesce
/// while an older GPU pass finishes instead of building an increasingly stale render queue.
pub(crate) struct RenderWorker {
    control: Arc<(Mutex<WorkerControl>, Condvar)>,
    events: Receiver<RenderEvent>,
    thread: Option<JoinHandle<()>>,
}

impl RenderWorker {
    pub(crate) fn spawn(context: egui::Context) -> std::io::Result<Self> {
        let control = Arc::new((Mutex::new(WorkerControl::default()), Condvar::new()));
        let worker_control = Arc::clone(&control);
        let (event_sender, events) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("chess-rtx-renderer".to_owned())
            .spawn(move || worker_loop(&worker_control, &event_sender, &context))?;
        Ok(Self {
            control,
            events,
            thread: Some(thread),
        })
    }

    pub(crate) fn submit(&self, request: RenderRequest) {
        let (state, wake) = &*self.control;
        lock_unpoisoned(state).pending = Some(request);
        wake.notify_one();
    }

    pub(crate) fn cancel_pending(&self) -> bool {
        lock_unpoisoned(&self.control.0).pending.take().is_some()
    }

    pub(crate) fn try_recv(&self) -> Result<RenderEvent, TryRecvError> {
        self.events.try_recv()
    }
}

impl Drop for RenderWorker {
    fn drop(&mut self) {
        let (state, wake) = &*self.control;
        {
            let mut state = lock_unpoisoned(state);
            state.shutdown = true;
            state.pending = None;
        }
        wake.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn next_request(control: &Arc<(Mutex<WorkerControl>, Condvar)>) -> Option<RenderRequest> {
    let (state, wake) = &**control;
    let mut state = lock_unpoisoned(state);
    while state.pending.is_none() && !state.shutdown {
        state = wake
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    if state.shutdown {
        None
    } else {
        state.pending.take()
    }
}

fn worker_loop(
    control: &Arc<(Mutex<WorkerControl>, Condvar)>,
    events: &mpsc::Sender<RenderEvent>,
    context: &egui::Context,
) {
    let mut renderer: Option<RayTracer> = None;
    let mut active_scene_revision = None;
    let mut active_generation = None;
    let mut geometry_label = "loading piece geometry";

    while let Some(request) = next_request(control) {
        if renderer.is_none() || active_scene_revision != Some(request.scene_revision) {
            let scene = ChessScene::from_board(&request.board);
            geometry_label = scene.geometry_label();
            match RayTracer::new(request.dimensions[0], request.dimensions[1], &scene) {
                Ok(new_renderer) => {
                    renderer = Some(new_renderer);
                    active_scene_revision = Some(request.scene_revision);
                    active_generation = None;
                }
                Err(error) => {
                    send_event(
                        events,
                        context,
                        RenderEvent::Failed {
                            generation: request.generation,
                            stage: RenderFailureStage::Initialization,
                            message: format!("{error:#}"),
                        },
                    );
                    continue;
                }
            }
        }

        let ray_tracer = renderer
            .as_mut()
            .expect("the renderer is initialized before resize and dispatch");
        let dimensions = ray_tracer.dimensions();
        if dimensions
            != [
                request.dimensions[0] as usize,
                request.dimensions[1] as usize,
            ]
        {
            if let Err(error) = ray_tracer.resize(request.dimensions[0], request.dimensions[1]) {
                send_event(
                    events,
                    context,
                    RenderEvent::Failed {
                        generation: request.generation,
                        stage: RenderFailureStage::Resize,
                        message: format!("{error:#}"),
                    },
                );
                continue;
            }
            active_generation = None;
        }
        if active_generation != Some(request.generation) {
            ray_tracer.reset_accumulation();
            active_generation = Some(request.generation);
        }

        let samples_per_pixel = request.settings.samples;
        let started = Instant::now();
        match ray_tracer.render(&request.settings) {
            Ok(pixels) => {
                let frame = RenderedFrame {
                    generation: request.generation,
                    scene_revision: request.scene_revision,
                    dimensions: ray_tracer.dimensions(),
                    pixels,
                    samples_per_pixel,
                    accumulated_samples_per_pixel: ray_tracer.accumulated_samples_per_pixel(),
                    elapsed_milliseconds: started.elapsed().as_secs_f32() * 1_000.0,
                    device_name: ray_tracer.device_name().to_owned(),
                    geometry_label,
                };
                if !send_event(events, context, RenderEvent::Frame(frame)) {
                    break;
                }
            }
            Err(error) => {
                send_event(
                    events,
                    context,
                    RenderEvent::Failed {
                        generation: request.generation,
                        stage: RenderFailureStage::Dispatch,
                        message: format!("{error:#}"),
                    },
                );
            }
        }
    }
}

fn send_event(
    events: &mpsc::Sender<RenderEvent>,
    context: &egui::Context,
    event: RenderEvent,
) -> bool {
    if events.send(event).is_err() {
        return false;
    }
    context.request_repaint();
    true
}
