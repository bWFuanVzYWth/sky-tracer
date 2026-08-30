use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use egui::{Color32, RichText};
use sky_unreal_atmosphere_8wave::{AerosolPreset, HillairePhaseMode, HillaireSettings};

use crate::assets::RealtimeAsset;
use crate::catalog::{AssetEntry, CatalogScan};
use crate::controls::{
    DEFAULT_HDR_PAPER_WHITE_NITS, DEFAULT_HDR_PEAK_NITS, DEFAULT_REINHARD_OVEREXPOSURE,
    MAX_ATMOSPHERE_THICKNESS_KM, MAX_EXPOSURE_EV, MAX_HDR_PAPER_WHITE_NITS, MAX_HDR_PEAK_NITS,
    MAX_PLANET_RADIUS_KM, MAX_REINHARD_OVEREXPOSURE, MIN_ATMOSPHERE_THICKNESS_KM, MIN_EXPOSURE_EV,
    MIN_HDR_PAPER_WHITE_NITS, MIN_HDR_PEAK_NITS, MIN_PLANET_RADIUS_KM, MIN_REINHARD_OVEREXPOSURE,
    RealtimeControls,
};
use crate::experiment::CompareMode;
use crate::view::ViewState;

const PANEL_FILL: Color32 = Color32::from_rgb(8, 12, 16);
const TOOLBAR_FILL: Color32 = Color32::from_rgb(11, 17, 22);
const BORDER: Color32 = Color32::from_rgb(35, 49, 58);
const MUTED: Color32 = Color32::from_rgb(137, 154, 164);
const ACCENT: Color32 = Color32::from_rgb(65, 151, 211);
const ERROR: Color32 = Color32::from_rgb(224, 94, 94);

#[derive(Clone, Debug)]
pub enum WorkbenchAction {
    OpenAsset,
    ChooseAssetRoot,
    RefreshCatalog,
    LoadAsset(PathBuf),
}

#[derive(Debug)]
pub struct WorkbenchFrame {
    pub viewport_points: egui::Rect,
    pub actions: Vec<WorkbenchAction>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatusKind {
    Info,
    Error,
}

#[derive(Debug)]
pub struct WorkbenchState {
    pub controls: RealtimeControls,
    asset_root: PathBuf,
    entries: Vec<AssetEntry>,
    current_path: PathBuf,
    filter: String,
    assets_open: bool,
    inspector_open: bool,
    inspector_width: f32,
    ground_albedo_linked: bool,
    scanning: bool,
    scan_generation: u64,
    status: Option<(StatusKind, String)>,
}

impl WorkbenchState {
    pub fn new(asset: &RealtimeAsset) -> Self {
        let current_path = normalized_path(asset.manifest_path());
        let asset_root = current_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_owned();
        Self {
            controls: RealtimeControls::from_asset(asset),
            entries: vec![AssetEntry::from_path(current_path.clone(), &asset_root)],
            current_path,
            asset_root,
            filter: String::new(),
            assets_open: true,
            inspector_open: true,
            inspector_width: 340.0,
            ground_albedo_linked: true,
            scanning: false,
            scan_generation: 0,
            status: None,
        }
    }

    pub fn asset_root(&self) -> &Path {
        &self.asset_root
    }

    pub fn set_asset_root(&mut self, root: PathBuf) {
        self.asset_root = normalized_path(&root);
    }

    pub fn begin_scan(&mut self) -> (u64, PathBuf) {
        self.scan_generation = self.scan_generation.wrapping_add(1);
        self.scanning = true;
        self.status = Some((
            StatusKind::Info,
            format!("Scanning {}...", display_path(&self.asset_root)),
        ));
        (self.scan_generation, self.asset_root.clone())
    }

    pub fn apply_scan(&mut self, generation: u64, result: Result<CatalogScan, String>) {
        if generation != self.scan_generation {
            return;
        }
        self.scanning = false;
        match result {
            Ok(scan) => {
                self.asset_root = scan.root;
                self.entries = scan.entries;
                let invalid = self
                    .entries
                    .iter()
                    .filter(|entry| !entry.is_valid())
                    .count();
                self.status = Some((
                    StatusKind::Info,
                    format!(
                        "Found {} assets{}.",
                        self.entries.len().saturating_sub(invalid),
                        if invalid == 0 {
                            String::new()
                        } else {
                            format!("; {invalid} invalid")
                        }
                    ),
                ));
            }
            Err(error) => self.status = Some((StatusKind::Error, error)),
        }
    }

    pub fn set_current_asset(&mut self, asset: &RealtimeAsset) {
        self.current_path = normalized_path(asset.manifest_path());
        self.controls.sync_asset_bound(asset);
        if !self
            .entries
            .iter()
            .any(|entry| normalized_path(&entry.path) == self.current_path)
        {
            self.entries.push(AssetEntry::from_path(
                self.current_path.clone(),
                &self.asset_root,
            ));
            self.entries
                .sort_by_key(|entry| entry.path.to_string_lossy().to_lowercase());
        }
        self.status = Some((
            StatusKind::Info,
            format!("Loaded {}", display_path(&self.current_path)),
        ));
    }

    pub fn set_error(&mut self, message: impl Into<String>) {
        self.status = Some((StatusKind::Error, message.into()));
    }

    pub fn step_asset(&self, delta: i32) -> Option<PathBuf> {
        let visible = self.filtered_indices();
        if visible.len() < 2 {
            return None;
        }
        let current = visible
            .iter()
            .position(|index| normalized_path(&self.entries[*index].path) == self.current_path);
        let next = match (current, delta.signum()) {
            (Some(index), -1) => (index + visible.len() - 1) % visible.len(),
            (Some(index), _) => (index + 1) % visible.len(),
            (None, -1) => visible.len() - 1,
            (None, _) => 0,
        };
        Some(self.entries[visible[next]].path.clone())
    }

    pub fn show(
        &mut self,
        root: &mut egui::Ui,
        asset: &RealtimeAsset,
        reference_available: bool,
        hdr_supported: bool,
        hdr_active: bool,
    ) -> WorkbenchFrame {
        let mut actions = Vec::new();

        self.top_toolbar(
            root,
            asset,
            reference_available,
            hdr_supported,
            &mut actions,
        );
        self.status_bar(root, asset, reference_available, hdr_supported, hdr_active);
        if self.assets_open {
            self.assets_panel(root, &mut actions);
        }
        let remaining = root.available_rect_before_wrap();
        let viewport_points = if self.inspector_open {
            let maximum_width = (remaining.width() - 320.0).clamp(280.0, 480.0);
            self.inspector_width = self.inspector_width.clamp(280.0, maximum_width);

            let divider_x = remaining.max.x - self.inspector_width;
            let divider = egui::Rect::from_min_max(
                egui::pos2(divider_x - 3.0, remaining.min.y),
                egui::pos2(divider_x + 3.0, remaining.max.y),
            );
            let resize = root.interact(
                divider,
                root.make_persistent_id("workbench-inspector-resize"),
                egui::Sense::drag(),
            );
            if resize.dragged() {
                self.inspector_width =
                    (self.inspector_width - resize.drag_delta().x).clamp(280.0, maximum_width);
            }
            if resize.hovered() || resize.dragged() {
                root.ctx()
                    .set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }

            let divider_x = remaining.max.x - self.inspector_width;
            let viewport = egui::Rect::from_min_max(
                remaining.min,
                egui::pos2(divider_x - 1.0, remaining.max.y),
            );
            let inspector =
                egui::Rect::from_min_max(egui::pos2(divider_x, remaining.min.y), remaining.max);
            self.inspector_at(root, inspector, asset, hdr_supported);
            self.viewport_at(root, viewport, reference_available)
        } else {
            self.viewport_at(root, remaining, reference_available)
        };
        self.controls = self.controls.normalized();

        WorkbenchFrame {
            viewport_points,
            actions,
        }
    }

    fn top_toolbar(
        &mut self,
        root: &mut egui::Ui,
        asset: &RealtimeAsset,
        reference_available: bool,
        hdr_supported: bool,
        actions: &mut Vec<WorkbenchAction>,
    ) {
        egui::Panel::top("workbench-toolbar")
            .exact_size(44.0)
            .frame(
                egui::Frame::default()
                    .fill(TOOLBAR_FILL)
                    .inner_margin(egui::Margin::symmetric(10, 7))
                    .stroke(egui::Stroke::new(1.0, BORDER)),
            )
            .show(root, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui
                        .button("Open...")
                        .on_hover_text("Open an asset.json (Ctrl+O)")
                        .clicked()
                    {
                        actions.push(WorkbenchAction::OpenAsset);
                    }
                    if ui
                        .add_enabled(self.step_asset(-1).is_some(), egui::Button::new("Prev"))
                        .on_hover_text("Previous visible asset (Page Up)")
                        .clicked()
                        && let Some(path) = self.step_asset(-1)
                    {
                        actions.push(WorkbenchAction::LoadAsset(path));
                    }
                    if ui
                        .add_enabled(self.step_asset(1).is_some(), egui::Button::new("Next"))
                        .on_hover_text("Next visible asset (Page Down)")
                        .clicked()
                        && let Some(path) = self.step_asset(1)
                    {
                        actions.push(WorkbenchAction::LoadAsset(path));
                    }

                    ui.separator();
                    for (mode, label, shortcut) in [
                        (CompareMode::Realtime, "Realtime", "1"),
                        (CompareMode::Reference, "Reference", "2"),
                        (CompareMode::AbsoluteDifference, "Absolute Diff", "3"),
                        (CompareMode::SignedDifference, "Signed Diff", "4"),
                    ] {
                        let enabled = mode == CompareMode::Realtime || reference_available;
                        let selected = self.controls.compare_mode == mode;
                        let response =
                            ui.add_enabled(enabled, egui::Button::new(label).selected(selected));
                        let response = if enabled {
                            response.on_hover_text(format!("Comparison mode ({shortcut})"))
                        } else {
                            response.on_disabled_hover_text(
                                "The current asset has no readable reference EXR",
                            )
                        };
                        if response.clicked() {
                            self.controls.compare_mode = mode;
                        }
                    }

                    ui.separator();
                    if ui
                        .button("Reset All")
                        .on_hover_text("Restore renderer defaults")
                        .clicked()
                    {
                        self.controls.reset_all(asset);
                        self.ground_albedo_linked = true;
                    }

                    let hdr = ui.add_enabled(
                        hdr_supported,
                        egui::Button::new("HDR").selected(self.controls.hdr_enabled),
                    );
                    let hdr = if hdr_supported {
                        hdr.on_hover_text("Toggle 16-bit floating-point scRGB output (F6)")
                    } else {
                        hdr.on_disabled_hover_text(
                            "The active adapter/surface does not expose scRGB Rgba16Float",
                        )
                    };
                    if hdr.clicked() {
                        self.controls.hdr_enabled = !self.controls.hdr_enabled;
                    }

                    ui.separator();
                    if ui.selectable_label(self.assets_open, "Assets").clicked() {
                        self.assets_open = !self.assets_open;
                    }
                    if ui
                        .selectable_label(self.inspector_open, "Inspector")
                        .clicked()
                    {
                        self.inspector_open = !self.inspector_open;
                    }
                });
            });
    }

    fn status_bar(
        &self,
        root: &mut egui::Ui,
        asset: &RealtimeAsset,
        reference_available: bool,
        hdr_supported: bool,
        hdr_active: bool,
    ) {
        egui::Panel::bottom("workbench-status")
            .exact_size(28.0)
            .frame(
                egui::Frame::default()
                    .fill(TOOLBAR_FILL)
                    .inner_margin(egui::Margin::symmetric(10, 5))
                    .stroke(egui::Stroke::new(1.0, BORDER)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    let manifest = asset.manifest();
                    ui.label(
                        RichText::new(format!(
                            "{} × {}  ·  {} bands  ·  {} spp",
                            manifest.dimensions[0],
                            manifest.dimensions[1],
                            manifest.band_centers_nm.len(),
                            manifest.spp
                        ))
                        .color(MUTED),
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(if hdr_active {
                            "HDR scRGB active"
                        } else if hdr_supported {
                            "SDR output"
                        } else {
                            "SDR output · HDR unavailable"
                        })
                        .color(if hdr_active { ACCENT } else { MUTED }),
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(if reference_available {
                            "Reference ready"
                        } else {
                            "Reference missing"
                        })
                        .color(if reference_available {
                            ACCENT
                        } else {
                            ERROR
                        }),
                    );
                    if self.scanning {
                        ui.separator();
                        ui.spinner();
                        ui.label(RichText::new("Scanning assets").color(MUTED));
                    }
                    if let Some((kind, message)) = &self.status {
                        ui.separator();
                        ui.label(RichText::new(message).color(if *kind == StatusKind::Error {
                            ERROR
                        } else {
                            MUTED
                        }));
                    }
                });
            });
    }

    fn assets_panel(&mut self, root: &mut egui::Ui, actions: &mut Vec<WorkbenchAction>) {
        egui::Panel::left("workbench-assets")
            .resizable(true)
            .default_size(260.0)
            .size_range(190.0..=440.0)
            .frame(
                egui::Frame::default()
                    .fill(PANEL_FILL)
                    .inner_margin(egui::Margin::same(10))
                    .stroke(egui::Stroke::new(1.0, BORDER)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Assets");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("Refresh")
                            .on_hover_text("Rescan root (F5)")
                            .clicked()
                        {
                            actions.push(WorkbenchAction::RefreshCatalog);
                        }
                        if ui
                            .small_button("Folder...")
                            .on_hover_text("Choose scan root")
                            .clicked()
                        {
                            actions.push(WorkbenchAction::ChooseAssetRoot);
                        }
                    });
                });
                ui.label(
                    RichText::new(display_path(&self.asset_root))
                        .small()
                        .color(MUTED),
                )
                .on_hover_text(display_path(&self.asset_root));
                ui.add_space(6.0);
                ui.add(
                    egui::TextEdit::singleline(&mut self.filter)
                        .hint_text("Filter path or type...")
                        .desired_width(f32::INFINITY),
                );
                ui.separator();

                let visible = self.matching_indices();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if visible.is_empty() {
                            ui.label(RichText::new("No matching asset.json files.").color(MUTED));
                        }
                        for index in visible {
                            let entry = &self.entries[index];
                            let selected = normalized_path(&entry.path) == self.current_path;
                            let label = entry.relative_path.to_string_lossy();
                            let response = ui.selectable_label(
                                selected,
                                RichText::new(label.as_ref()).color(if entry.is_valid() {
                                    Color32::from_rgb(220, 230, 235)
                                } else {
                                    ERROR
                                }),
                            );
                            let response = if let Some(error) = &entry.error {
                                response.on_hover_text(error)
                            } else {
                                response.on_hover_text(display_path(&entry.path))
                            };
                            if response.clicked() && entry.is_valid() && !selected {
                                actions.push(WorkbenchAction::LoadAsset(entry.path.clone()));
                            }
                            if let Some(metadata) = &entry.metadata {
                                ui.label(
                                    RichText::new(format!(
                                        "{}  ·  {}×{}  ·  {} bands  ·  {} spp  ·  sun {:.1}°{}",
                                        metadata.kind,
                                        metadata.dimensions[0],
                                        metadata.dimensions[1],
                                        metadata.bands,
                                        metadata.spp,
                                        metadata.sun_elevation_deg,
                                        if metadata.reference_available {
                                            ""
                                        } else {
                                            "  ·  no EXR"
                                        }
                                    ))
                                    .small()
                                    .color(MUTED),
                                );
                            }
                            ui.add_space(6.0);
                        }
                    });
            });
    }

    fn inspector_at(
        &mut self,
        root: &mut egui::Ui,
        rect: egui::Rect,
        asset: &RealtimeAsset,
        hdr_supported: bool,
    ) {
        root.painter().rect_filled(rect, 0.0, PANEL_FILL);
        root.painter().line_segment(
            [rect.left_top(), rect.left_bottom()],
            egui::Stroke::new(1.0, BORDER),
        );
        let content_rect = rect.shrink(12.0);
        root.scope_builder(
            egui::UiBuilder::new()
                .id_salt("workbench-inspector-content")
                .max_rect(content_rect),
            |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.heading("Inspector");
                        ui.add_space(8.0);

                        if section_header(ui, "View") {
                            self.controls.reset_view();
                        }
                        resettable_slider(
                            ui,
                            &mut self.controls.view.yaw_deg,
                            0.0..=360.0,
                            "Yaw",
                            ViewState::default().yaw_deg,
                            "Horizontal view direction. Drag the viewport with the left mouse button.",
                        );
                        resettable_slider(
                            ui,
                            &mut self.controls.view.pitch_deg,
                            ViewState::MIN_PITCH_DEG..=ViewState::MAX_PITCH_DEG,
                            "Pitch",
                            ViewState::default().pitch_deg,
                            "Vertical view direction.",
                        );
                        resettable_slider(
                            ui,
                            &mut self.controls.view.fov_y_deg,
                            ViewState::MIN_FOV_Y_DEG..=ViewState::MAX_FOV_Y_DEG,
                            "Vertical FOV",
                            ViewState::default().fov_y_deg,
                            "Vertical field of view. Scroll over the viewport to zoom.",
                        );

                        ui.add_space(10.0);
                        if section_header(ui, "Sun & Observer") {
                            self.controls.reset_sun_observer(asset);
                        }
                        resettable_slider(
                            ui,
                            &mut self.controls.sun_azimuth_deg,
                            0.0..=360.0,
                            "Azimuth",
                            asset.manifest().sun_azimuth_deg.rem_euclid(360.0),
                            "Sun direction around the horizon.",
                        );
                        resettable_slider(
                            ui,
                            &mut self.controls.sun_elevation_deg,
                            -90.0..=90.0,
                            "Elevation",
                            asset.manifest().sun_elevation_deg.clamp(-90.0, 90.0),
                            "Sun height above the horizon. Use [ and ] for one-degree steps.",
                        );
                        let maximum_altitude = self.controls.maximum_observer_altitude_km();
                        resettable_slider(
                            ui,
                            &mut self.controls.observer_altitude_km,
                            0.0..=maximum_altitude,
                            "Observer altitude (km)",
                            asset
                                .manifest()
                                .observer_altitude_km
                                .clamp(0.0, maximum_altitude),
                            "Observer altitude above the planet surface.",
                        );

                        ui.add_space(10.0);
                        if section_header(ui, "Display") {
                            self.controls.reset_display();
                        }
                        resettable_slider(
                            ui,
                            &mut self.controls.exposure_ev,
                            MIN_EXPOSURE_EV..=MAX_EXPOSURE_EV,
                            "Exposure compensation (EV)",
                            0.0,
                            "Relative to the calibrated 0.10 display exposure; each EV doubles or halves it.",
                        );
                        resettable_log_slider(
                            ui,
                            &mut self.controls.difference_scale,
                            0.25..=32.0,
                            "Difference scale",
                            4.0,
                            "Magnification applied in absolute and signed difference modes.",
                        );
                        ui.checkbox(&mut self.controls.tone_mapping_enabled, "Tone mapping")
                            .on_hover_text(
                                "Apply the drt-bench Enhanced Reinhard curve and Oklab gamut mapping.",
                            );
                        let hdr = ui.add_enabled(
                            hdr_supported,
                            egui::Checkbox::new(&mut self.controls.hdr_enabled, "HDR scRGB output"),
                        );
                        if hdr_supported {
                            hdr.on_hover_text(
                                "Use a 16-bit floating-point scRGB surface. Enable HDR in Windows for physical HDR output.",
                            );
                        } else {
                            hdr.on_disabled_hover_text(
                                "The active adapter/surface does not expose scRGB Rgba16Float",
                            );
                        }
                        ui.add_enabled_ui(self.controls.tone_mapping_enabled, |ui| {
                            resettable_slider(
                                ui,
                                &mut self.controls.reinhard_overexposure,
                                MIN_REINHARD_OVEREXPOSURE..=MAX_REINHARD_OVEREXPOSURE,
                                "Reinhard asymptote",
                                DEFAULT_REINHARD_OVEREXPOSURE,
                                "Enhanced Reinhard overexposure asymptote. Middle gray remains fixed at 18%.",
                            );
                        });
                        ui.add_enabled_ui(self.controls.hdr_enabled && hdr_supported, |ui| {
                            resettable_slider(
                                ui,
                                &mut self.controls.hdr_paper_white_nits,
                                MIN_HDR_PAPER_WHITE_NITS..=MAX_HDR_PAPER_WHITE_NITS,
                                "HDR paper white (nits)",
                                DEFAULT_HDR_PAPER_WHITE_NITS,
                                "Luminance assigned to mapped display white in the scRGB surface.",
                            );
                            resettable_slider(
                                ui,
                                &mut self.controls.hdr_peak_nits,
                                MIN_HDR_PEAK_NITS..=MAX_HDR_PEAK_NITS,
                                "HDR peak clamp (nits)",
                                DEFAULT_HDR_PEAK_NITS,
                                "Safety clamp for untone-mapped HDR highlights.",
                            );
                        });
                        ui.label(
                            RichText::new(if !self.controls.tone_mapping_enabled {
                                if self.controls.hdr_enabled {
                                    "Output: raw exposed linear scRGB"
                                } else {
                                    "Output: raw exposed linear sRGB (clipped)"
                                }
                            } else if self.controls.hdr_enabled {
                                "Output: linear scRGB / Enhanced Reinhard gamut"
                            } else {
                                "Output: sRGB / Enhanced Reinhard gamut"
                            })
                                .small()
                                .color(MUTED),
                        );

                        ui.add_space(10.0);
                        if section_header(ui, "Atmosphere") {
                            self.controls.reset_atmosphere();
                        }
                        const MONTHS: [&str; 12] = [
                            "January", "February", "March", "April", "May", "June", "July",
                            "August", "September", "October", "November", "December",
                        ];
                        egui::ComboBox::from_label("Ozone month")
                            .selected_text(MONTHS[self.controls.month as usize % 12])
                            .show_ui(ui, |ui| {
                                for (index, month) in MONTHS.iter().enumerate() {
                                    ui.selectable_value(&mut self.controls.month, index as u32, *month);
                                }
                            });
                        let atmosphere_defaults = sky_unreal_atmosphere_8wave::HillaireAtmosphere::default();
                        resettable_slider(
                            ui,
                            &mut self.controls.planet_radius_km,
                            MIN_PLANET_RADIUS_KM..=MAX_PLANET_RADIUS_KM,
                            "Planet radius (km)",
                            atmosphere_defaults.bottom_radius_m * 0.001,
                            "Changing geometry rebuilds the static atmosphere LUTs.",
                        );
                        resettable_slider(
                            ui,
                            &mut self.controls.atmosphere_thickness_km,
                            MIN_ATMOSPHERE_THICKNESS_KM..=MAX_ATMOSPHERE_THICKNESS_KM,
                            "Atmosphere thickness (km)",
                            (atmosphere_defaults.top_radius_m - atmosphere_defaults.bottom_radius_m)
                                * 0.001,
                            "Changing geometry rebuilds the static atmosphere LUTs.",
                        );

                        ui.add_space(10.0);
                        if section_header(ui, "Aerosol") {
                            self.controls.reset_aerosol();
                        }
                        egui::ComboBox::from_label("Preset")
                            .selected_text(aerosol_label(self.controls.aerosol))
                            .show_ui(ui, |ui| {
                                for preset in [
                                    AerosolPreset::RemoteContinental,
                                    AerosolPreset::Rural,
                                    AerosolPreset::ContinentalPolluted,
                                    AerosolPreset::Urban,
                                ] {
                                    ui.selectable_value(
                                        &mut self.controls.aerosol,
                                        preset,
                                        aerosol_label(preset),
                                    );
                                }
                            });
                        resettable_slider(
                            ui,
                            &mut self.controls.aerosol_turbidity,
                            0.0..=10.0,
                            "Turbidity",
                            HillaireSettings::default().aerosol_turbidity,
                            "Scales aerosol density and rebuilds the static atmosphere LUTs.",
                        );
                        egui::ComboBox::from_label("Mie phase")
                            .selected_text(phase_label(self.controls.phase_mode))
                            .show_ui(ui, |ui| {
                                for mode in [
                                    HillairePhaseMode::Lut,
                                    HillairePhaseMode::CornetteShanks,
                                ] {
                                    ui.selectable_value(
                                        &mut self.controls.phase_mode,
                                        mode,
                                        phase_label(mode),
                                    );
                                }
                            });

                        ui.add_space(10.0);
                        if section_header(ui, "Ground Albedo") {
                            self.controls.reset_ground_albedo();
                            self.ground_albedo_linked = true;
                        }
                        ui.checkbox(&mut self.ground_albedo_linked, "Link spectral lanes")
                            .on_hover_text("When linked, changing any lane updates all four lanes.");
                        let defaults = HillaireSettings::default().ground_albedo_spectral;
                        for (index, label) in [
                            "Lane 1 (410/540 nm)",
                            "Lane 2 (440/580 nm)",
                            "Lane 3 (480/610 nm)",
                            "Lane 4 (520/670 nm)",
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let changed = resettable_slider(
                                ui,
                                &mut self.controls.ground_albedo_spectral[index],
                                0.0..=1.0,
                                label,
                                defaults[index],
                                "Spectral ground reflectance. Changing it rebuilds static LUTs.",
                            );
                            if changed && self.ground_albedo_linked {
                                let value = self.controls.ground_albedo_spectral[index];
                                self.controls.ground_albedo_spectral = [value; 4];
                            }
                        }
                    });
            },
        );
    }

    fn viewport_at(
        &mut self,
        root: &mut egui::Ui,
        viewport: egui::Rect,
        reference_available: bool,
    ) -> egui::Rect {
        let response = root.interact(
            viewport,
            root.make_persistent_id("workbench-viewport"),
            egui::Sense::click_and_drag(),
        );
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.controls
                .view
                .orbit_pixels(delta.x as f64, delta.y as f64);
        }
        if response.hovered() {
            let scroll = root.input(|input| input.smooth_scroll_delta().y);
            if scroll.abs() > f32::EPSILON {
                self.controls.view.zoom_steps(scroll / 48.0);
            }
        }
        let hud_rect = egui::Rect::from_min_size(
            viewport.min + egui::vec2(12.0, 12.0),
            egui::vec2(226.0, 56.0),
        );
        root.painter().rect_filled(
            hud_rect,
            5.0,
            Color32::from_rgba_unmultiplied(4, 8, 11, 205),
        );
        root.painter().text(
            hud_rect.min + egui::vec2(10.0, 9.0),
            egui::Align2::LEFT_TOP,
            compare_label(self.controls.compare_mode),
            egui::FontId::proportional(14.0),
            Color32::WHITE,
        );
        root.painter().text(
            hud_rect.min + egui::vec2(10.0, 31.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Sun {:.1}° / {:.1}°  ·  FOV {:.0}°{}",
                self.controls.sun_azimuth_deg,
                self.controls.sun_elevation_deg,
                self.controls.view.fov_y_deg,
                if reference_available {
                    ""
                } else {
                    "  ·  no reference"
                }
            ),
            egui::FontId::proportional(11.0),
            MUTED,
        );
        viewport
    }

    fn filtered_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.is_valid() && entry.matches(&self.filter))
            .map(|(index, _)| index)
            .collect()
    }

    fn matching_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.matches(&self.filter))
            .map(|(index, _)| index)
            .collect()
    }
}

pub fn configure_context(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(9.0, 4.0);
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = PANEL_FILL;
    style.visuals.window_fill = PANEL_FILL;
    style.visuals.extreme_bg_color = Color32::from_rgb(5, 8, 11);
    style.visuals.faint_bg_color = Color32::from_rgb(17, 25, 31);
    style.visuals.selection.bg_fill = ACCENT;
    style.visuals.hyperlink_color = ACCENT;
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(25, 35, 42);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(38, 75, 98);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(48, 112, 154);
    ctx.set_style_of(egui::Theme::Dark, style);
}

fn section_header(ui: &mut egui::Ui, title: &str) -> bool {
    let mut reset = false;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(title)
                .strong()
                .color(Color32::from_rgb(225, 234, 239)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            reset = ui.small_button("Reset").clicked();
        });
    });
    ui.separator();
    reset
}

fn resettable_slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: RangeInclusive<f32>,
    label: &str,
    default: f32,
    tooltip: &str,
) -> bool {
    let response = ui
        .add(
            egui::Slider::new(value, range)
                .text(label)
                .fixed_decimals(2),
        )
        .on_hover_text(format!("{tooltip}\nRight-click to reset."));
    let mut changed = response.changed();
    if response.secondary_clicked() {
        *value = default;
        changed = true;
    }
    changed
}

fn resettable_log_slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: RangeInclusive<f32>,
    label: &str,
    default: f32,
    tooltip: &str,
) -> bool {
    let response = ui
        .add(
            egui::Slider::new(value, range)
                .text(label)
                .logarithmic(true)
                .fixed_decimals(2),
        )
        .on_hover_text(format!("{tooltip}\nRight-click to reset."));
    let mut changed = response.changed();
    if response.secondary_clicked() {
        *value = default;
        changed = true;
    }
    changed
}

fn aerosol_label(preset: AerosolPreset) -> &'static str {
    match preset {
        AerosolPreset::RemoteContinental => "Remote Continental",
        AerosolPreset::Rural => "Rural",
        AerosolPreset::ContinentalPolluted => "Continental Polluted",
        AerosolPreset::Urban => "Urban",
    }
}

fn phase_label(mode: HillairePhaseMode) -> &'static str {
    match mode {
        HillairePhaseMode::Lut => "Measured LUT",
        HillairePhaseMode::CornetteShanks => "Cornette-Shanks",
    }
}

fn compare_label(mode: CompareMode) -> &'static str {
    match mode {
        CompareMode::Realtime => "Realtime",
        CompareMode::Reference => "Offline Reference",
        CompareMode::AbsoluteDifference => "Absolute Difference",
        CompareMode::SignedDifference => "Signed Difference",
    }
}

fn normalized_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_owned()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        }
    })
}

fn display_path(path: &Path) -> String {
    let display = path.display().to_string();
    display.strip_prefix(r"\\?\").unwrap_or(&display).to_owned()
}
