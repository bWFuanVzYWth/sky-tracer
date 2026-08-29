use sky_unreal_atmosphere_8wave::{
    AerosolPreset, HillaireAtmosphere, HillairePhaseMode, HillaireSettings,
};

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::experiment::CompareMode;
use crate::view::ViewState;

pub const MIN_EXPOSURE_EV: f32 = -20.0;
pub const MAX_EXPOSURE_EV: f32 = 20.0;
pub const MIN_PLANET_RADIUS_KM: f32 = 1_000.0;
pub const MAX_PLANET_RADIUS_KM: f32 = 10_000.0;
pub const MIN_ATMOSPHERE_THICKNESS_KM: f32 = 1.0;
pub const MAX_ATMOSPHERE_THICKNESS_KM: f32 = 1_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealtimeControls {
    pub view: ViewState,
    pub compare_mode: CompareMode,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub observer_altitude_km: f32,
    pub exposure_ev: f32,
    pub difference_scale: f32,
    pub planet_radius_km: f32,
    pub atmosphere_thickness_km: f32,
    pub month: u32,
    pub aerosol_turbidity: f32,
    pub ground_albedo_spectral: [f32; 4],
    pub aerosol: AerosolPreset,
    pub phase_mode: HillairePhaseMode,
}

impl RealtimeControls {
    pub fn from_asset(asset: &RealtimeAsset) -> Self {
        let manifest = asset.manifest();
        let atmosphere = HillaireAtmosphere::default();
        let settings = HillaireSettings::default();
        Self {
            view: ViewState::default(),
            compare_mode: CompareMode::default(),
            sun_azimuth_deg: manifest.sun_azimuth_deg,
            sun_elevation_deg: manifest.sun_elevation_deg,
            observer_altitude_km: manifest.observer_altitude_km,
            exposure_ev: 0.0,
            difference_scale: 4.0,
            planet_radius_km: atmosphere.bottom_radius_m * 0.001,
            atmosphere_thickness_km: (atmosphere.top_radius_m - atmosphere.bottom_radius_m) * 0.001,
            month: settings.month,
            aerosol_turbidity: settings.aerosol_turbidity,
            ground_albedo_spectral: settings.ground_albedo_spectral,
            aerosol: AerosolPreset::default(),
            phase_mode: HillairePhaseMode::default(),
        }
        .normalized()
    }

    pub fn normalized(mut self) -> Self {
        self.view.normalize();
        self.sun_azimuth_deg = self.sun_azimuth_deg.rem_euclid(360.0);
        self.sun_elevation_deg = self.sun_elevation_deg.clamp(-90.0, 90.0);
        self.exposure_ev = self.exposure_ev.clamp(MIN_EXPOSURE_EV, MAX_EXPOSURE_EV);
        self.difference_scale = self.difference_scale.clamp(0.25, 32.0);
        self.planet_radius_km = self
            .planet_radius_km
            .clamp(MIN_PLANET_RADIUS_KM, MAX_PLANET_RADIUS_KM);
        self.atmosphere_thickness_km = self
            .atmosphere_thickness_km
            .clamp(MIN_ATMOSPHERE_THICKNESS_KM, MAX_ATMOSPHERE_THICKNESS_KM);
        self.observer_altitude_km = self
            .observer_altitude_km
            .clamp(0.0, self.maximum_observer_altitude_km());
        self.month %= 12;
        self.aerosol_turbidity = self.aerosol_turbidity.clamp(0.0, 10.0);
        for albedo in &mut self.ground_albedo_spectral {
            *albedo = albedo.clamp(0.0, 1.0);
        }
        self
    }

    pub fn sync_asset_bound(&mut self, asset: &RealtimeAsset) {
        let manifest = asset.manifest();
        self.sun_azimuth_deg = manifest.sun_azimuth_deg.rem_euclid(360.0);
        self.sun_elevation_deg = manifest.sun_elevation_deg.clamp(-90.0, 90.0);
        self.observer_altitude_km = manifest
            .observer_altitude_km
            .clamp(0.0, self.maximum_observer_altitude_km());
    }

    pub fn reset_all(&mut self, asset: &RealtimeAsset) {
        *self = Self::from_asset(asset);
    }

    pub fn reset_view(&mut self) {
        self.view = ViewState::default();
    }

    pub fn reset_sun_observer(&mut self, asset: &RealtimeAsset) {
        self.sync_asset_bound(asset);
    }

    pub fn reset_display(&mut self) {
        self.compare_mode = CompareMode::default();
        self.exposure_ev = 0.0;
        self.difference_scale = 4.0;
    }

    pub fn reset_atmosphere(&mut self) {
        let defaults = HillaireAtmosphere::default();
        self.planet_radius_km = defaults.bottom_radius_m * 0.001;
        self.atmosphere_thickness_km = (defaults.top_radius_m - defaults.bottom_radius_m) * 0.001;
        self.month = HillaireSettings::default().month;
        self.observer_altitude_km = self
            .observer_altitude_km
            .clamp(0.0, self.maximum_observer_altitude_km());
    }

    pub fn reset_aerosol(&mut self) {
        let defaults = HillaireSettings::default();
        self.aerosol = AerosolPreset::default();
        self.aerosol_turbidity = defaults.aerosol_turbidity;
        self.phase_mode = HillairePhaseMode::default();
    }

    pub fn reset_ground_albedo(&mut self) {
        self.ground_albedo_spectral = HillaireSettings::default().ground_albedo_spectral;
    }

    pub fn maximum_observer_altitude_km(&self) -> f32 {
        (self.atmosphere_thickness_km - 0.001).max(0.0)
    }

    pub fn exposure_multiplier(&self) -> f32 {
        DisplayTransform::default().exposure * self.exposure_ev.exp2()
    }

    pub fn atmosphere(&self) -> HillaireAtmosphere {
        let bottom_radius_m = self.planet_radius_km * 1_000.0;
        HillaireAtmosphere {
            bottom_radius_m,
            top_radius_m: bottom_radius_m + self.atmosphere_thickness_km * 1_000.0,
            world_y0_radius_m: bottom_radius_m + self.observer_altitude_km * 1_000.0,
            scene_units_to_m: 1.0,
        }
    }

    pub fn settings(&self) -> HillaireSettings {
        HillaireSettings {
            month: self.month,
            aerosol_turbidity: self.aerosol_turbidity,
            ground_albedo_spectral: self.ground_albedo_spectral,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use sky_core::asset::{SpectralAssetFiles, SpectralAssetManifest};

    use super::{MAX_EXPOSURE_EV, RealtimeControls};
    use crate::assets::RealtimeAsset;

    static NEXT_TEST_ROOT: AtomicU64 = AtomicU64::new(0);

    fn test_asset() -> RealtimeAsset {
        let root = std::env::temp_dir().join(format!(
            "sky-realtime-controls-{}-{}",
            std::process::id(),
            NEXT_TEST_ROOT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create test root");
        let manifest = SpectralAssetManifest::spectral_panorama(
            [4, 2],
            8,
            1,
            12.0,
            34.0,
            0.2,
            vec![500.0],
            SpectralAssetFiles {
                rgb_exr: "sky.exr".to_owned(),
                rgb_png: "sky.png".to_owned(),
                band_exrs: vec!["band.exr".to_owned()],
            },
        );
        let path = root.join("asset.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        RealtimeAsset::load(path).expect("load test asset")
    }

    #[test]
    fn exposure_ev_is_relative_to_the_current_display_baseline() {
        let asset = test_asset();
        let mut controls = RealtimeControls::from_asset(&asset);
        let baseline = controls.exposure_multiplier();
        controls.exposure_ev = 1.0;
        assert!((controls.exposure_multiplier() - baseline * 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn normalization_clamps_unsafe_values() {
        let asset = test_asset();
        let mut controls = RealtimeControls::from_asset(&asset);
        controls.exposure_ev = 100.0;
        controls.sun_elevation_deg = -120.0;
        controls.atmosphere_thickness_km = 5.0;
        controls.observer_altitude_km = 20.0;
        controls = controls.normalized();
        assert_eq!(controls.exposure_ev, MAX_EXPOSURE_EV);
        assert_eq!(controls.sun_elevation_deg, -90.0);
        assert!(controls.observer_altitude_km < controls.atmosphere_thickness_km);
    }

    #[test]
    fn asset_switch_only_updates_asset_bound_controls() {
        let asset = test_asset();
        let mut controls = RealtimeControls::from_asset(&asset);
        controls.view.yaw_deg = 90.0;
        controls.exposure_ev = 2.0;
        controls.aerosol_turbidity = 4.0;
        controls.sun_elevation_deg = -10.0;
        controls.sync_asset_bound(&asset);
        assert_eq!(controls.view.yaw_deg, 90.0);
        assert_eq!(controls.exposure_ev, 2.0);
        assert_eq!(controls.aerosol_turbidity, 4.0);
        assert_eq!(controls.sun_elevation_deg, 12.0);
    }

    #[test]
    fn group_and_global_resets_restore_their_expected_scope() {
        let asset = test_asset();
        let defaults = RealtimeControls::from_asset(&asset);
        let mut controls = defaults;
        controls.view.yaw_deg = 90.0;
        controls.exposure_ev = 3.0;
        controls.difference_scale = 16.0;
        controls.aerosol_turbidity = 8.0;

        controls.reset_display();
        assert_eq!(controls.view.yaw_deg, 90.0);
        assert_eq!(controls.exposure_ev, defaults.exposure_ev);
        assert_eq!(controls.difference_scale, defaults.difference_scale);
        assert_eq!(controls.aerosol_turbidity, 8.0);

        controls.reset_all(&asset);
        assert_eq!(controls, defaults);
    }
}
