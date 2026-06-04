use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use sky_core::asset::{SPECTRAL_PANORAMA_KIND, SpectralAssetManifest};

#[derive(Clone, Debug)]
pub struct RealtimeAsset {
    manifest_path: PathBuf,
    root_dir: PathBuf,
    manifest: SpectralAssetManifest,
}

impl RealtimeAsset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AssetLoadError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| AssetLoadError::Io {
            path: path.to_owned(),
            source,
        })?;
        let manifest: SpectralAssetManifest =
            serde_json::from_reader(file).map_err(|source| AssetLoadError::Json {
                path: path.to_owned(),
                source,
            })?;
        validate_manifest(&manifest)?;

        let root_dir = path.parent().unwrap_or_else(|| Path::new(".")).to_owned();
        Ok(Self {
            manifest_path: path.to_owned(),
            root_dir,
            manifest,
        })
    }

    pub fn manifest(&self) -> &SpectralAssetManifest {
        &self.manifest
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn title(&self) -> String {
        format!(
            "sky realtime demo - {}x{} {} bands - {} spp",
            self.manifest.dimensions[0],
            self.manifest.dimensions[1],
            self.manifest.band_centers_nm.len(),
            self.manifest.spp
        )
    }

    pub fn summary_line(&self) -> String {
        format!(
            "{} [{}x{}, {} bands, {} spp, sun elev {:.2} deg, observer {:.3} km]",
            self.manifest_path.display(),
            self.manifest.dimensions[0],
            self.manifest.dimensions[1],
            self.manifest.band_centers_nm.len(),
            self.manifest.spp,
            self.manifest.sun_elevation_deg,
            self.manifest.observer_altitude_km
        )
    }

    pub fn missing_referenced_files(&self) -> Vec<PathBuf> {
        self.referenced_files()
            .into_iter()
            .filter(|path| !path.exists())
            .collect()
    }

    pub fn rgb_exr_path(&self) -> PathBuf {
        self.root_dir.join(&self.manifest.files.rgb_exr)
    }

    fn referenced_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::with_capacity(2 + self.manifest.files.band_exrs.len());
        files.push(self.root_dir.join(&self.manifest.files.rgb_exr));
        files.push(self.root_dir.join(&self.manifest.files.rgb_png));
        files.extend(
            self.manifest
                .files
                .band_exrs
                .iter()
                .map(|file| self.root_dir.join(file)),
        );
        files
    }
}

#[derive(Debug)]
pub enum AssetLoadError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidManifest(String),
}

impl fmt::Display for AssetLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read asset manifest {}: {source}",
                    path.display()
                )
            }
            Self::Json { path, source } => {
                write!(
                    f,
                    "failed to parse asset manifest {}: {source}",
                    path.display()
                )
            }
            Self::InvalidManifest(message) => write!(f, "invalid asset manifest: {message}"),
        }
    }
}

impl Error for AssetLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::InvalidManifest(_) => None,
        }
    }
}

fn validate_manifest(manifest: &SpectralAssetManifest) -> Result<(), AssetLoadError> {
    if manifest.version != 0 {
        return Err(AssetLoadError::InvalidManifest(format!(
            "unsupported version {}; expected 0",
            manifest.version
        )));
    }
    if manifest.kind != SPECTRAL_PANORAMA_KIND {
        return Err(AssetLoadError::InvalidManifest(format!(
            "unsupported kind {}; expected {SPECTRAL_PANORAMA_KIND}",
            manifest.kind
        )));
    }
    if manifest.dimensions[0] == 0 || manifest.dimensions[1] == 0 {
        return Err(AssetLoadError::InvalidManifest(
            "panorama dimensions must be non-zero".to_owned(),
        ));
    }
    if manifest.spp == 0 {
        return Err(AssetLoadError::InvalidManifest(
            "sample count must be non-zero".to_owned(),
        ));
    }
    if manifest.band_centers_nm.is_empty() {
        return Err(AssetLoadError::InvalidManifest(
            "at least one spectral band is required".to_owned(),
        ));
    }
    if manifest.band_centers_nm.len() != manifest.files.band_exrs.len() {
        return Err(AssetLoadError::InvalidManifest(format!(
            "band count mismatch: {} wavelengths but {} band files",
            manifest.band_centers_nm.len(),
            manifest.files.band_exrs.len()
        )));
    }
    if !manifest.sun_elevation_deg.is_finite()
        || !manifest.sun_azimuth_deg.is_finite()
        || !manifest.observer_altitude_km.is_finite()
    {
        return Err(AssetLoadError::InvalidManifest(
            "sun angles and observer altitude must be finite".to_owned(),
        ));
    }
    if let Some(center_nm) = manifest
        .band_centers_nm
        .iter()
        .find(|center_nm| !center_nm.is_finite() || **center_nm <= 0.0)
    {
        return Err(AssetLoadError::InvalidManifest(format!(
            "invalid wavelength center {center_nm} nm"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use sky_core::asset::{SpectralAssetFiles, SpectralAssetManifest};

    use super::{AssetLoadError, validate_manifest};

    fn sample_manifest() -> SpectralAssetManifest {
        SpectralAssetManifest::spectral_panorama(
            [4, 2],
            8,
            1,
            0.0,
            0.0,
            0.2,
            vec![500.0],
            SpectralAssetFiles {
                rgb_exr: "sky_rgb.exr".to_owned(),
                rgb_png: "sky_rgb.png".to_owned(),
                band_exrs: vec!["bands/sky_500nm.exr".to_owned()],
            },
        )
    }

    #[test]
    fn accepts_valid_spectral_panorama_manifest() {
        validate_manifest(&sample_manifest()).expect("valid manifest");
    }

    #[test]
    fn rejects_band_file_mismatch() {
        let mut manifest = sample_manifest();
        manifest.files.band_exrs.clear();
        let error = validate_manifest(&manifest).expect_err("invalid manifest");
        assert!(matches!(error, AssetLoadError::InvalidManifest(_)));
    }

    #[test]
    fn parses_manifest_json() {
        let manifest = sample_manifest();
        let json = serde_json::to_string(&manifest).expect("json");
        let parsed: SpectralAssetManifest = serde_json::from_str(&json).expect("parse");
        validate_manifest(&parsed).expect("valid manifest");
    }
}
