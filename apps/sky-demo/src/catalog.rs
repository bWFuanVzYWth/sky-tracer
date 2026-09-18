use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::assets::RealtimeAsset;

#[derive(Clone, Debug)]
pub struct AssetMetadata {
    pub kind: String,
    pub dimensions: [u32; 2],
    pub bands: usize,
    pub spp: u32,
    pub sun_elevation_deg: f32,
    pub reference_available: bool,
}

#[derive(Clone, Debug)]
pub struct AssetEntry {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub metadata: Option<AssetMetadata>,
    pub error: Option<String>,
}

impl AssetEntry {
    pub fn from_path(path: PathBuf, root: &Path) -> Self {
        let relative_path = path.strip_prefix(root).unwrap_or(&path).to_owned();
        match RealtimeAsset::load(&path) {
            Ok(asset) => {
                let manifest = asset.manifest();
                Self {
                    path,
                    relative_path,
                    metadata: Some(AssetMetadata {
                        kind: short_kind(&manifest.kind).to_owned(),
                        dimensions: [manifest.dimensions[0] as u32, manifest.dimensions[1] as u32],
                        bands: manifest.band_centers_nm.len(),
                        spp: manifest.spp as u32,
                        sun_elevation_deg: manifest.sun_elevation_deg,
                        reference_available: asset.rgb_exr_path().is_file(),
                    }),
                    error: None,
                }
            }
            Err(error) => Self {
                path,
                relative_path,
                metadata: None,
                error: Some(error.to_string()),
            },
        }
    }

    pub fn is_valid(&self) -> bool {
        self.metadata.is_some()
    }

    pub fn matches(&self, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        if query.is_empty() {
            return true;
        }
        self.relative_path
            .to_string_lossy()
            .to_lowercase()
            .contains(&query)
            || self
                .metadata
                .as_ref()
                .is_some_and(|metadata| metadata.kind.to_lowercase().contains(&query))
    }
}

#[derive(Clone, Debug)]
pub struct CatalogScan {
    pub root: PathBuf,
    pub entries: Vec<AssetEntry>,
}

pub fn scan_asset_root(root: PathBuf) -> Result<CatalogScan, String> {
    if !root.is_dir() {
        return Err(format!("asset root is not a directory: {}", root.display()));
    }
    let mut paths = WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.eq_ignore_ascii_case("asset.json"))
        })
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    paths.sort_by_key(|path| path.to_string_lossy().to_lowercase());
    let entries = paths
        .into_iter()
        .map(|path| AssetEntry::from_path(path, &root))
        .collect();
    Ok(CatalogScan { root, entries })
}

fn short_kind(kind: &str) -> &str {
    if kind.contains("sky_view") {
        "LUT"
    } else if kind.contains("panorama") {
        "PANORAMA"
    } else {
        kind
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    use sky_assets::asset::{SpectralAssetFiles, SpectralAssetManifest};

    use super::scan_asset_root;

    static NEXT_TEST_ROOT: AtomicU64 = AtomicU64::new(0);

    fn write_manifest(path: &Path, sun_elevation_deg: f32) {
        let manifest = SpectralAssetManifest::spectral_panorama(
            [4, 2],
            8,
            1,
            sun_elevation_deg,
            0.0,
            0.2,
            vec![500.0],
            SpectralAssetFiles {
                rgb_exr: "sky.exr".to_owned(),
                rgb_png: "sky.png".to_owned(),
                band_exrs: vec!["band.exr".to_owned()],
            },
        );
        std::fs::create_dir_all(path.parent().expect("manifest parent")).expect("create parent");
        std::fs::write(path, serde_json::to_vec(&manifest).expect("serialize"))
            .expect("write manifest");
    }

    #[test]
    fn recursive_scan_is_sorted_and_keeps_invalid_manifests() {
        let root = std::env::temp_dir().join(format!(
            "sky-realtime-catalog-{}-{}",
            std::process::id(),
            NEXT_TEST_ROOT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        write_manifest(&root.join("b/asset.json"), 20.0);
        write_manifest(&root.join("A/asset.json"), 10.0);
        std::fs::create_dir_all(root.join("c")).expect("create invalid parent");
        std::fs::write(root.join("c/asset.json"), b"not json").expect("write invalid");

        let scan = scan_asset_root(root.clone()).expect("scan root");
        assert_eq!(scan.entries.len(), 3);
        assert_eq!(scan.entries[0].relative_path, Path::new("A/asset.json"));
        assert!(scan.entries[0].matches("panorama"));
        assert!(!scan.entries[0].matches("urban"));
        assert!(
            !scan.entries[0]
                .metadata
                .as_ref()
                .expect("valid metadata")
                .reference_available
        );
        assert!(scan.entries[2].error.is_some());
        std::fs::remove_dir_all(root).expect("remove test root");
    }
}
