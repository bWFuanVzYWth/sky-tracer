use sky_reference::{Result, asset::Manifest, config::BakeConfig, model::Model, solver::BakedBand};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn interrupted_write_files_cannot_push_the_teacher_over_its_budget() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "sky-teacher-budget-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    let result = (|| -> Result<()> {
        let mut model = Model::earth()?;
        model.bands.truncate(1);
        let mut config = BakeConfig::smoke();
        config.max_asset_bytes = config.asset_bytes(1)?;
        let mut manifest = Manifest::new(&model, config.clone(), "CPU budget test".into())?;
        manifest.save(&root)?;
        let orphan = root.join("interrupted.tmp");
        fs::File::create(&orphan)?.set_len(config.band_bytes()? * 2)?;
        let band = || BakedBand {
            radiance: vec![1.0; config.scattering_len()],
            optical_depth: vec![0.0; config.optical_depth_len()],
            ground_irradiance: vec![1.0; config.ground_sun_samples],
            orders: Vec::new(),
            stopped_by_tolerance: false,
        };
        assert!(manifest.write_band(&root, 0, band()).is_err());
        assert!(manifest.records[0].is_none());
        assert!(!root.join("band_000.safetensors").exists());
        fs::remove_file(&orphan)?;
        manifest.write_band(&root, 0, band())?;
        assert!(manifest.complete());
        assert_eq!(
            manifest.read_band(&root, 0)?.radiance,
            vec![1.0; config.scattering_len()]
        );
        let actual = fs::read_dir(&root)?.try_fold(0u64, |bytes, e| -> Result<u64> {
            Ok(bytes + e?.metadata()?.len())
        })?;
        assert!(actual <= config.max_asset_bytes);
        assert!(manifest.write_band(&root, 0, band()).is_err());
        // Even a later manifest-only save must account for retained temporary files.
        fs::File::create(&orphan)?.set_len(config.max_asset_bytes)?;
        assert!(manifest.save(&root).is_err());
        Ok(())
    })();
    assert!(root.starts_with(std::env::temp_dir()));
    fs::remove_dir_all(root)?;
    result
}
