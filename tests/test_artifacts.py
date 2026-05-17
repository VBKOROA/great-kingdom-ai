from great_kingdom_ai.artifacts import (
    ArtifactPaths,
    load_artifact_paths,
)


def test_artifact_paths_load_from_config_and_create_directories(tmp_path) -> None:
    config_path = tmp_path / "artifacts.json"
    config_path.write_text(
        """
{
  "replay_dir": "replay",
  "checkpoint_dir": "checkpoints",
  "log_dir": "logs"
}
""".strip(),
        encoding="utf-8",
    )

    paths = load_artifact_paths(config_path)
    rooted = ArtifactPaths(
        replay_dir=tmp_path / paths.replay_dir,
        checkpoint_dir=tmp_path / paths.checkpoint_dir,
        log_dir=tmp_path / paths.log_dir,
    )
    rooted.ensure_dirs()

    assert rooted.replay_dir.is_dir()
    assert rooted.checkpoint_dir.is_dir()
    assert rooted.log_dir.is_dir()
    assert rooted.to_dict()["replay_dir"].endswith("replay")
