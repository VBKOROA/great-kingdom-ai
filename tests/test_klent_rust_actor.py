from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
_onnx_spec = importlib.util.find_spec("onnx")
_onnxruntime_spec = importlib.util.find_spec("onnxruntime")
_core_spec = importlib.util.find_spec("great_kingdom_core")
pytestmark = pytest.mark.skipif(
    _torch_spec is None
    or _onnx_spec is None
    or _onnxruntime_spec is None
    or _core_spec is None,
    reason="torch, onnx, onnxruntime, and great_kingdom_core are required",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None
core = importlib.import_module("great_kingdom_core") if _core_spec is not None else None

from great_kingdom_ai.features import ACTION_SPACE  # noqa: E402
from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    KlentTrainState,
    save_klent_checkpoint,
)
from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx  # noqa: E402
from great_kingdom_ai.klent.rust_actor import (  # noqa: E402
    RustKlentActorConfig,
    play_rust_klent_zero_search,
    run_rust_klent_zero_search,
)
from great_kingdom_ai.klent.targets import (  # noqa: E402
    compute_analytical_policy,
    masked_state_value,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import create_model  # noqa: E402
from great_kingdom_ai.training.checkpoint import create_optimizer  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402


def _export_actor(tmp_path: Path) -> Path:
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-3, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    checkpoint = save_klent_checkpoint(state, tmp_path / "actor-source.pt")
    actor_path = tmp_path / "actor.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, actor_path, kind="actor")
    return actor_path


def _actor_config(actor_path: Path, tmp_path: Path, games: int = 2) -> RustKlentActorConfig:
    return RustKlentActorConfig(
        actor_onnx_path=actor_path,
        output_dir=tmp_path / "artifacts",
        games=games,
        seed_start=0,
        rust_self_play_batch_size=games,
        max_turns=200,
    )


def test_rust_analytical_policy_matches_python_reference() -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        logits = rng.normal(size=ACTION_SPACE).astype(np.float32)
        q_values = rng.normal(size=ACTION_SPACE).astype(np.float32)
        legal_mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
        legal_indexes = rng.choice(ACTION_SPACE, size=4, replace=False)
        legal_mask[legal_indexes] = True

        rust_target = core.klent_analytical_policy(
            logits.tolist(),
            q_values.tolist(),
            legal_mask.tolist(),
            0.03,
            0.1,
        )
        rust_value = core.klent_masked_state_value(
            logits.tolist(),
            q_values.tolist(),
            legal_mask.tolist(),
        )
        python_target = compute_analytical_policy(
            torch.from_numpy(logits).unsqueeze(0),
            torch.from_numpy(q_values).unsqueeze(0),
            torch.from_numpy(legal_mask).unsqueeze(0),
            alpha=0.03,
            beta=0.1,
        )[0].numpy()
        python_value = float(
            masked_state_value(
                torch.from_numpy(logits).unsqueeze(0),
                torch.from_numpy(q_values).unsqueeze(0),
                torch.from_numpy(legal_mask).unsqueeze(0),
            )[0]
        )

        assert np.allclose(rust_target, python_target, atol=1e-5)
        assert np.allclose(np.asarray(rust_target)[~legal_mask], 0.0)
        assert abs(rust_value - python_value) < 1e-5


def test_rust_analytical_policy_rejects_empty_legal_mask() -> None:
    with pytest.raises(ValueError, match="at least one legal action"):
        core.klent_analytical_policy(
            [0.0] * ACTION_SPACE,
            [0.0] * ACTION_SPACE,
            [False] * ACTION_SPACE,
            0.03,
            0.1,
        )


def test_rust_zero_search_actor_plays_legal_games(tmp_path: Path) -> None:
    actor_path = _export_actor(tmp_path)
    config = _actor_config(actor_path, tmp_path)

    summary = run_rust_klent_zero_search(config)

    assert summary.games == 2
    assert summary.transitions > 0
    assert (tmp_path / "artifacts" / "trajectory-replay.npz").exists()
    assert (tmp_path / "artifacts" / "game_logs.json").exists()
    for episode in summary.trajectory_episodes:
        assert episode.transitions
        for index, transition in enumerate(episode.transitions):
            assert transition.legal_mask is not None
            assert bool(transition.legal_mask[transition.action])
            assert transition.policy_target.sum() == pytest.approx(1.0, abs=1e-5)
            assert np.all(transition.policy_target[~transition.legal_mask] == 0.0)
            assert transition.lambda_return is not None
            assert -1.0 <= transition.lambda_return <= 1.0
            assert transition.model_version == config.model_version
            assert transition.created_iteration == config.created_iteration
            assert transition.terminal is (index == len(episode.transitions) - 1)


def test_rust_zero_search_actor_rejects_two_output_model(tmp_path: Path) -> None:
    checkpoint = save_klent_checkpoint(
        KlentTrainState(
            model=create_model("small_klent"),
            optimizer=create_optimizer(
                torch,
                create_model("small_klent"),
                TrainingConfig(model_preset="small_klent", learning_rate=1e-3, device="cpu"),
            ),
            iteration=0,
            total_steps=0,
            klent_config=KlentConfig(),
            model_preset="small_klent",
        ),
        tmp_path / "source.pt",
    )
    eval_path = tmp_path / "eval.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, eval_path, kind="eval")

    with pytest.raises(ValueError, match="q_values"):
        core.KlentZeroSearchBatch(
            str(eval_path),
            1,
            0,
            device="cpu",
            max_batch_size=4,
            alpha=0.03,
            beta=0.1,
            max_turns=200,
        )


def test_rust_zero_search_actor_respects_episode_id_offset(tmp_path: Path) -> None:
    actor_path = _export_actor(tmp_path)
    config = RustKlentActorConfig(
        actor_onnx_path=actor_path,
        output_dir=tmp_path / "artifacts",
        games=1,
        episode_id_offset=10,
        rust_self_play_batch_size=1,
    )

    summary = play_rust_klent_zero_search(config)

    assert summary.trajectory_episodes[0].episode_id == 10