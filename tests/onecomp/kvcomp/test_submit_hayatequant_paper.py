"""Source-only tests for the HayateQuant paper Slurm/SSH orchestrator."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmark/kvcomp-codec/submit_hayatequant_paper.sh"
WRAPPER = REPO / "benchmark/kvcomp-codec/run_paper_job.sh"
LIFECYCLE = REPO / "benchmark/kvcomp-codec/paper_job_lifecycle.py"
RECONCILER = REPO / "benchmark/kvcomp-codec/reconcile_hayatequant_paper_jobs.py"
MODELS = Path("/lustre/home/frj/yuma-ichikawa/conference/models")
PY = Path("/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python")
PYVLLM = Path("/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv-vllm/bin/python")
RESULTS_RELATIVE_ROOT = Path("papers/hayatequant_paper/experiments/results")


@pytest.fixture(scope="module")
def results_worktree(tmp_path_factory):
    parent = tmp_path_factory.mktemp("paper-results-worktree")
    checkout = parent / "checkout"
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach", str(checkout), "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield checkout
    finally:
        subprocess.run(
            ["git", "-C", str(REPO), "worktree", "remove", "--force", str(checkout)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def _worker_repository_args(
    run_root: Path, *, attempt: str, phase: str = "smoke"
) -> list[str]:
    return [
        "--work-root",
        str(run_root.parents[4].parent / "mutable-work"),
        "--attempt-id",
        attempt,
        "--stdout-log",
        str(run_root / f"logs/{phase}/worker.out"),
        "--stderr-log",
        str(run_root / f"logs/{phase}/worker.err"),
    ]


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _plan(command: str, tmp_path: Path, **extra: str) -> str:
    env = dict(os.environ)
    env.update(
        {
            "DRYRUN": "1",
            "SKIP_INPUT_PREFLIGHT": "1",
            "ALLOW_DIRTY_DRYRUN": "1",
            "SPEED_REPLICATES": "1",
            "RUN_ROOT": str(tmp_path / "run"),
            "MODELS": str(MODELS),
            "PY": str(PY),
            "PYVLLM": str(PYVLLM),
            "PREFERRED_NODE": "",
        }
    )
    env.update(extra)
    completed = subprocess.run(
        ["bash", str(SCRIPT), command],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.fixture(scope="module")
def formal_plan(tmp_path_factory) -> str:
    return _plan("formal-plan", tmp_path_factory.mktemp("formal-plan"))


def test_script_is_valid_bash_and_help_documents_fixed_contract(tmp_path):
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    help_text = _plan("help", tmp_path)
    assert "smoke" in help_text and "formal" in help_text
    assert "approve-smoke" in help_text and "formal-plan" in help_text
    assert "gpu11-smoke" in help_text
    assert "data/hf_home" in help_text
    assert "pinned_hf_datasets_manifest.json" in help_text
    assert "there is intentionally no" not in help_text  # comments are not CLI output


def test_read_only_plans_and_preflight_never_write_outside_results_repo(tmp_path):
    run_root = tmp_path / "not-a-results-worktree/run"
    env = {
        **os.environ,
        "DRYRUN": "1",
        "SKIP_INPUT_PREFLIGHT": "0",
        "ALLOW_DIRTY_DRYRUN": "1",
        "RUN_ROOT": str(run_root),
        "MODELS": str(MODELS),
        "PY": str(PY),
        "PYVLLM": str(PYVLLM),
    }
    plan = subprocess.run(
        ["bash", str(SCRIPT), "formal-plan"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert plan.returncode != 0
    assert "requires SKIP_INPUT_PREFLIGHT=1" in plan.stderr
    assert not run_root.exists()

    preflight = subprocess.run(
        ["bash", str(SCRIPT), "preflight"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert preflight.returncode != 0
    assert "unavailable in DRYRUN mode" in preflight.stderr
    assert not run_root.exists()


def test_formal_commands_fail_closed_without_validated_smoke_approval(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "DRYRUN": "1",
            "SKIP_INPUT_PREFLIGHT": "1",
            "ALLOW_DIRTY_DRYRUN": "1",
            "SPEED_REPLICATES": "1",
            "RUN_ROOT": str(tmp_path / "run"),
            "MODELS": str(MODELS),
            "PY": str(PY),
            "PYVLLM": str(PYVLLM),
            "PREFERRED_NODE": "",
        }
    )
    for command in ("formal", "ppl", "math", "speed"):
        completed = subprocess.run(
            ["bash", str(SCRIPT), command],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode != 0
        assert "approve-smoke" in completed.stderr
        assert "JOB phase=formal" not in completed.stdout


def test_formal_plan_has_complete_unique_matrix_and_every_job_uses_wrapper(formal_plan):
    jobs = re.findall(r"^JOB phase=(\S+) key=(\S+) result=(\S+) model=(\S+)$", formal_plan, re.M)
    workers = re.findall(r"^WORKER (.+)$", formal_plan, re.M)
    # Four public HQ rows add the frontier and rotation cells to the previous
    # two-row matrix: 7 PPL + 2 mechanism + 420 zero-shot + 60 RULER +
    # 208 task-level LongBench + 160 MATH + 448 LongCodeQA + 12 fair-speed.
    # FibQuant and BlockQuant share each PPL/zero-shot model load but have
    # separate long-axis quality jobs; both are absent from fair-speed.
    assert len(jobs) == 1137
    assert len(workers) == len(jobs)
    assert all(phase == "formal" for phase, *_ in jobs)
    keys = [key for _, key, _, _ in jobs]
    results = [result for _, _, result, _ in jobs]
    assert len(keys) == len(set(keys))
    assert len(results) == len(set(results))
    assert all("run_paper_job.sh" in line for line in workers)
    assert all("--job-key formal-" in line for line in workers)
    assert all("--input-manifest" in line for line in workers)
    assert all("--input-manifest-sha256" in line for line in workers)
    assert all("--expected-result" in line for line in workers)
    assert formal_plan.count("--output") >= len(jobs)
    assert "%x_%j.out" in formal_plan and "%x_%j.err" in formal_plan
    assert formal_plan.count("--gres gpu:nvidia_b200:1") == 1125
    assert formal_plan.count("--gres gpu:nvidia_b200:8") == 12


def test_fixed_five_replicate_formal_plan_has_exactly_1185_unique_jobs(tmp_path):
    plan = _plan("formal-plan", tmp_path, SPEED_REPLICATES="5")
    jobs = re.findall(r"^JOB phase=formal key=(\S+) result=(\S+)", plan, re.M)
    assert len(jobs) == 1185
    assert len({key for key, _ in jobs}) == 1185
    assert len({result for _, result in jobs}) == 1185
    speed = [key for key, _ in jobs if key.startswith("speed-")]
    assert len(speed) == 60
    assert plan.count("--nodelist DRYRUN-SAME-B200-NODE") == 60
    assert plan.count("--exclusive=user") == 60
    assert plan.count("--partition batch-8gpu") == 60
    assert plan.count("--gres gpu:nvidia_b200:8") == 60
    assert plan.count("--cpus-per-task 160") == 60
    assert plan.count("--mem 2400G") == 60
    assert plan.count("--dependency afterok:") == 59
    assert plan.count("--kill-on-invalid-dep=yes") == 60
    assert plan.count("--exclude DRYRUN-SAME-B200-NODE") == 1125


def test_formal_plan_uses_only_canonical_comparisons_and_full_protocols(formal_plan):
    for misleading in ("HayateQuant-auto",):
        assert misleading not in formal_plan
    assert "FibQuant-paper" in formal_plan
    assert "BlockQuant-paper" in formal_plan
    assert "fibquant_variants=\\[\\{label:FibQuant-paper\\,bits:3\\}\\]" in formal_plan
    assert "blockquant_variants=\\[\\{label:BlockQuant-paper\\,bits:3\\}\\]" in formal_plan
    assert (
        "methods=\\[TurboQuant-paper\\,FibQuant-paper\\,BlockQuant-paper\\,"
        "HQ\\,HQ-Frontier\\,AdaptiveHQ\\,RotationHQ\\]" in formal_plan
    )
    assert "ppl_windows=null" in formal_plan
    assert "official_ruler_limit=null" in formal_plan
    assert "longbench_limit=null" in formal_plan
    assert "math_limit=50" in formal_plan
    assert "math_grader=math_verify" in formal_plan
    assert "math_verify_version=0.9.0" in formal_plan
    assert "longcodeqa_limit=null" not in formal_plan  # formal LongCodeQA is exact row-sharded
    assert "pinned_hf_datasets_manifest.json" in formal_plan


def test_formal_plan_quantizes_every_layer_and_token_under_one_recipe(formal_plan):
    assert "skip_outliers=false" in formal_plan
    assert "--no-skip-outliers" in formal_plan
    assert "sink_tokens=0" in formal_plan
    assert "recent_tokens=0" in formal_plan
    assert "--sink-tokens 0" in formal_plan
    assert "--recent-tokens 0" in formal_plan
    assert "sink_tokens=4" not in formal_plan
    assert "--sink-tokens 4" not in formal_plan
    blocks = re.findall(
        r"^JOB phase=formal key=(\S+).*$\n^SBATCH.*$\n^WORKER (.+)$",
        formal_plan,
        re.M,
    )
    assert len(blocks) == 1137
    for key, worker in blocks:
        if key.startswith("zs-"):
            assert "--no-skip-outliers" in worker
            assert "--sink-tokens 0" in worker and "--recent-tokens 0" in worker
        elif key.startswith("speed-"):
            assert "--sink-tokens 0" in worker and "--recent-tokens 0" in worker
            assert "FibQuant" not in worker
            assert "BlockQuant" not in worker
        else:
            assert "skip_outliers=false" in worker
            assert "sink_tokens=0" in worker and "recent_tokens=0" in worker


def test_zeroshot_intervals_cover_hellaswag_without_overlap(formal_plan):
    keys = re.findall(r"^JOB phase=formal key=(zs-\S+)", formal_plan, re.M)
    llama_full = sorted(
        int(match.group(1))
        for key in keys
        if (match := re.fullmatch(r"zs-llama31-8b-full-hellaswag-s(\d+)", key))
    )
    assert llama_full == [0, 2500, 5000, 7500, 10000]
    for start, limit in ((0, 2500), (2500, 2500), (5000, 2500), (7500, 2500), (10000, 42)):
        assert f"--start {start} --limit {limit}" in formal_plan
    # FullKV plus three rate shards at every deterministic interval.
    assert len(keys) == 252


def test_longcodeqa_is_context_validated_and_row_sharded(formal_plan):
    keys = re.findall(r"^JOB phase=formal key=(longcode-\S+)", formal_plan, re.M)
    assert len(keys) == 448
    assert "longcodeqa_context_llama31-8b.json" in formal_plan
    assert "longcodeqa_context_qwen3-8b.json" in formal_plan
    assert "+longcodeqa_brackets=\\[128K\\]" in formal_plan
    assert "+longcodeqa_start=88" in formal_plan
    assert "+longcodeqa_limit=4" in formal_plan
    assert "+longcodeqa_max_context=131072" in formal_plan
    assert "+longcodeqa_max_context=40960" in formal_plan


def test_longbench_is_exactly_one_complete_official_task_per_job(formal_plan):
    keys = re.findall(r"^JOB phase=formal key=(longbench-\S+)", formal_plan, re.M)
    assert len(keys) == 208
    assert any("-qasper-n224-" in key for key in keys)
    assert any("-multifieldqa_en-n150-" in key for key in keys)
    assert any("-multi_news-n294-" in key for key in keys)
    assert any("-repobench-p-n300-" in key for key in keys)
    assert "longbench_tasks=\\[qasper\\]" in formal_plan
    assert "longbench_tasks=\\[repobench-p\\]" in formal_plan
    assert "longbench_max_length=65536" in formal_plan
    for budget in (40432, 40816, 40880, 40912):
        assert f"longbench_max_length={budget}" in formal_plan


def test_math500_exact_row_intervals_cover_dataset(formal_plan):
    keys = re.findall(r"^JOB phase=formal key=(math500-\S+)", formal_plan, re.M)
    assert len(keys) == 160
    starts = sorted(
        {int(match.group(1)) for key in keys if (match := re.search(r"-s(\d+)$", key))}
    )
    assert starts == list(range(0, 500, 50))
    for start in starts:
        assert f"math_start={start}" in formal_plan
    assert "max_new_math=1024" in formal_plan


def test_default_formal_plan_pins_only_speed_to_one_b200_stratum_and_exports_no_credentials(
    formal_plan,
):
    ordered_keys = re.findall(r"^JOB phase=formal key=(\S+)", formal_plan, re.M)
    assert all(key.startswith("speed-") for key in ordered_keys[:12])
    assert all(not key.startswith("speed-") for key in ordered_keys[12:])
    assert formal_plan.count("--nodelist DRYRUN-SAME-B200-NODE") == 12
    assert formal_plan.count("--exclusive=user") == 12
    assert formal_plan.count("--dependency afterok:") == 11
    assert formal_plan.count("--kill-on-invalid-dep=yes") == 12
    assert formal_plan.count("--exclude DRYRUN-SAME-B200-NODE") == 1125
    assert (
        formal_plan.count("policy=full_8gpu_allocation_afterok_serial quality_jobs=excluded") == 1
    )
    for line in formal_plan.splitlines():
        if "--nodelist" in line:
            assert (
                "key=speed-"
                in formal_plan[max(0, formal_plan.find(line) - 300) : formal_plan.find(line)]
            )
    assert "kagura-gpu11" not in "\n".join(
        line for line in formal_plan.splitlines() if line.startswith("SBATCH")
    )
    for secret_name in ("GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert secret_name not in formal_plan
    assert "example-secret-placeholder-alpha" not in formal_plan
    assert "example-secret-placeholder-beta" not in formal_plan


def test_smoke_and_gpu11_ssh_fallback_are_separate_and_wrapped(tmp_path):
    smoke = _plan("smoke", tmp_path / "smoke")
    jobs = re.findall(r"^JOB phase=smoke key=(\S+) result=(\S+)", smoke, re.M)
    assert len(jobs) == 8
    assert all("/results/smoke/" in result for _, result in jobs)
    assert "/results/formal/" not in smoke
    smoke_workers = dict(
        re.findall(
            r"^JOB phase=smoke key=(\S+).*$\n^SBATCH.*$\n^WORKER (.+)$",
            smoke,
            re.M,
        )
    )
    for key in ("ppl", "ruler", "longbench", "math500", "longcode"):
        assert "skip_outliers=false" in smoke_workers[key]
        assert "sink_tokens=0" in smoke_workers[key]
        assert "recent_tokens=0" in smoke_workers[key]
    assert "--no-skip-outliers" in smoke_workers["zeroshot"]
    assert "--include-fullkv" in smoke_workers["zeroshot"]
    for key in ("zeroshot", "speed"):
        assert "--sink-tokens 0" in smoke_workers[key]
        assert "--recent-tokens 0" in smoke_workers[key]
    ssh_plan = _plan("gpu11-smoke", tmp_path / "ssh", GPU11_RUN_TAG="pytest", GPU11_DEVICE="3")
    assert ssh_plan.count("^SSH_JOB") == 0  # literal caret is not output
    assert len(re.findall(r"^SSH_JOB ", ssh_plan, re.M)) == 2
    assert "kagura-gpu11" in ssh_plan
    assert "CUDA_VISIBLE_DEVICES=3" in ssh_plan
    assert "env\\ -i" in ssh_plan
    assert "/runtime/home" in ssh_plan
    assert ssh_plan.count("run_paper_job.sh") >= 2
    assert "sbatch" not in ssh_plan


def test_multi_gpu_override_and_unsafe_export_encoding_are_rejected(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "DRYRUN": "1",
            "SKIP_INPUT_PREFLIGHT": "1",
            "ALLOW_DIRTY_DRYRUN": "1",
            "SPEED_REPLICATES": "1",
            "RUN_ROOT": str(tmp_path / "run"),
            "MODELS": str(MODELS),
            "PY": str(PY),
            "PYVLLM": str(PYVLLM),
            "GPU_GRES": "gpu:nvidia_b200:2",
        }
    )
    rejected = subprocess.run(
        ["bash", str(SCRIPT), "formal-plan"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode != 0
    assert "quality and smoke allocations" in rejected.stderr

    env["GPU_GRES"] = "gpu:nvidia_b200:1"
    dummy_hf_assignment = "HF_" + "TOKEN=must-not-propagate"
    env["HF_HOME"] = f"{tmp_path}/cache,{dummy_hf_assignment}"
    rejected = subprocess.run(
        ["bash", str(SCRIPT), "formal-plan"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode != 0
    assert "cannot be encoded safely" in rejected.stderr
    assert dummy_hf_assignment not in rejected.stdout

    env["HF_HOME"] = str(tmp_path / "cache")
    env["SMOKE_APPROVAL_MARKER"] = str(tmp_path / "outside-marker.json")
    rejected = subprocess.run(
        ["bash", str(SCRIPT), "formal-plan"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode != 0
    assert "must stay below RUN_ROOT/manifests" in rejected.stderr


def test_individual_quality_commands_require_validated_speed_node_exclusion():
    source = SCRIPT.read_text(encoding="utf-8")
    assert re.search(
        r"prepare_formal_quality_submission\(\).*?validate_speed_node_identity",
        source,
        re.S,
    )
    for command in (
        "ppl",
        "mechanism",
        "zeroshot",
        "ruler",
        "longbench",
        "math",
        "longcodeqa",
    ):
        block = re.search(rf"\n  {command}\)\n(.*?)\n    ;;", source, re.S)
        assert block is not None
        assert "prepare_formal_quality_submission" in block.group(1)
    assert "formal quality job has no validated SPEED_NODE exclusion" in source


def _write_fake_scontrol(path: Path, node: dict) -> Path:
    executable = path / "scontrol"
    executable.parent.mkdir(parents=True, exist_ok=True)
    payload = {"nodes": [node], "errors": []}
    executable.write_text(
        "#!/usr/bin/env python3\n" "import json\n" f"print(json.dumps({payload!r}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _speed_node_preflight(tmp_path: Path, node: dict, *, require_idle: bool):
    fake_bin = tmp_path / "bin"
    _write_fake_scontrol(fake_bin, node)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MODELS": str(MODELS),
            "PY": str(PY),
            "PYVLLM": str(PYVLLM),
            "SPEED_NODE": "kagura-gpu07",
            "REQUIRE_SPEED_NODE_IDLE": "1" if require_idle else "0",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), "speed-node-preflight"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _speed_node_fixture(*, state: str, gres_used: str, gres: str = "gpu:nvidia_b200:8"):
    allocated = state != "IDLE"
    return {
        "name": "kagura-gpu07",
        "partitions": ["batch-1gpu", "batch-8gpu"],
        "gres": gres,
        "cpus": 192,
        "state": [state],
        "alloc_cpus": 16 if allocated else 0,
        "alloc_memory": 98304 if allocated else 0,
        "gres_used": gres_used,
        "tres_used": "cpu=16,mem=96G,gres/gpu=1" if allocated else "",
    }


def test_quality_exclusion_accepts_mixed_identity_but_timing_requires_live_idle_schema(
    tmp_path,
):
    mixed = _speed_node_fixture(state="MIXED", gres_used="gpu:nvidia_b200:1(IDX:0)")
    identity = _speed_node_preflight(tmp_path / "mixed-id", mixed, require_idle=False)
    assert identity.returncode == 0, identity.stderr
    assert "state=MIXED" in identity.stdout
    timing = _speed_node_preflight(tmp_path / "mixed-timing", mixed, require_idle=True)
    assert timing.returncode != 0
    assert "not completely idle" in timing.stderr

    idle = _speed_node_fixture(state="IDLE", gres_used="gpu:nvidia_b200:0(IDX:N/A)")
    ready = _speed_node_preflight(tmp_path / "idle", idle, require_idle=True)
    assert ready.returncode == 0, ready.stderr
    assert "SPEED_STRATUM node=kagura-gpu07" in ready.stdout


def test_speed_node_identity_rejects_wrong_b200_gres(tmp_path):
    wrong = _speed_node_fixture(
        state="MIXED",
        gres_used="gpu:nvidia_h100:1(IDX:0)",
        gres="gpu:nvidia_h100:8",
    )
    completed = _speed_node_preflight(tmp_path, wrong, require_idle=False)
    assert completed.returncode != 0
    assert "identity mismatch" in completed.stderr


def test_worker_requires_full_pinned_identity_and_scrubs_named_credentials(tmp_path):
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="utf-8")
    assert "umask 077" in source
    for allocation_field in (
        "SLURM_JOB_PARTITION",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_GPUS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES",
        "PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE",
    ):
        assert allocation_field in source
    assert 'export CUDA_VISIBLE_DEVICES="${SPEED_VISIBLE_DEVICES[0]}"' in source
    for secret_name in (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITLAB_TOKEN",
        "GITLAB_ACCESS_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SSH_AUTH_SOCK",
    ):
        assert secret_name in source
    assert "--expected-result" in source
    completed = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--repo",
            str(REPO),
            "--python",
            str(PY),
            "--run-root",
            str(tmp_path),
            "--job-key",
            "missing-model",
            "--",
            "/bin/true",
        ],
        cwd=REPO,
        env={**os.environ, "CODE_REPO_COMMIT": "a" * 40},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "Usage:" in completed.stderr


def test_submitter_persists_sbatch_stdout_stderr_and_terminal_failure_receipt():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"${sbatch_cmd[@]}" >"$scheduler_stdout" 2>"$scheduler_stderr"' in source
    assert "persist_active_submission submission_failed sbatch_failed" in source
    assert "ACTIVE_SUBMISSION_RECORD=$submission_record" in source
    assert "persist_active_submission submission_interrupted" in source
    for signal_name in ("HUP", "INT", "TERM"):
        assert f"trap 'submitter_on_signal {signal_name.lower()}" in source
    assert "trap submitter_on_exit EXIT" in source
    assert '--comment "$submission_tag"' in source


def test_lifecycle_helper_retains_a_scheduler_submission_failure(results_worktree):
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / "pytest-sbatch-failure"
    expected = run_root / "results/smoke/never-started.json"
    attempt = "sbatch-failure-attempt"
    record = run_root / f"job_records/smoke/smoke-sbatch/{attempt}/submission.json"
    stdout = record.parent / "sbatch.stdout"
    stderr = record.parent / "sbatch.stderr"
    record.parent.mkdir(parents=True)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("scheduler rejected request\n", encoding="utf-8")
    common = [
        str(PY),
        str(LIFECYCLE),
        "record",
        "--source-repo",
        str(REPO),
        "--run-root",
        str(run_root),
        "--source-commit",
        _head(),
        "--path",
        str(record),
        "--kind",
        "submission",
        "--phase",
        "smoke",
        "--job-key",
        "smoke-sbatch",
        "--attempt-id",
        attempt,
        "--expected-result",
        str(expected),
        "--stage",
        "submitter",
        "--scheduler-stdout",
        str(stdout),
        "--scheduler-stderr",
        str(stderr),
    ]
    subprocess.run(
        [*common, "--status", "submission_started", "--reason-code", "submission_started"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        [
            *common,
            "--status",
            "submission_failed",
            "--reason-code",
            "sbatch_failed",
            "--exit-code",
            "17",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    receipt = json.loads(record.read_text(encoding="utf-8"))
    assert receipt["status"] == "submission_failed"
    assert receipt["reason_code"] == "sbatch_failed"
    assert receipt["exit_code"] == 17
    assert receipt["terminal"] is True


def test_lifecycle_helper_rejects_source_worktree_as_result_root():
    run_root = REPO / RESULTS_RELATIVE_ROOT / "pytest-forbidden-source-worktree"
    completed = subprocess.run(
        [
            str(PY),
            str(LIFECYCLE),
            "validate-root",
            "--source-repo",
            str(REPO),
            "--run-root",
            str(run_root),
            "--expected-commit",
            _head(),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "separate linked Git worktree" in completed.stderr


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        ("duplicate", '{"value": 1, "value": 2}\n'),
        ("nonfinite", '{"value": NaN}\n'),
        ("array", "[]\n"),
        ("empty", ""),
    ],
)
def test_lifecycle_helper_never_marks_invalid_result_successful(
    results_worktree, suffix, payload
):
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / f"pytest-invalid-{suffix}"
    expected = run_root / "results/smoke/result.json"
    expected.parent.mkdir(parents=True)
    expected.write_text(payload, encoding="utf-8")
    attempt = "invalid-result-attempt"
    record = run_root / f"job_records/smoke/smoke-invalid/{attempt}/worker.json"
    common = [
        str(PY),
        str(LIFECYCLE),
        "record",
        "--source-repo",
        str(REPO),
        "--run-root",
        str(run_root),
        "--source-commit",
        _head(),
        "--path",
        str(record),
        "--kind",
        "worker",
        "--phase",
        "smoke",
        "--job-key",
        "smoke-invalid",
        "--attempt-id",
        attempt,
        "--expected-result",
        str(expected),
        "--stage",
        "result-validation",
    ]
    subprocess.run(
        [*common, "--status", "worker_started", "--reason-code", "worker_invoked"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    completed = subprocess.run(
        [
            *common,
            "--status",
            "succeeded",
            "--reason-code",
            "benchmark_and_result_validated",
            "--exit-code",
            "0",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    receipt = json.loads(record.read_text(encoding="utf-8"))
    assert receipt["status"] == "worker_started"
    assert receipt["terminal"] is False


def test_worker_preflight_failure_is_terminal_and_repo_local(tmp_path, results_worktree):
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / "pytest-preflight-failure"
    manifest = run_root / "manifests/inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    result = run_root / "results/smoke/result.json"
    completed = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--repo",
            str(REPO),
            "--python",
            str(PY),
            "--run-root",
            str(run_root),
            *_worker_repository_args(run_root, attempt="preflight-attempt"),
            "--job-key",
            "smoke-preflight",
            "--model",
            str(MODELS / "Qwen3-0.6B"),
            "--input-manifest",
            str(manifest),
            "--input-manifest-sha256",
            "0" * 64,
            "--expected-result",
            str(result),
            "--dataset-file",
            str(manifest),
            "--",
            "/bin/true",
        ],
        cwd=REPO,
        env={**os.environ, "CODE_REPO_COMMIT": _head()},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    receipt_path = (
        run_root
        / "job_records/smoke/smoke-preflight/preflight-attempt/worker.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "preflight"
    assert receipt["reason_code"] == "preflight_failed"
    assert receipt["exit_code"] != 0


def test_worker_rejects_result_that_appeared_while_pending(tmp_path, results_worktree):
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / "pytest-worker-race"
    result = run_root / "results/smoke/already-there.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    manifest = run_root / "manifests/inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--repo",
            str(REPO),
            "--python",
            str(PY),
            "--run-root",
            str(run_root),
            *_worker_repository_args(run_root, attempt="race-attempt"),
            "--job-key",
            "smoke-race",
            "--model",
            str(MODELS / "Qwen3-0.6B"),
            "--input-manifest",
            str(manifest),
            "--input-manifest-sha256",
            "0" * 64,
            "--expected-result",
            str(result),
            "--dataset-file",
            str(manifest),
            "--",
            "/bin/true",
        ],
        cwd=REPO,
        env={**os.environ, "CODE_REPO_COMMIT": _head()},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "refusing overwrite/resume race" in completed.stderr
    receipt = run_root / "job_records/smoke/smoke-race/race-attempt/worker.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "failed"


def test_worker_result_lock_rejects_concurrent_writer(tmp_path, results_worktree):
    run_root = (results_worktree / RESULTS_RELATIVE_ROOT / "pytest-worker-lock").resolve()
    result = (run_root / "results/smoke/concurrent.json").resolve()
    manifest = run_root / "manifests/inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    lock_id = hashlib.sha256(str(result).encode("utf-8") + b"\0").hexdigest()
    work_root = run_root.parents[4].parent / "mutable-work"
    lock_path = work_root / "locks/results" / f"{lock_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(
            [
                "bash",
                str(WRAPPER),
                "--repo",
                str(REPO),
                "--python",
                str(PY),
                "--run-root",
                str(run_root),
                *_worker_repository_args(run_root, attempt="lock-attempt"),
                "--job-key",
                "smoke-concurrent",
                "--model",
                str(MODELS / "Qwen3-0.6B"),
                "--input-manifest",
                str(manifest),
                "--input-manifest-sha256",
                "0" * 64,
                "--expected-result",
                str(result),
                "--dataset-file",
                str(manifest),
                "--",
                "/bin/true",
            ],
            cwd=REPO,
            env={**os.environ, "CODE_REPO_COMMIT": _head()},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    assert completed.returncode == 2
    assert "already owns the result lock" in completed.stderr
    receipt = run_root / "job_records/smoke/smoke-concurrent/lock-attempt/worker.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["reason_code"] == (
        "argument_validation_failed"
    )


def test_formal_worker_rejects_missing_or_disagreeing_smoke_binding_environment(
    tmp_path, results_worktree
):
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / "pytest-formal-binding"
    manifest = run_root / "manifests/inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    marker = run_root / "manifests/formal_smoke_approval.json"
    marker.write_text("{}\n", encoding="utf-8")
    result = run_root / "results/formal/result.json"
    argv = [
        "bash",
        str(WRAPPER),
        "--repo",
        str(REPO),
        "--python",
        str(PY),
        "--run-root",
        str(run_root),
        *_worker_repository_args(run_root, attempt="missing-binding", phase="formal"),
        "--job-key",
        "formal-binding-test",
        "--model",
        str(MODELS / "Qwen3-0.6B"),
        "--input-manifest",
        str(manifest),
        "--input-manifest-sha256",
        "0" * 64,
        "--smoke-approval-marker",
        str(marker),
        "--smoke-approval-marker-sha256",
        "1" * 64,
        "--smoke-validated-plan-sha256",
        "2" * 64,
        "--expected-result",
        str(result),
        "--dataset-file",
        str(manifest),
        "--",
        "/bin/true",
    ]
    base_env = {**os.environ, "CODE_REPO_COMMIT": _head()}
    for variable in (
        "PAPER_SMOKE_APPROVAL_MARKER",
        "PAPER_SMOKE_APPROVAL_MARKER_SHA256",
        "PAPER_SMOKE_VALIDATED_PLAN_SHA256",
    ):
        base_env.pop(variable, None)
    missing = subprocess.run(
        argv,
        cwd=REPO,
        env=base_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing.returncode == 2
    assert "environment omitted" in missing.stderr
    missing_receipt = (
        run_root
        / "job_records/formal/formal-binding-test/missing-binding/worker.json"
    )
    assert json.loads(missing_receipt.read_text(encoding="utf-8"))["status"] == "failed"

    disagreeing_env = {
        **base_env,
        "PAPER_SMOKE_APPROVAL_MARKER": str(marker),
        "PAPER_SMOKE_APPROVAL_MARKER_SHA256": "f" * 64,
        "PAPER_SMOKE_VALIDATED_PLAN_SHA256": "2" * 64,
    }
    disagreeing_argv = list(argv)
    attempt_index = disagreeing_argv.index("--attempt-id") + 1
    disagreeing_argv[attempt_index] = "disagreeing-binding"
    disagreeing = subprocess.run(
        disagreeing_argv,
        cwd=REPO,
        env=disagreeing_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert disagreeing.returncode == 2
    assert "argv/environment smoke identities disagree" in disagreeing.stderr


def _submission_fixture(
    results_worktree: Path,
    *,
    run_id: str,
    state: str = "submitted",
    job_id: str | None = "71234",
) -> tuple[Path, Path, Path, dict]:
    run_root = results_worktree / RESULTS_RELATIVE_ROOT / run_id
    phase = "formal"
    job_key = "formal-reconcile"
    attempt = "attempt-1"
    expected = run_root / "results/formal/reconcile/result.json"
    attempt_root = run_root / f"job_records/{phase}/{job_key}/{attempt}"
    attempt_root.mkdir(parents=True)
    stdout = attempt_root / "sbatch.stdout"
    stderr = attempt_root / "sbatch.stderr"
    stdout.write_text(f"{job_id}\n" if job_id is not None else "", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    receipt_path = attempt_root / "submission.json"
    common = [
        str(PY),
        str(LIFECYCLE),
        "record",
        "--source-repo",
        str(REPO),
        "--run-root",
        str(run_root),
        "--source-commit",
        _head(),
        "--path",
        str(receipt_path),
        "--kind",
        "submission",
        "--phase",
        phase,
        "--job-key",
        job_key,
        "--attempt-id",
        attempt,
        "--expected-result",
        str(expected),
        "--stage",
        "submitter",
        "--scheduler-stdout",
        str(stdout),
        "--scheduler-stderr",
        str(stderr),
    ]
    subprocess.run(
        [*common, "--status", "submission_started", "--reason-code", "submission_started"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    if state == "submitted":
        assert job_id is not None
        subprocess.run(
            [
                *common,
                "--status",
                "submitted",
                "--reason-code",
                "sbatch_accepted",
                "--exit-code",
                "0",
                "--slurm-job-id",
                job_id,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    elif state == "interrupted":
        subprocess.run(
            [
                *common,
                "--status",
                "submission_interrupted",
                "--reason-code",
                "submitter_signal_term",
                "--exit-code",
                "143",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    elif state != "started":
        raise AssertionError(f"unknown test submission state {state}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return run_root, expected, receipt_path, receipt


def _fake_sacct(
    tmp_path: Path,
    receipt: dict,
    *,
    state: str,
    job_id: str | None = None,
    user: str | None = None,
    job_name: str | None = None,
    comment: str | None = None,
    rows: int = 1,
    returncode: int = 0,
) -> Path:
    executable = tmp_path / "sacct"
    executable.parent.mkdir(parents=True, exist_ok=True)
    submitted = datetime.fromisoformat(receipt["created_at_utc"]).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    values = (
        job_id or receipt.get("slurm_job_id") or "71234",
        user or receipt["scheduler_user"],
        job_name or receipt["scheduler_job_name"],
        state,
        "1:0" if state not in {"PENDING", "RUNNING", "COMPLETED"} else "0:0",
        submitted,
        "Unknown" if state == "PENDING" else submitted,
        "Unknown" if state in {"PENDING", "RUNNING"} else submitted,
        comment or receipt["submission_tag"],
    )
    line = "|".join(values) + "|\n"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({(line * rows)!r})\n"
        f"raise SystemExit({returncode})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _reconcile(run_root: Path, sacct: Path):
    return subprocess.run(
        [
            str(PY),
            str(RECONCILER),
            "--source-repo",
            str(REPO),
            "--run-root",
            str(run_root),
            "--source-commit",
            _head(),
            "--phase",
            "formal",
            "--sacct",
            str(sacct),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_scheduler_reconciler_writes_hashed_failure_only_receipt(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id="pytest-reconcile-failed"
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="NODE_FAIL")
    )
    assert completed.returncode == 0, completed.stderr
    scheduler_path = submission_path.with_name("scheduler.json")
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    assert scheduler["schema_version"] == 2
    assert scheduler["status"] == "scheduler_terminal_failure"
    assert scheduler["terminal"] is True
    assert scheduler["slurm"]["state"] == "NODE_FAIL"
    assert "result" not in scheduler
    assert scheduler["run_id"] == run_root.name
    assert scheduler["results_worktree"] == str(run_root.parents[4])
    for stream in scheduler["scheduler_query_evidence"].values():
        retained = run_root / stream["path"]
        assert stream["bytes"] == retained.stat().st_size
        assert stream["sha256"] == hashlib.sha256(retained.read_bytes()).hexdigest()


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_scheduler_reconciler_leaves_nonterminal_jobs_without_receipt(
    tmp_path, results_worktree, state
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id=f"pytest-reconcile-{state.lower()}"
    )
    completed = _reconcile(run_root, _fake_sacct(tmp_path, submission, state=state))
    assert completed.returncode == 0, completed.stderr
    assert not submission_path.with_name("scheduler.json").exists()
    summary = json.loads(completed.stdout)
    assert summary["counts"]["nonterminal"] == 1


def test_scheduler_reconciler_rejects_completed_without_worker(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id="pytest-reconcile-completed"
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="COMPLETED")
    )
    assert completed.returncode == 2
    assert not submission_path.with_name("scheduler.json").exists()
    assert "COMPLETED cannot replace" in completed.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "99999"),
        ("user", "wrong-user"),
        ("job_name", "wrong-job"),
        ("comment", "wrong-attempt-tag"),
    ],
)
def test_scheduler_reconciler_rejects_wrong_authoritative_identity(
    tmp_path, results_worktree, field, value
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id=f"pytest-reconcile-wrong-{field}"
    )
    sacct = _fake_sacct(
        tmp_path,
        submission,
        state="FAILED",
        **{field: value},
    )
    completed = _reconcile(run_root, sacct)
    assert completed.returncode == 2
    assert not submission_path.with_name("scheduler.json").exists()
    assert json.loads(completed.stdout)["counts"]["errors"] == 1


def test_scheduler_reconciler_rejects_ambiguous_accounting_rows(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id="pytest-reconcile-ambiguous"
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="FAILED", rows=2)
    )
    assert completed.returncode == 2
    assert not submission_path.with_name("scheduler.json").exists()


def test_scheduler_reconciler_rejects_result_without_worker(
    tmp_path, results_worktree
):
    run_root, expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id="pytest-reconcile-fake-result"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text('{"untrusted": true}\n', encoding="utf-8")
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="FAILED")
    )
    assert completed.returncode == 2
    assert not submission_path.with_name("scheduler.json").exists()
    assert "exists without a worker receipt" in completed.stdout


def test_interrupted_submission_without_accepted_id_becomes_terminal_failure(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree,
        run_id="pytest-reconcile-interrupted-empty",
        state="interrupted",
        job_id=None,
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="FAILED")
    )
    assert completed.returncode == 0, completed.stderr
    terminal = json.loads(submission_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "submission_failed"
    assert terminal["reason_code"] == "scheduler_acceptance_unproven"
    assert not submission_path.with_name("scheduler.json").exists()


def test_interrupted_submission_recovers_only_exact_tagged_acceptance(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree,
        run_id="pytest-reconcile-interrupted-recovered",
        state="interrupted",
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="PENDING")
    )
    assert completed.returncode == 0, completed.stderr
    recovered = json.loads(submission_path.read_text(encoding="utf-8"))
    assert recovered["status"] == "submitted"
    assert recovered["reason_code"] == "sacct_recovered_acceptance"
    assert recovered["slurm_job_id"] == "71234"
    assert not submission_path.with_name("scheduler.json").exists()


def test_interrupted_submission_wrong_tag_is_failed_not_recovered(
    tmp_path, results_worktree
):
    run_root, _expected, submission_path, submission = _submission_fixture(
        results_worktree,
        run_id="pytest-reconcile-interrupted-wrong-tag",
        state="interrupted",
    )
    completed = _reconcile(
        run_root,
        _fake_sacct(tmp_path, submission, state="FAILED", comment="wrong-tag"),
    )
    assert completed.returncode == 2
    failed = json.loads(submission_path.read_text(encoding="utf-8"))
    assert failed["status"] == "submission_failed"
    assert failed["reason_code"] == "scheduler_identity_unproven"
    assert not submission_path.with_name("scheduler.json").exists()


def test_scheduler_receipt_blocks_worker_receipt_for_same_attempt(
    tmp_path, results_worktree
):
    run_root, expected, submission_path, submission = _submission_fixture(
        results_worktree, run_id="pytest-reconcile-exclusive"
    )
    completed = _reconcile(
        run_root, _fake_sacct(tmp_path, submission, state="TIMEOUT")
    )
    assert completed.returncode == 0, completed.stderr
    worker_path = submission_path.with_name("worker.json")
    rejected = subprocess.run(
        [
            str(PY),
            str(LIFECYCLE),
            "record",
            "--source-repo",
            str(REPO),
            "--run-root",
            str(run_root),
            "--source-commit",
            _head(),
            "--path",
            str(worker_path),
            "--kind",
            "worker",
            "--status",
            "worker_started",
            "--phase",
            "formal",
            "--job-key",
            "formal-reconcile",
            "--attempt-id",
            "attempt-1",
            "--expected-result",
            str(expected),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode == 2
    assert "mutually exclusive" in rejected.stderr
