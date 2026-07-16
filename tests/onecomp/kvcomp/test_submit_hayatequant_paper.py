"""Source-only tests for the HayateQuant paper Slurm/SSH orchestrator."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmark/kvcomp-codec/submit_hayatequant_paper.sh"
WRAPPER = REPO / "benchmark/kvcomp-codec/run_paper_job.sh"
MODELS = Path("/lustre/home/frj/yuma-ichikawa/conference/models")
PY = Path("/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python")
PYVLLM = Path("/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv-vllm/bin/python")


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
    # 7 PPL + 2 mechanism + 252 zero-shot + 36 RULER + 156 task-level
    # LongBench + 120 MATH + 336 LongCodeQA + 12 fair-speed (replicates=1).
    # FibQuant and BlockQuant share each PPL/zero-shot model load but have
    # separate long-axis quality jobs; both are absent from fair-speed.
    assert len(jobs) == 921
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
    assert formal_plan.count("--gres gpu:nvidia_b200:1") == 909
    assert formal_plan.count("--gres gpu:nvidia_b200:8") == 12


def test_fixed_five_replicate_formal_plan_has_exactly_969_unique_jobs(tmp_path):
    plan = _plan("formal-plan", tmp_path, SPEED_REPLICATES="5")
    jobs = re.findall(r"^JOB phase=formal key=(\S+) result=(\S+)", plan, re.M)
    assert len(jobs) == 969
    assert len({key for key, _ in jobs}) == 969
    assert len({result for _, result in jobs}) == 969
    speed = [key for key, _ in jobs if key.startswith("speed-")]
    assert len(speed) == 60
    assert plan.count("--nodelist DRYRUN-SAME-B200-NODE") == 60
    assert plan.count("--exclusive=user") == 60
    assert plan.count("--partition batch-8gpu") == 60
    assert plan.count("--gres gpu:nvidia_b200:8") == 60
    assert plan.count("--cpus-per-task 160") == 60
    assert plan.count("--mem 2400G") == 60
    assert plan.count("--dependency afterok:") == 59
    assert plan.count("--exclude DRYRUN-SAME-B200-NODE") == 909


def test_formal_plan_uses_only_canonical_comparisons_and_full_protocols(formal_plan):
    for misleading in ("HayateQuant-auto",):
        assert misleading not in formal_plan
    assert "FibQuant-paper" in formal_plan
    assert "BlockQuant-paper" in formal_plan
    assert "fibquant_variants=\\[\\{label:FibQuant-paper\\,bits:3\\}\\]" in formal_plan
    assert "blockquant_variants=\\[\\{label:BlockQuant-paper\\,bits:3\\}\\]" in formal_plan
    assert (
        "methods=\\[TurboQuant-paper\\,FibQuant-paper\\,BlockQuant-paper\\,"
        "HQ\\,AdaptiveHQ\\]" in formal_plan
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
    assert len(blocks) == 921
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
    assert len(keys) == 336
    assert "longcodeqa_context_llama31-8b.json" in formal_plan
    assert "longcodeqa_context_qwen3-8b.json" in formal_plan
    assert "+longcodeqa_brackets=\\[128K\\]" in formal_plan
    assert "+longcodeqa_start=88" in formal_plan
    assert "+longcodeqa_limit=4" in formal_plan
    assert "+longcodeqa_max_context=131072" in formal_plan
    assert "+longcodeqa_max_context=40960" in formal_plan


def test_longbench_is_exactly_one_complete_official_task_per_job(formal_plan):
    keys = re.findall(r"^JOB phase=formal key=(longbench-\S+)", formal_plan, re.M)
    assert len(keys) == 156
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
    assert len(keys) == 120
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
    assert formal_plan.count("--exclude DRYRUN-SAME-B200-NODE") == 909
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


def test_worker_rejects_result_that_appeared_while_pending(tmp_path):
    run_root = tmp_path / "run"
    result = run_root / "results/formal/already-there.json"
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
            "--job-key",
            "formal-race",
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
        env={**os.environ, "CODE_REPO_COMMIT": "a" * 40},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "refusing overwrite/resume race" in completed.stderr


def test_worker_result_lock_rejects_concurrent_writer(tmp_path):
    run_root = (tmp_path / "run").resolve()
    result = (run_root / "results/formal/concurrent.json").resolve()
    manifest = run_root / "manifests/inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    lock_id = hashlib.sha256(str(result).encode("utf-8") + b"\0").hexdigest()
    lock_path = run_root / "locks/results" / f"{lock_id}.lock"
    lock_path.parent.mkdir(parents=True)
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
                "--job-key",
                "formal-concurrent",
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
            env={**os.environ, "CODE_REPO_COMMIT": "a" * 40},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    assert completed.returncode == 2
    assert "already owns the result lock" in completed.stderr
