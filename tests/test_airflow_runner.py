"""Tests for DAG run success, failure, and timeouts"""

import subprocess
from unittest.mock import patch

import pytest

from data_pipeline.orchestration import run_dag as runner


@pytest.fixture
def runner_environment():
    clock = {"now": 0.0}

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += seconds

    with (
        patch.object(runner.time, "monotonic", side_effect=fake_monotonic),
        patch.object(runner.time, "sleep", side_effect=fake_sleep),
        patch.object(runner, "dag_is_ready", return_value=True),
        patch.object(runner, "get_run_state", return_value="success"),
        patch.object(runner.subprocess, "run") as trigger,
    ):
        yield trigger


def test_waits_for_the_selected_run_to_succeed(runner_environment):
    with (
        patch.object(runner, "dag_is_ready", side_effect=[False, True]),
        patch.object(
            runner, "get_run_state", side_effect=[None, "queued", "running", "success"]
        ) as states,
    ):
        assert runner.main(["--run-id", "test-success", "--timeout", "60"]) == 0
    runner_environment.assert_called_once_with(
        ["airflow", "dags", "trigger", "--run-id", "test-success", runner.DAG_ID],
        check=True,
        timeout=30,
    )
    states.assert_called_with(runner.DAG_ID, "test-success")


def test_failed_dag_returns_a_failure(runner_environment, capsys):
    with patch.object(runner, "get_run_state", return_value="failed"):
        assert runner.main(["--run-id", "test-failure"]) == 1
    assert "test-failure failed" in capsys.readouterr().out


def test_missing_dag_times_out_without_a_trigger(runner_environment, capsys):
    with patch.object(runner, "dag_is_ready", return_value=False):
        assert runner.main(["--run-id", "test-missing", "--timeout", "10"]) == 1
    runner_environment.assert_not_called()
    assert "was not ready before the timeout" in capsys.readouterr().out


def test_unfinished_run_times_out(runner_environment, capsys):
    with patch.object(runner, "get_run_state", return_value="running"):
        assert runner.main(["--run-id", "test-timeout", "--timeout", "10"]) == 1
    assert "did not finish before the timeout" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, ["airflow", "dags", "trigger"]),
        subprocess.TimeoutExpired(["airflow", "dags", "trigger"], 30),
    ],
)
def test_trigger_errors_return_a_failure(runner_environment, error):
    runner_environment.side_effect = error

    assert runner.main(["--run-id", "test-trigger-error"]) == 1
    runner.get_run_state.assert_not_called()


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_timeout_must_be_positive(runner_environment, timeout):
    with pytest.raises(SystemExit) as error:
        runner.main(["--run-id", "test-invalid", "--timeout", timeout])
    assert error.value.code == 2
    runner_environment.assert_not_called()
