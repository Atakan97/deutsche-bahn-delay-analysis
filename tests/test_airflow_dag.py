"""Tests for the Airflow DAG and dbt connection setup"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

from data_pipeline.orchestration.flows import (  # noqa: E402
    elt_pipeline,
    run_dbt_transformations,
)


def test_elt_dag_is_active_serial_and_ordered() -> None:
    assert elt_pipeline.dag_id == "deutsche-bahn-elt-pipeline"
    assert elt_pipeline.catchup is False
    assert elt_pipeline.is_paused_upon_creation is False
    assert elt_pipeline.max_active_runs == 1
    assert set(elt_pipeline.task_ids) == {"extract_and_load", "transform"}
    assert elt_pipeline.get_task("transform").upstream_task_ids == {
        "extract_and_load"
    }
    assert elt_pipeline.get_task("extract_and_load").retries == 2


def test_local_dag_has_a_fifteen_minute_schedule() -> None:
    environment = os.environ.copy()
    environment.pop("PIPELINE_SCHEDULE", None)

    python_code = (
        "from data_pipeline.orchestration.flows import elt_pipeline\n"
        "assert elt_pipeline.schedule_interval == '*/15 * * * *'"
    )

    subprocess.run(
        [sys.executable, "-c", python_code],
        env=environment,
        check=True,
    )


def test_github_dag_only_uses_external_triggers() -> None:
    environment = os.environ.copy()
    environment["PIPELINE_SCHEDULE"] = "none"

    python_code = (
        "from data_pipeline.orchestration.flows import elt_pipeline\n"
        "assert elt_pipeline.schedule_interval is None"
    )

    subprocess.run(
        [sys.executable, "-c", python_code],
        env=environment,
        check=True,
    )


def test_airflow_can_discover_the_production_dag() -> None:
    dags_folder = Path(__file__).resolve().parent.parent / "dags"
    dag_bag = DagBag(dag_folder=str(dags_folder), include_examples=False)

    assert dag_bag.import_errors == {}
    assert "deutsche-bahn-elt-pipeline" in dag_bag.dags


def test_database_url_is_decoded_for_dbt() -> None:
    database_url = (
        "postgresql://postgres.project:p%40ss%23word@"
        "pooler.example.com:5432/postgres?sslmode=require"
    )

    with (
        patch.dict(os.environ, {"DATABASE_URL": database_url}, clear=True),
        patch(
            "data_pipeline.orchestration.flows.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
        ) as dbt_command,
    ):
        run_dbt_transformations()

        assert os.environ["SUPABASE_HOST"] == "pooler.example.com"
        assert os.environ["SUPABASE_PORT"] == "5432"
        assert os.environ["SUPABASE_USER"] == "postgres.project"
        assert os.environ["SUPABASE_PASSWORD"] == "p@ss#word"
        assert os.environ["SUPABASE_DB"] == "postgres"
        assert os.environ["SUPABASE_SSLMODE"] == "require"
        assert dbt_command.call_args.args[0] == ["dbt", "run", "--profiles-dir", "."]
