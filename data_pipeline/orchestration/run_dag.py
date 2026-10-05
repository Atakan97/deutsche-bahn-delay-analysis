"""Trigger an Airflow DAG and wait for its result"""

import argparse
import subprocess
import time

DAG_ID = "deutsche-bahn-elt-pipeline"
POLL_INTERVAL_SECONDS = 5


def dag_is_ready(dag_id: str) -> bool:
    """Check that the scheduler has saved the DAG"""
    from airflow.models import DagModel
    from airflow.models.serialized_dag import SerializedDagModel
    from airflow.utils.session import create_session

    with create_session() as session:
        dag = session.query(DagModel).filter_by(dag_id=dag_id).first()
        saved_dag = session.query(SerializedDagModel).filter_by(dag_id=dag_id).first()
        return dag is not None and not dag.is_paused and saved_dag is not None


def get_run_state(dag_id: str, run_id: str) -> str | None:
    """Read the state of one DAG run"""
    from airflow.models import DagRun
    from airflow.utils.session import create_session

    with create_session() as session:
        dag_run = session.query(DagRun).filter_by(dag_id=dag_id, run_id=run_id).first()
        if dag_run is None:
            return None
        return dag_run.state


def wait_for_dag(dag_id: str, deadline: float) -> None:
    """Wait until the DAG is ready for a trigger"""
    while time.monotonic() < deadline:
        if dag_is_ready(dag_id):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"DAG {dag_id} was not ready before the timeout.")


def wait_for_run(dag_id: str, run_id: str, deadline: float) -> None:
    """Wait for success or failure from the scheduler"""
    previous_state = None
    while time.monotonic() < deadline:
        state = get_run_state(dag_id, run_id)
        if state != previous_state:
            print(f"DAG {dag_id}, run {run_id}: {state}", flush=True)
            previous_state = state
        if state == "success":
            return
        if state == "failed":
            raise RuntimeError(f"DAG {dag_id}, run {run_id} failed.")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"DAG {dag_id}, run {run_id} did not finish before the timeout.")


def run_dag(dag_id: str, run_id: str, timeout_seconds: int) -> None:
    """Trigger one run and check its final state"""
    deadline = time.monotonic() + timeout_seconds
    wait_for_dag(dag_id, deadline)
    time_left = deadline - time.monotonic()
    if time_left <= 0:
        raise TimeoutError("The timeout expired before the DAG trigger.")
    subprocess.run(
        ["airflow", "dags", "trigger", "--run-id", run_id, dag_id],
        check=True,
        timeout=min(30, time_left),
    )
    wait_for_run(dag_id, run_id, deadline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-id", default=DAG_ID)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        run_dag(args.dag_id, args.run_id, args.timeout)
    except (
        RuntimeError,
        TimeoutError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Airflow run failed: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
