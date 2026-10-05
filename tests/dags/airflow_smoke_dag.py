"""Offline DAGs for real scheduler checks"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task


@dag(
    dag_id="airflow-smoke-success",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
)
def smoke_success_dag():
    @task()
    def first_task():
        return 1

    @task()
    def second_task(value):
        assert value == 1

    second_task(first_task())


@dag(
    dag_id="airflow-smoke-failure",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=1)},
)
def smoke_failure_dag():
    @task()
    def first_task():
        raise RuntimeError("This test task must fail.")

    @task()
    def second_task():
        raise RuntimeError("This task must not run after an upstream failure.")

    first_task() >> second_task()


smoke_success = smoke_success_dag()
smoke_failure = smoke_failure_dag()
