"""performance_stale on evaluation_windows, notification_attempts on alerts

Revision ID: 5e2c1a7b9d40
Revises: 10f9b6cf7932
Create Date: 2026-09-19 18:40:00.000000

Label ingestion no longer recomputes performance inside the request: it
flags the windows a batch touches and the scheduler recomputes them on its
next tick. A failed notification is retried on later ticks, a bounded number
of times, instead of being recorded once and forgotten.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5e2c1a7b9d40"
down_revision: Union[str, Sequence[str], None] = "10f9b6cf7932"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "evaluation_windows",
        sa.Column("performance_stale", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.create_index(
        "ix_evaluation_windows_performance_stale",
        "evaluation_windows",
        ["performance_stale"],
        postgresql_where=sa.text("performance_stale"),
    )
    op.add_column(
        "alerts",
        sa.Column("notification_attempts", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("alerts", "notification_attempts")
    op.drop_index("ix_evaluation_windows_performance_stale", table_name="evaluation_windows")
    op.drop_column("evaluation_windows", "performance_stale")
