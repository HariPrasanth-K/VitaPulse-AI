"""
rPPG Helper Functions
=====================
Trigger inference and fetch results from DB.
"""

from lib.rppg_pipeline import run_rppg_pipeline
from lib.util import get_rppg_results


def rppg_trigger_inference(user_id=None):
    """
    Triggers rPPG pipeline for one user or all users.
    Returns list of { user_id, reading_id } dicts.
    """
    return run_rppg_pipeline(user_id=user_id)


def rppg_get_results(user_id=None):
    """
    Fetch rPPG results from DB.
    Returns list of result rows.
    """
    return get_rppg_results(user_id=user_id)