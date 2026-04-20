"""
lambda_function.py
──────────────────
AWS Lambda handler for rPPG processing.

Routes:
  POST  body={user_id (optional)}      → rppg_trigger_inference()
  GET   ?action=rppg_results&user_id=  → rppg_get_results()
"""

import json
import os
from lib.helper       import rppg_trigger_inference, rppg_get_results
from lib.request_data import ProcessLambdaInput


def lambda_handler(event, context):
    try:
        req    = ProcessLambdaInput(event, context)
        method = req.request_method
        result = None

        # ── GET routes ─────────────────────────────
        if method == "GET":
            action = req.rppg_action

            if action == "rppg_results":
                result = rppg_get_results(user_id=req.user_id)

            else:
                return _response(400, {"error": f"Unknown action: {action}"})

        # ── POST route ─────────────────────────────
        elif method == "POST":
            result = rppg_trigger_inference(user_id=req.user_id)

        else:
            return _response(405, {"error": f"Method not allowed: {method}"})

        return _response(200, result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _response(500, {"error": str(e)})


def _response(status_code: int, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type":                     "application/json",
            "Access-Control-Allow-Origin":      os.environ.get("HEADER_DOMAIN", "*"),
            "Access-Control-Allow-Credentials": True,
        },
        "body": json.dumps(body, default=str)
    }