"""
ProcessLambdaInput Dataclass
============================
Parses AWS Lambda HTTP events.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class ProcessLambdaInput:
    event:   Dict[str, Any]
    context: Any

    event_body:           Optional[str]            = field(init=False, repr=False)
    request_body:         Optional[Dict[str, Any]]  = field(init=False, default=None)
    origin:               Optional[str]             = field(init=False, default=None)
    event_query_params:   Optional[str]             = field(init=False, repr=False)
    request_query_params: Optional[Dict[str, Any]]  = field(init=False, default=None)
    request_method:       str                       = field(init=False, default=None)
    request_header:       Any                       = field(init=False, default=None)
    request_cookies:      Any                       = field(init=False, default=None)

    client_id:   str           = field(init=False, default=None)
    report_type: str           = field(init=False, default=None)
    user_id:     Optional[str] = field(init=False, default=None)
    rppg_action: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        # Parse POST body
        self.event_body   = self.event.get("body", "")
        self.request_body = json.loads(self.event_body) if self.event_body else {}

        # Parse GET query parameters
        self.event_query_params   = self.event.get("queryStringParameters", {})
        self.request_query_params = self.event_query_params if self.event_query_params else {}

        # HTTP metadata
        self.request_method  = self.event.get("httpMethod", "")
        self.request_header  = self.event.get("headers", {})
        self.origin          = self.request_header.get("origin", "origin header not found")
        self.request_cookies = (
            self.request_header.get("cookie", "")
            if self.request_header.get("cookie", "")
            else self.request_header.get("Cookie", "")
        )

        # Common parameters
        self.client_id   = self.request_query_params.get("client_id", "")
        self.report_type = self.request_query_params.get("report_type", "")

        # rPPG fields — user_id from GET query params or POST body
        self.rppg_action = self.request_query_params.get("action", "")
        self.user_id     = (
            self.request_query_params.get("user_id") or
            self.request_body.get("user_id") or
            None
        )