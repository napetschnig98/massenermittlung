"""
supabase_client.py
------------------
Supabase connection + helper functions.

Tables:
  config  (key TEXT PK, value TEXT)       -- stores anthropic_api_key etc.
  results (id UUID PK, filename, raeume, fenster, tueren, konfidenz, methode, created_at)
"""

from __future__ import annotations
import os
from supabase import create_client, Client

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def get_anthropic_key() -> str:
    row = (
        get_supabase()
        .table("config")
        .select("value")
        .eq("key", "anthropic_api_key")
        .single()
        .execute()
    )
    return row.data["value"]


def save_result(filename: str, result: dict) -> str:
    """Insert a parsing result and return the new UUID."""
    resp = (
        get_supabase()
        .table("results")
        .insert({
            "filename": filename,
            "raeume": result.get("raeume", []),
            "fenster": result.get("fenster", []),
            "tueren": result.get("tueren", []),
            "konfidenz": result.get("konfidenz", {}),
            "methode": result.get("methode", ""),
        })
        .execute()
    )
    return resp.data[0]["id"]


def update_result(result_id: str, data: dict) -> None:
    get_supabase().table("results").update(data).eq("id", result_id).execute()


def get_results(limit: int = 20) -> list[dict]:
    return (
        get_supabase()
        .table("results")
        .select("id, filename, methode, konfidenz, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_result(result_id: str) -> dict | None:
    resp = (
        get_supabase()
        .table("results")
        .select("*")
        .eq("id", result_id)
        .single()
        .execute()
    )
    return resp.data
