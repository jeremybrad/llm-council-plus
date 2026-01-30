"""Evidence retrieval interface for SADB integration.

This module provides an async interface to retrieve evidence from the SADB
knowledge base (C003_sadb_canonical). Key features:

- Graceful degradation when SADB is unavailable
- Full provenance tracking for all retrieved evidence
- Uses asyncio.to_thread() to avoid blocking the event loop
- Hybrid retrieval (graph + vector + FTS) for best results
"""

import asyncio
import hashlib
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .claims import Evidence

# Path to SADB CLI module - configurable via environment variable
# Defaults to ~/SyncedProjects/C003_sadb_canonical/50_cli if not set
SADB_CLI_PATH = Path(os.environ.get("SADB_CLI_PATH", str(Path.home() / "SyncedProjects/C003_sadb_canonical/50_cli")))

# Neo4j connection URI - configurable via environment variable
# Defaults to localhost for host machine (not Docker)
SADB_NEO4J_URI = os.environ.get("SADB_NEO4J_URI", "neo4j://localhost:7687")
SADB_NEO4J_USER = os.environ.get("SADB_NEO4J_USER", "neo4j")
SADB_NEO4J_PASSWORD = os.environ.get("SADB_NEO4J_PASSWORD", "password")


def _check_sadb_available() -> bool:
    """Check if SADB is available for queries.

    Returns:
        True if SADB can be imported, False otherwise
    """
    # Temporarily add SADB to path
    sys.path.insert(0, str(SADB_CLI_PATH))
    try:
        from sadb_retrieve import retrieve  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False
    finally:
        # Clean up path
        if str(SADB_CLI_PATH) in sys.path:
            sys.path.remove(str(SADB_CLI_PATH))


def _do_sadb_retrieve(query: str, top_k: int):
    """Synchronous SADB retrieval (called in thread pool).

    Args:
        query: Search query
        top_k: Maximum results to return

    Returns:
        RetrievalBundle from SADB or None if unavailable
    """
    sys.path.insert(0, str(SADB_CLI_PATH))
    try:
        from sadb_retrieve import RetrieveOptions, retrieve

        opts = RetrieveOptions(
            candidate_mode="hybrid+fts",
            max_windows=top_k,
            max_facts=top_k,
        )

        return retrieve(query, k=top_k, opts=opts)
    except Exception as e:
        print(f"SADB retrieval error: {e}")
        return None
    finally:
        if str(SADB_CLI_PATH) in sys.path:
            sys.path.remove(str(SADB_CLI_PATH))


async def search_evidence(query: str, top_k: int = 5) -> list[Evidence]:
    """Search SADB for evidence supporting/contradicting a claim.

    Uses asyncio.to_thread() to avoid blocking the event loop.
    Returns evidence with full provenance.

    Args:
        query: Natural language query (usually the claim text)
        top_k: Maximum number of evidence pieces to return

    Returns:
        List of Evidence objects with provenance metadata
    """
    if not _check_sadb_available():
        return []  # Graceful degradation

    # Run sync retrieve in thread pool to avoid blocking event loop
    bundle = await asyncio.to_thread(_do_sadb_retrieve, query, top_k)

    if bundle is None:
        return []

    evidence = []
    now = datetime.now(timezone.utc).isoformat()

    # Process windows (transcript evidence)
    windows = getattr(bundle, "windows", []) or []
    for window in windows:
        try:
            # Extract snippet
            snippet = ""
            if hasattr(window, "snippet"):
                snippet = window.snippet[:500] if window.snippet else ""

            # Extract provenance
            provenance = getattr(window, "provenance", {}) or {}
            conversation_id = provenance.get("conversation_id", "")
            span_start = provenance.get("span_start")
            span_end = provenance.get("span_end")

            # Extract score
            score_components = getattr(window, "score_components", {}) or {}
            weight = score_components.get("vector", 0.5)

            evidence.append(
                Evidence(
                    evidence_id=str(uuid.uuid4()),
                    source_type="transcript",
                    source_id=conversation_id,
                    quote=snippet,
                    support="neutral",  # To be classified by scorer
                    weight=weight,
                    retrieved_at=now,
                    retrieval_query=query,
                    span_start=span_start,
                    span_end=span_end,
                    source_hash=hashlib.sha256(snippet.encode()).hexdigest()[:16] if snippet else None,
                )
            )
        except Exception as e:
            print(f"Error processing window: {e}")
            continue

    # Process facts (J1 fact evidence)
    facts = getattr(bundle, "facts", []) or []
    for fact in facts:
        try:
            # Extract claim text
            if hasattr(fact, "claim_text"):
                claim_text = fact.claim_text
            else:
                claim_text = str(fact)

            # Extract fact ID
            fact_id = getattr(fact, "fact_id", "") or ""

            # Extract confidence
            confidence = getattr(fact, "confidence", 0.7)
            if confidence is None:
                confidence = 0.7

            evidence.append(
                Evidence(
                    evidence_id=str(uuid.uuid4()),
                    source_type="j1_fact",
                    source_id=fact_id,
                    quote=claim_text,
                    support="supports",  # J1 facts are pre-extracted claims, default to supports
                    weight=confidence,
                    retrieved_at=now,
                    retrieval_query=query,
                    span_start=None,
                    span_end=None,
                    source_hash=hashlib.sha256(claim_text.encode()).hexdigest()[:16] if claim_text else None,
                )
            )
        except Exception as e:
            print(f"Error processing fact: {e}")
            continue

    return evidence


async def get_document(source_id: str) -> str | None:
    """Retrieve full document text for deep inspection.

    This is a placeholder for v2 implementation that would retrieve
    the full source document for more detailed analysis.

    Args:
        source_id: The document/conversation ID to retrieve

    Returns:
        Full document text or None if unavailable
    """
    if not _check_sadb_available():
        return None

    # Placeholder for v2 - would use SADB lookup_fact or conversation retrieval
    return None


def _check_neo4j_driver_available() -> tuple[bool, str | None]:
    """Check if neo4j Python driver is installed.

    Returns:
        Tuple of (available, error_message)
    """
    try:
        import neo4j  # noqa: F401

        return True, None
    except ImportError as e:
        return False, f"neo4j driver not installed: {e}"


def _check_neo4j_connectivity() -> tuple[bool, str | None]:
    """Check if Neo4j is reachable with a simple ping query.

    Returns:
        Tuple of (reachable, error_message)
    """
    try:
        from neo4j import GraphDatabase

        # Quick connectivity check with short timeout
        driver = GraphDatabase.driver(
            SADB_NEO4J_URI,
            auth=(SADB_NEO4J_USER, SADB_NEO4J_PASSWORD),
        )
        try:
            # Verify connectivity with a trivial query
            with driver.session() as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single()
                if record and record["ping"] == 1:
                    return True, None
                return False, "Neo4j ping query returned unexpected result"
        finally:
            driver.close()
    except Exception as e:
        # Sanitize error message (don't leak credentials)
        error_str = str(e)
        if SADB_NEO4J_PASSWORD in error_str:
            error_str = error_str.replace(SADB_NEO4J_PASSWORD, "***")
        return False, f"Neo4j unreachable ({SADB_NEO4J_URI}): {error_str}"


def get_sadb_status() -> dict:
    """Get SADB availability status with real readiness checks.

    Performs three-level check:
    1. SADB module importable (code available)
    2. Neo4j driver importable (dependency present)
    3. Neo4j connectivity (service reachable)

    Returns:
        Dict with:
        - sadb_importable: bool - can import SADB retrieve module
        - neo4j_driver_available: bool - neo4j Python package installed
        - neo4j_reachable: bool - can connect and ping Neo4j
        - sadb_ready: bool - all checks pass, retrieval should work
        - sadb_error: str | None - first error encountered (if any)
        - sadb_path: str - configured SADB CLI path
        - neo4j_uri: str - configured Neo4j URI (no credentials)
    """
    errors = []

    # Check 1: SADB module importable
    sadb_importable = _check_sadb_available()
    if not sadb_importable:
        errors.append("SADB module not importable")

    # Check 2: Neo4j driver available
    neo4j_driver_available, driver_error = _check_neo4j_driver_available()
    if not neo4j_driver_available and driver_error:
        errors.append(driver_error)

    # Check 3: Neo4j connectivity (only if driver is available)
    neo4j_reachable = False
    if neo4j_driver_available:
        neo4j_reachable, conn_error = _check_neo4j_connectivity()
        if not neo4j_reachable and conn_error:
            errors.append(conn_error)

    # Overall readiness: all checks must pass
    sadb_ready = sadb_importable and neo4j_driver_available and neo4j_reachable

    return {
        "sadb_importable": sadb_importable,
        "neo4j_driver_available": neo4j_driver_available,
        "neo4j_reachable": neo4j_reachable,
        "sadb_ready": sadb_ready,
        "sadb_error": errors[0] if errors else None,
        "sadb_path": str(SADB_CLI_PATH),
        "sadb_path_exists": SADB_CLI_PATH.exists(),
        "neo4j_uri": SADB_NEO4J_URI,
        # Legacy field for backwards compatibility
        "available": sadb_importable,
    }
