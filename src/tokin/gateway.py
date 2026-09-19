from __future__ import annotations

from http import HTTPStatus
from typing import ClassVar

from .session import Session


class GatewayError(Exception):
    """A request the gateway refuses; the subclass says with which status."""

    status_code: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR


class SessionNotFoundError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class DuplicateSessionError(GatewayError):
    status_code = HTTPStatus.CONFLICT


class Gateway:
    """tokin's core as a plain object; the HTTP layer maps routes onto its methods.

    Free of any web framework, so a trainer can drive it in-process and tests
    exercise it without a client. What it refuses, it refuses with a
    `GatewayError` whose subclass names the HTTP status, and the HTTP layer
    renders that once instead of choosing a status route by route.

    Sessions live in memory, keyed by id, from `create` until `delete`. There is
    no expiry: only the harness knows when a run is over, and the trainer reads
    the rollouts back after the last turn, so an idle session is not a stale one.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def create(self, id: str | None = None) -> Session:
        session = Session(id) if id else Session()
        if session.id in self.sessions:
            raise DuplicateSessionError(f"session {session.id!r} already exists")
        self.sessions[session.id] = session
        return session

    def get(self, id: str) -> Session:
        try:
            return self.sessions[id]
        except KeyError:
            raise SessionNotFoundError(f"no session {id!r}") from None

    def delete(self, id: str) -> Session:
        return self.sessions.pop(self.get(id).id)
