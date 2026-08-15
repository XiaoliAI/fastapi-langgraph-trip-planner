from uuid import uuid4

from ..models.schemas import (
    ChatMessage,
    PendingPatchIntent,
    TripPlan,
    TripRequest,
    TripSession,
)


class TripSessionService:
    def __init__(self):
        self._sessions: dict[str, TripSession] = {}

    def create_session(
        self,
        request: TripRequest,
        plan: TripPlan,
    ) -> TripSession:
        session = TripSession(
            id=str(uuid4()),
            request=request,
            current_plan=plan,
            messages=[],
            status="draft",
            pending_patch_intent=None,
            pending_revision_summary=None,
            plan_versions=[],
        )

        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> TripSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Trip session not found: {session_id}")
        return session

    def append_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.messages.append(message)
        self._sessions[session_id] = session
        return session

    def update_plan(
        self,
        session_id: str,
        plan: TripPlan,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.current_plan = plan
        self._sessions[session_id] = session
        return session

    def save_current_plan_version(
        self,
        session_id: str,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.plan_versions.append(session.current_plan.model_copy(deep=True))
        self._sessions[session_id] = session
        return session

    def update_pending_revision_summary(
        self,
        session_id: str,
        revision_summary: str,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.pending_revision_summary = revision_summary
        self._sessions[session_id] = session
        return session

    def clear_pending_revision_summary(
        self,
        session_id: str,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.pending_revision_summary = None
        self._sessions[session_id] = session
        return session

    def update_pending_patch_intent(
        self,
        session_id: str,
        pending_patch_intent: PendingPatchIntent,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.pending_patch_intent = pending_patch_intent
        self._sessions[session_id] = session
        return session

    def clear_pending_patch_intent(
        self,
        session_id: str,
    ) -> TripSession:
        session = self.get_session(session_id)
        session.pending_patch_intent = None
        self._sessions[session_id] = session
        return session

_trip_session_service = TripSessionService()


def get_trip_session_service() -> TripSessionService:
    return _trip_session_service


def reset_trip_session_service() -> None:
    global _trip_session_service
    _trip_session_service = TripSessionService()

def update_pending_patch_intent(
    self,
    session_id: str,
    pending_patch_intent: PendingPatchIntent,
) -> TripSession:
    session = self.get_session(session_id)
    session.pending_patch_intent = pending_patch_intent
    self._sessions[session_id] = session
    return session


def clear_pending_patch_intent(
    self,
    session_id: str,
) -> TripSession:
    session = self.get_session(session_id)
    session.pending_patch_intent = None
    self._sessions[session_id] = session
    return session
