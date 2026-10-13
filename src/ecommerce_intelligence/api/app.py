"""FastAPI application: `uvicorn ecommerce_intelligence.api.app:app`."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ecommerce_intelligence.api import schemas
from ecommerce_intelligence.api.settings import Settings
from ecommerce_intelligence.copilot.agent import Agent, ChatModel, GeminiChat
from ecommerce_intelligence.copilot.prompt import system_prompt
from ecommerce_intelligence.copilot.sandbox import TABLES, Sandbox
from ecommerce_intelligence.copilot.tools import DECLARATIONS, Toolbox
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.encoder import Encoder
from ecommerce_intelligence.search.index import ProductIndex
from ecommerce_intelligence.serving.store import NotFound, ServingStore

ARTICLE_ID = r"^\d{10}$"
CUSTOMER_ID = r"^[0-9a-f]{64}$"
bearer = HTTPBearer(auto_error=False)


def _store(request: Request) -> ServingStore:
    return request.app.state.store


def _index(request: Request) -> ProductIndex:
    return request.app.state.index


# Module-level aliases: annotations are resolved as strings in module scope.
Store = Annotated[ServingStore, Depends(_store)]
Index = Annotated[ProductIndex, Depends(_index)]


class TextSearch:
    """Loads the FashionCLIP text encoder on first use, once."""

    def __init__(self, factory: Callable[[], Encoder]):
        self._factory = factory
        self._encoder: Encoder | None = None
        self._lock = threading.Lock()

    def encoder(self) -> Encoder:
        with self._lock:
            if self._encoder is None:
                self._encoder = self._factory()
            return self._encoder

    def warm_up(self) -> None:
        """Load the model in the background so the first query does not wait."""
        threading.Thread(
            target=self.encoder, name="text-search-warm-up", daemon=True
        ).start()


def create_app(
    settings: Settings | None = None,
    paths: ProjectPaths | None = None,
    encoder_factory: Callable[[], Encoder] | None = None,
    chat_model_factory: Callable[[], ChatModel] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    paths = paths or ProjectPaths.default()

    if chat_model_factory is None and settings.gemini_api_key:

        def chat_model_factory() -> ChatModel:
            return GeminiChat(
                settings.gemini_api_key.get_secret_value(),
                settings.copilot_model,
                settings.copilot_fallback_model,
            )

    if chat_model_factory is not None:
        # One shared client per app, so quota exhaustion it observes on one
        # request routes later requests to the fallback model.
        build_chat_model, shared = chat_model_factory, {}
        shared_lock = threading.Lock()

        def chat_model_factory() -> ChatModel:
            with shared_lock:
                if "model" not in shared:
                    shared["model"] = build_chat_model()
                return shared["model"]

    if encoder_factory is None and settings.eie_text_search:

        def encoder_factory() -> Encoder:
            from ecommerce_intelligence.search.encoder import FashionClipEncoder

            return FashionClipEncoder(device=settings.eie_search_device)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = ServingStore(paths)
        app.state.index = ProductIndex(ArticleEmbeddings.load(paths.embeddings))
        app.state.text_search = TextSearch(encoder_factory) if encoder_factory else None
        if app.state.text_search is not None:
            app.state.text_search.warm_up()
        app.state.sandbox = Sandbox(paths)
        app.state.system_prompt = system_prompt(app.state.sandbox.schema())
        yield

    app = FastAPI(
        title="E-Commerce Intelligence Engine API",
        version=version("ecommerce-intelligence-engine"),
        description=(
            "Recommendations, product search, and customer intelligence over the "
            "H&M Personalized Fashion dataset (historical, as of 2020-09-23). "
            "Prices and values are H&M's normalized price units."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        expected = (
            settings.eie_api_key.get_secret_value() if settings.eie_api_key else ""
        )
        if not expected:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Set EIE_API_KEY to enable the API"
            )
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode(), expected.encode()
        ):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid bearer key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    secured = [Depends(authorize)]

    def not_found(kind: str, key: str) -> HTTPException:
        return HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown {kind} {key}")

    # System --------------------------------------------------------------------

    @app.get("/health", response_model=schemas.Health, tags=["system"])
    def health(request: Request, data: Store) -> dict:
        return {
            "status": "ready",
            "version": app.version,
            "data_end": data.manifest["data_end"],
            "built_at": data.manifest["built_at"],
            "counts": {k: int(v) for k, v in data.counts().items()},
            "text_search": request.app.state.text_search is not None,
        }

    @app.get("/api/models", tags=["system"], dependencies=secured)
    def models(data: Store) -> dict:
        """Evaluation reports: data quality, recommender, search, and customers."""
        return data.reports

    @app.get("/api/overview", tags=["analytics"], dependencies=secured)
    def overview(data: Store) -> dict:
        """KPIs, weekly trends, category mix, age bands, and cohort retention."""
        return data.overview

    # Products ------------------------------------------------------------------

    @app.get(
        "/api/products",
        response_model=schemas.ArticlePage,
        tags=["products"],
        dependencies=secured,
    )
    def browse(
        data: Store,
        index_group: str | None = None,
        product_group: str | None = None,
        colour: str | None = None,
        q: Annotated[str | None, Query(max_length=80)] = None,
        sort: Literal["popular", "newest", "price_low", "price_high"] = "popular",
        limit: Annotated[int, Query(ge=1, le=100)] = 24,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        total, items = data.browse(
            index_group=index_group,
            product_group=product_group,
            colour=colour,
            text=q,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        return {"total": total, "items": items}

    @app.get(
        "/api/products/facets",
        response_model=schemas.Facets,
        tags=["products"],
        dependencies=secured,
    )
    def facets(data: Store) -> dict:
        return data.facets()

    @app.get(
        "/api/products/{article_id}",
        response_model=schemas.ArticleDetail,
        tags=["products"],
        dependencies=secured,
    )
    def product(
        article_id: Annotated[str, Path(pattern=ARTICLE_ID)], data: Store
    ) -> dict:
        try:
            return data.article(article_id)
        except NotFound:
            raise not_found("article", article_id) from None

    @app.get(
        "/api/products/{article_id}/similar",
        response_model=list[schemas.SearchHit],
        tags=["products"],
        dependencies=secured,
    )
    def similar(
        article_id: Annotated[str, Path(pattern=ARTICLE_ID)],
        data: Store,
        search: Index,
        k: Annotated[int, Query(ge=1, le=50)] = 12,
    ) -> list[dict]:
        try:
            hits = search.similar(article_id, k)
        except KeyError:
            raise not_found("article", article_id) from None
        details = data.articles([h.article_id for h in hits])
        return [{"score": h.score, "article": details[h.article_id]} for h in hits]

    @app.get(
        "/api/search",
        response_model=list[schemas.SearchHit],
        tags=["products"],
        dependencies=secured,
    )
    def text_search(
        request: Request,
        data: Store,
        search: Index,
        q: Annotated[str, Query(min_length=2, max_length=120)],
        k: Annotated[int, Query(ge=1, le=50)] = 24,
    ) -> list[dict]:
        """Natural-language search over product photos and descriptions."""
        engine: TextSearch | None = request.app.state.text_search
        if engine is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Text search is off"
            )
        hits = search.search_text(engine.encoder(), q, k)
        details = data.articles([h.article_id for h in hits])
        return [{"score": h.score, "article": details[h.article_id]} for h in hits]

    @app.get("/images/{article_id}.webp", tags=["products"], include_in_schema=False)
    def image(article_id: str, data: Store) -> FileResponse:
        path = data.thumbnail(article_id) if len(article_id) == 10 else None
        if path is None:
            raise not_found("image", article_id)
        return FileResponse(
            path,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Customers -----------------------------------------------------------------

    @app.get(
        "/api/customers/sample",
        response_model=list[schemas.CustomerSample],
        tags=["customers"],
        dependencies=secured,
    )
    def sample(
        data: Store,
        segment_id: int | None = None,
        risk_band: Literal["low", "medium", "high"] | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        seed: int = 0,
    ) -> list[dict]:
        """A repeatable sample of active customers to explore."""
        return data.sample_customers(
            segment_id=segment_id, risk_band=risk_band, limit=limit, seed=seed
        )

    @app.get(
        "/api/customers/lookup",
        response_model=list[str],
        tags=["customers"],
        dependencies=secured,
    )
    def lookup(
        data: Store,
        prefix: Annotated[
            str, Query(min_length=4, max_length=64, pattern=r"^[0-9a-f]+$")
        ],
        limit: Annotated[int, Query(ge=1, le=20)] = 10,
    ) -> list[str]:
        """Customer IDs starting with a prefix (IDs are 64-character hashes)."""
        return data.customer_exists_prefix(prefix, limit)

    @app.get(
        "/api/customers/{customer_id}",
        response_model=schemas.Customer,
        tags=["customers"],
        dependencies=secured,
    )
    def customer(
        customer_id: Annotated[str, Path(pattern=CUSTOMER_ID)], data: Store
    ) -> dict:
        try:
            return data.customer(customer_id)
        except NotFound:
            raise not_found("customer", customer_id) from None

    @app.get(
        "/api/customers/{customer_id}/history",
        response_model=list[schemas.Purchase],
        tags=["customers"],
        dependencies=secured,
    )
    def history(
        customer_id: Annotated[str, Path(pattern=CUSTOMER_ID)],
        data: Store,
        limit: Annotated[int, Query(ge=1, le=200)] = 30,
    ) -> list[dict]:
        rows = data.history(customer_id, limit)
        if not rows:
            try:
                data.customer(customer_id)
            except NotFound:
                raise not_found("customer", customer_id) from None
        return rows

    @app.get(
        "/api/customers/{customer_id}/recommendations",
        response_model=list[schemas.Recommendation],
        tags=["customers"],
        dependencies=secured,
    )
    def recommendations(
        customer_id: Annotated[str, Path(pattern=CUSTOMER_ID)], data: Store
    ) -> list[dict]:
        try:
            return data.recommendations(customer_id)
        except NotFound:
            raise not_found("customer", customer_id) from None

    # Segments and campaigns ----------------------------------------------------

    @app.get(
        "/api/segments",
        response_model=list[schemas.Segment],
        tags=["customers"],
        dependencies=secured,
    )
    def segments(data: Store) -> list[dict]:
        return data.segments()

    @app.post(
        "/api/campaigns/plan",
        response_model=schemas.CampaignPlan,
        tags=["customers"],
        dependencies=secured,
    )
    def plan(request_body: schemas.CampaignRequest, data: Store) -> dict:
        """Rank customers by expected value saved and report spend, value, and ROI."""
        return data.plan_campaign(**request_body.model_dump())

    # Copilot -------------------------------------------------------------------

    @app.get(
        "/api/copilot/status",
        response_model=schemas.CopilotStatus,
        tags=["copilot"],
        dependencies=secured,
    )
    def copilot_status() -> dict:
        return {
            "configured": chat_model_factory is not None,
            "model": settings.copilot_model,
            "tables": sorted(TABLES),
        }

    @app.post("/api/copilot/chat", tags=["copilot"], dependencies=secured)
    def copilot_chat(
        request: Request, body: schemas.ChatRequest, data: Store, search: Index
    ) -> StreamingResponse:
        """Answer a question with tools; streams Server-Sent Events.

        Each event is one JSON object: tool_call, table, chart, customer,
        products, campaign, tool_error, thinking, answer, error, and finally done.
        """
        if chat_model_factory is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Set GEMINI_API_KEY in .env to enable the copilot",
            )
        if body.messages[-1].role != "user":
            raise HTTPException(
                422,
                "The last message must be from the user",
            )
        text_search: TextSearch | None = request.app.state.text_search

        def encoder() -> Encoder:
            if text_search is None:
                raise RuntimeError("Text search is off")
            return text_search.encoder()

        toolbox = Toolbox(request.app.state.sandbox, data, search, encoder)
        agent = Agent(
            chat_model_factory(), toolbox, request.app.state.system_prompt, DECLARATIONS
        )
        history = [m.model_dump() for m in body.messages[:-1]]

        def events():
            for event in agent.run(body.messages[-1].content, history):
                yield "data: " + json.dumps(event, default=str) + "\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def __getattr__(name: str):
    # `uvicorn ecommerce_intelligence.api.app:app` builds the app on first access,
    # so importing this module (for tests or tooling) has no side effects.
    if name == "app":
        return create_app()
    raise AttributeError(name)
