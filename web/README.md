# Web app

The Next.js frontend for the E-Commerce Intelligence Engine. See the main [README](../README.md) for the
full project and setup.

```powershell
npm install
npm run dev        # http://127.0.0.1:3000, needs the API running (eie serve run)
npm run build      # production build
npm run start      # serve the production build on 127.0.0.1:3000
npm run lint
npm run typecheck
npm run gen:api    # regenerate lib/api-schema.ts from lib/openapi.json
```

- **Pages:** `app/` holds thin route files; each renders a client view from `components/views/`.
- **API access:** the browser only calls this app. `app/api/[...path]/route.ts` forwards requests to
  FastAPI and adds the bearer key on the server, and `app/images/[file]/route.ts` serves thumbnails.
  The key is read from `EIE_API_KEY`, or from the repository's `.env`.
- **Types:** `lib/api-schema.ts` is generated from the API's OpenAPI schema. After changing an API
  response model, run `eie serve openapi` from the repository root, then `npm run gen:api`.
- **Settings:** `EIE_API_URL` (default `http://127.0.0.1:8000`) and `EIE_API_KEY`.
