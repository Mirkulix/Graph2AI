[OPEN] Frontend 404 Debug Session

Session ID: `frontend-404`

Symptom:
- User reports a `404` while using the running QO system.
- Direct checks already showed `GET /`, `GET /api/health`, and `GET /?secondary=multi-agent` returning `200`.

Current hypotheses:
1. The `404` is caused by a missing frontend subresource rather than the main HTML document.
2. The `404` happens on a path-style navigation such as `/multi-agent` rather than query-param navigation.
3. The SPA fallback/static serving layer does not handle some browser-requested paths.
4. A generated browser URL does not map to any backend route or static asset.
5. The issue is reproducible only under a specific browser refresh/navigation flow.

Evidence collected:
- `GET /` -> `200`
- `GET /api/health` -> `200`
- `GET /?secondary=multi-agent` -> `200`

Next steps:
1. Inspect server static-file/SPA fallback routing.
2. Reproduce likely 404 candidate URLs without changing business logic.
3. Add minimal route-level instrumentation if reproduction still unclear.
4. Fix only after evidence points to the failing path.
