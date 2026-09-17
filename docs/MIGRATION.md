# Migration guide

`burn-automl` guarantees that a persisted study created under one version stays
loadable under later versions of the same major line (PRD §19.1, §22). This guide
documents the storage schema history and the upgrade/downgrade rules.

## How migrations work

The SQLite backend (`SqliteStorage`, behind the `sqlite` feature) records its
schema version in a `_schema_version` table. On `SqliteStorage::open()` it applies
every migration whose version is greater than the database's current version, in
order, inside the connection. Migrations are **monotonic integer versions** and
run automatically — opening an older database upgrades it in place; opening an
already-current database is a no-op.

Because every schema change after v1 is **additive** (new tables, or new
*nullable* columns), an upgrade never rewrites or drops existing data, and older
rows read back with sensible defaults.

## Schema history

| Version | Added in | Change | Compatibility |
|--------:|----------|--------|---------------|
| **v1** | v0.1 | Initial schema: `studies`, `trials`, `reports`. | — |
| **v2** | v0.1 | Per-trial provenance & timing on `trials`: nullable `env` (JSON `EnvSnapshot`), `queued_at`, `started_at`, `completed_at`. | v1 DB upgrades in place; old rows read back with default env and empty timing. |
| **v3** | v0.5 | `artifacts` table (per-trial named `BLOB`s: checkpoints, exported models, configs, logs). | Purely additive; studies without artifacts are unaffected. |
| **v4** | v0.6 | Distributed lease columns on `trials`: nullable `lease_owner`, `lease_expiry`. | Purely additive; non-distributed studies leave them `NULL`. |

## Upgrade path

Nothing to do: open the database with a newer `burn-automl` and the migrations
apply automatically.

```rust
use automl_core::prelude::*;

// A v0.1-era database opens under v1.0 and is upgraded to the latest schema.
let storage = SqliteStorage::open("study.db")?;
```

Forward compatibility holds **within a major version**: any 1.x release opens a
database written by any earlier 1.x (or by the 0.x releases that introduced these
same migrations).

## Downgrade path

No destructive migration ships without a documented downgrade path. All
migrations to date are additive, so **downgrading is safe by construction**: an
older `burn-automl` ignores the columns and tables it does not know about
(`SELECT` lists are explicit), and a study written by a newer version remains
readable by an older one for the fields that existed then. If a future release
ever introduces a non-additive migration, this section will document the explicit
steps to move a database back a version.

## In-memory and other backends

`InMemoryStorage` has no persistence and therefore no schema or migrations. The
`PostgresStorage` backend (added in 1.1.0, and since moved to the separate
`automl-postgres` crate) implements the same versioned migrations (v1–v4) behind
the same `Storage` trait, so the compatibility guarantees above apply to it
identically (PRD §18.4). It lives in its own crate — excluded from the main
workspace — so its `postgres` client dependency does not burden the default
build; use it with `automl-postgres = { path = "automl-postgres" }` (or its
published version) rather than a feature flag on `automl-core`.
