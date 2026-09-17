# Publishing to crates.io

The workspace publishes as four crates. Because the domain crates depend on
`automl-core`, they must be published **in dependency order** — crates.io will
not accept a crate whose dependencies are not already on the index.

## Order

1. `automl-core`
2. `automl-tasks`, `automl-burn` (either order; both depend only on core)
3. `automl-cli`

## Pre-flight (all green as of 1.1.0)

```bash
cargo test --workspace --features sqlite
cargo clippy --all-targets --features sqlite -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features sqlite
```

Then verify packaging. `automl-core` fully dry-runs (packages + verifies a clean
build + would upload):

```bash
cargo publish --dry-run -p automl-core --features sqlite
```

The dependent crates cannot be *fully* dry-run until `automl-core` is live, since
cargo resolves their version requirement against the crates.io index. Their
manifests are valid (each references `automl-core` with both a path and a
`version`, via `[workspace.dependencies]`); confirm the file list with:

```bash
cargo package --list -p automl-burn --allow-dirty
```

## Publish

```bash
cargo publish -p automl-core --features sqlite
# wait for the index to update (usually under a minute), then:
cargo publish -p automl-tasks
cargo publish -p automl-burn
cargo publish -p automl-cli
```

Notes:
- `automl-core`'s `sqlite`/`postgres` features are optional; publishing does not
  require a database. `automl-burn`'s `wgpu` feature is optional and off by
  default, so the default publish stays CPU-only.
- After a version bump, update `version` in both `[workspace.package]` and the
  `[workspace.dependencies] automl-core` entry (they must match), or publishing
  the dependents will request a core version that is not yet on the index.
- Publishing is irreversible (a version cannot be re-uploaded, only yanked). Do a
  dry-run of `automl-core` first, and publish from a clean, tagged commit.

## docs.rs

docs.rs builds each crate's documentation automatically on publish. The
warning-free `cargo doc` gate above mirrors that build. To have docs.rs enable the
`sqlite` feature (so the persistent backend appears in the rendered docs), add to
`crates/automl-core/Cargo.toml`:

```toml
[package.metadata.docs.rs]
features = ["sqlite"]
```
