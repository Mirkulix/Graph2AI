//! QLANG Module System — Import/export and multi-file programs
//!
//! Enables splitting QLANG programs across multiple files:
//! ```qlang
//! import "std/math"
//! import "./helpers.qlang"
//!
//! export graph my_model {
//!   input x: f32[4]
//!   // ...
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A parsed import statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportStatement {
    /// The raw path string from the import (e.g. `"std/math"`, `"./helpers.qlang"`).
    pub path: String,
    /// Line number where the import appeared (1-based).
    pub line: usize,
}

/// An exported item from a module.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportedItem {
    Function { name: String, body: String },
    Graph { name: String, body: String },
}

/// A single loaded module.
#[derive(Debug, Clone)]
pub struct Module {
    /// Module name (derived from path).
    pub name: String,
    /// Canonical file path this module was loaded from.
    pub path: PathBuf,
    /// Exported functions.
    pub exported_functions: Vec<ExportedItem>,
    /// Exported graphs.
    pub exported_graphs: Vec<ExportedItem>,
    /// The full source text of the module (after stripping imports).
    pub body: String,
}

/// Errors that can occur during module resolution.
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error("line {line}: invalid import path '{path}'")]
    InvalidImportPath { line: usize, path: String },

    #[error("module not found: '{0}'")]
    ModuleNotFound(String),

    #[error("circular import detected: {0}")]
    CircularImport(String),

    #[error("parse error in module '{module}': {message}")]
    ParseError { module: String, message: String },

    #[error("I/O error: {0}")]
    IoError(String),
}

/// The kind of import path.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportKind {
    /// Built-in module (e.g. `"math"`).
    Builtin(String),
    /// Relative file path (e.g. `"./helpers.qlang"`).
    Relative(String),
    /// Standard library module (e.g. `"std/arrays"`).
    StdLib(String),
}

/// Classifies a raw import path string into an `ImportKind`.
pub fn classify_import(path: &str) -> ImportKind {
    if path.starts_with("./") || path.starts_with("../") {
        ImportKind::Relative(path.to_string())
    } else if path.starts_with("std/") {
        ImportKind::StdLib(path.to_string())
    } else {
        ImportKind::Builtin(path.to_string())
    }
}

// ---------------------------------------------------------------------------
// Import parsing
// ---------------------------------------------------------------------------

/// Parse all `import` statements from the top of a source string.
///
/// Import statements must appear before any other non-comment, non-blank line.
/// Returns the list of imports and the remaining source with imports stripped.
pub fn parse_imports(source: &str) -> Result<(Vec<ImportStatement>, String), ModuleError> {
    let mut imports = Vec::new();
    let mut body_lines: Vec<&str> = Vec::new();
    let mut done_with_imports = false;

    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        if !done_with_imports {
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                body_lines.push(line);
                continue;
            }
            if trimmed.starts_with("import ") {
                let rest = trimmed["import ".len()..].trim();
                // Expect a quoted string.
                let path = parse_quoted_string(rest).ok_or_else(|| ModuleError::ParseError {
                    module: "<current>".into(),
                    message: format!("line {line_num}: expected quoted string after 'import'"),
                })?;
                imports.push(ImportStatement {
                    path,
                    line: line_num,
                });
                continue; // don't add import lines to the body
            }
            // First non-import, non-comment line — switch to body mode.
            done_with_imports = true;
        }
        body_lines.push(line);
    }

    Ok((imports, body_lines.join("\n")))
}

/// Parse `export fn` and `export graph` markers in source, returning
/// `(exports, cleaned_source)` where cleaned_source has `export ` prefixes removed.
pub fn parse_exports(source: &str) -> (Vec<ExportedItem>, String) {
    let mut exports = Vec::new();
    let mut output_lines: Vec<String> = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        if trimmed.starts_with("export fn ") {
            let fn_header = &trimmed["export fn ".len()..];
            let name = fn_header
                .split(|c: char| c == '(' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .to_string();
            let body = collect_braced_block(&lines, i);
            exports.push(ExportedItem::Function {
                name,
                body: body.clone(),
            });
            // Emit line without the "export " prefix
            output_lines.push(lines[i].replacen("export ", "", 1));
            i += 1;
        } else if trimmed.starts_with("export graph ") {
            let graph_header = &trimmed["export graph ".len()..];
            let name = graph_header
                .split(|c: char| c == '{' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .to_string();
            let body = collect_braced_block(&lines, i);
            exports.push(ExportedItem::Graph {
                name,
                body: body.clone(),
            });
            output_lines.push(lines[i].replacen("export ", "", 1));
            i += 1;
        } else {
            output_lines.push(lines[i].to_string());
            i += 1;
        }
    }

    (exports, output_lines.join("\n"))
}

/// Collect lines from `start` until the matching closing `}` (inclusive),
/// joining them into a single string.
fn collect_braced_block(lines: &[&str], start: usize) -> String {
    let mut depth: i32 = 0;
    let mut collected = Vec::new();
    for line in &lines[start..] {
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
            }
        }
        collected.push(*line);
        if depth <= 0 && collected.len() > 0 {
            // If the opening brace hasn't appeared yet, keep going.
            if depth == 0 && collected.iter().any(|l| l.contains('{')) {
                break;
            }
        }
    }
    collected.join("\n")
}

/// Extract a double-quoted string, returning its contents.
fn parse_quoted_string(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// ModuleResolver
// ---------------------------------------------------------------------------

/// Resolves module import paths to canonical file paths.
pub struct ModuleResolver {
    /// Root directory of the current project (used for relative imports).
    _project_root: PathBuf,
    /// Directory for standard library modules.
    std_lib_dir: PathBuf,
    /// Built-in module names (do not need files on disk).
    builtins: HashSet<String>,
}

impl ModuleResolver {
    /// Create a new resolver.
    ///
    /// * `project_root` – base directory for resolving relative imports.
    /// * `std_lib_dir`  – directory containing standard-library `.qlang` files.
    pub fn new(project_root: impl Into<PathBuf>, std_lib_dir: impl Into<PathBuf>) -> Self {
        let mut builtins = HashSet::new();
        builtins.insert("math".into());
        builtins.insert("io".into());
        builtins.insert("random".into());

        Self {
            _project_root: project_root.into(),
            std_lib_dir: std_lib_dir.into(),
            builtins,
        }
    }

    /// Register an additional built-in module name.
    pub fn add_builtin(&mut self, name: impl Into<String>) {
        self.builtins.insert(name.into());
    }

    /// Resolve an import path to a canonical file path (or a sentinel for builtins).
    pub fn resolve(
        &self,
        import_path: &str,
        current_file_dir: &Path,
    ) -> Result<ResolvedPath, ModuleError> {
        match classify_import(import_path) {
            ImportKind::Builtin(name) => {
                if self.builtins.contains(&name) {
                    Ok(ResolvedPath::Builtin(name))
                } else {
                    Err(ModuleError::ModuleNotFound(name))
                }
            }
            ImportKind::Relative(rel) => {
                let resolved = current_file_dir.join(&rel);
                let canonical = normalize_path(&resolved);
                Ok(ResolvedPath::File(canonical))
            }
            ImportKind::StdLib(path) => {
                // "std/arrays" → <std_lib_dir>/arrays.qlang
                let sub = path.strip_prefix("std/").unwrap_or(&path);
                let file = self.std_lib_dir.join(format!("{sub}.qlang"));
                let canonical = normalize_path(&file);
                Ok(ResolvedPath::File(canonical))
            }
        }
    }
}

/// Result of resolving an import path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolvedPath {
    /// A built-in module (no file on disk).
    Builtin(String),
    /// A file on disk.
    File(PathBuf),
}

/// Simple path normalization (resolve `.` and `..` components) without hitting
/// the filesystem (so tests don't need real files).
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                components.pop();
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

// ---------------------------------------------------------------------------
// ModuleRegistry
// ---------------------------------------------------------------------------

/// Tracks which modules have been loaded and detects circular imports.
pub struct ModuleRegistry {
    /// Modules that have been fully loaded, keyed by resolved path.
    loaded: HashMap<ResolvedPath, Module>,
    /// Modules currently being loaded (import chain), for cycle detection.
    loading: HashSet<ResolvedPath>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            loading: HashSet::new(),
        }
    }

    /// Return a previously-loaded module, if any.
    pub fn get(&self, key: &ResolvedPath) -> Option<&Module> {
        self.loaded.get(key)
    }

    /// Mark a module as currently being loaded (begin of import).
    /// Returns an error if it is already in the loading set (circular import).
    pub fn begin_loading(&mut self, key: &ResolvedPath) -> Result<(), ModuleError> {
        if self.loading.contains(key) {
            return Err(ModuleError::CircularImport(format!("{key:?}")));
        }
        self.loading.insert(key.clone());
        Ok(())
    }

    /// Finish loading a module: move it from the loading set into the loaded map.
    pub fn finish_loading(&mut self, key: ResolvedPath, module: Module) {
        self.loading.remove(&key);
        self.loaded.insert(key, module);
    }

    /// Check whether a module has already been fully loaded.
    pub fn is_loaded(&self, key: &ResolvedPath) -> bool {
        self.loaded.contains_key(key)
    }

    /// Return all loaded modules.
    pub fn modules(&self) -> impl Iterator<Item = &Module> {
        self.loaded.values()
    }
}

// ---------------------------------------------------------------------------
// ResolvedProgram & resolve_imports
// ---------------------------------------------------------------------------

/// The result of resolving all imports: a single merged program.
#[derive(Debug, Clone)]
pub struct ResolvedProgram {
    /// All modules that were loaded (in dependency order).
    pub modules: Vec<Module>,
    /// The merged source text (bodies concatenated in dependency order).
    pub merged_source: String,
}

/// A function that can read source text given a file path.
/// Used to abstract over the filesystem during testing.
pub type SourceLoader = Box<dyn Fn(&Path) -> Result<String, ModuleError>>;

/// Resolve all imports starting from `source` (located at `source_path`)
/// and produce a `ResolvedProgram`.
///
/// `loader` is called to read the contents of imported files.
pub fn resolve_imports(
    source: &str,
    source_path: &Path,
    resolver: &ModuleResolver,
    loader: &SourceLoader,
) -> Result<ResolvedProgram, ModuleError> {
    let mut registry = ModuleRegistry::new();
    let mut ordered_modules: Vec<Module> = Vec::new();

    resolve_recursive(
        source,
        source_path,
        resolver,
        loader,
        &mut registry,
        &mut ordered_modules,
    )?;

    let merged_source = ordered_modules
        .iter()
        .map(|m| m.body.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(ResolvedProgram {
        modules: ordered_modules,
        merged_source,
    })
}

fn resolve_recursive(
    source: &str,
    source_path: &Path,
    resolver: &ModuleResolver,
    loader: &SourceLoader,
    registry: &mut ModuleRegistry,
    ordered: &mut Vec<Module>,
) -> Result<(), ModuleError> {
    let source_dir = source_path.parent().unwrap_or(Path::new("."));
    let (imports, remaining) = parse_imports(source)?;
    let (exports, clean_body) = parse_exports(&remaining);

    let self_key = ResolvedPath::File(normalize_path(source_path));
    registry.begin_loading(&self_key)?;

    // Process each import, depth-first.
    for imp in &imports {
        let resolved = resolver.resolve(&imp.path, source_dir).map_err(|_| {
            ModuleError::InvalidImportPath {
                line: imp.line,
                path: imp.path.clone(),
            }
        })?;

        if registry.is_loaded(&resolved) {
            continue; // already loaded
        }

        match &resolved {
            ResolvedPath::Builtin(name) => {
                let builtin_mod = make_builtin_module(name);
                registry.begin_loading(&resolved)?;
                registry.finish_loading(resolved.clone(), builtin_mod.clone());
                ordered.push(builtin_mod);
            }
            ResolvedPath::File(file_path) => {
                let child_source = loader(file_path)?;
                resolve_recursive(
                    &child_source,
                    file_path,
                    resolver,
                    loader,
                    registry,
                    ordered,
                )?;
            }
        }
    }

    let mut exported_functions = Vec::new();
    let mut exported_graphs = Vec::new();
    for e in exports {
        match &e {
            ExportedItem::Function { .. } => exported_functions.push(e),
            ExportedItem::Graph { .. } => exported_graphs.push(e),
        }
    }

    let module_name = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unnamed")
        .to_string();

    let module = Module {
        name: module_name,
        path: normalize_path(source_path),
        exported_functions,
        exported_graphs,
        body: clean_body,
    };

    registry.finish_loading(self_key, module.clone());
    ordered.push(module);

    Ok(())
}

/// Create a stub `Module` for a built-in module.
fn make_builtin_module(name: &str) -> Module {
    let body = match name {
        "math" => "// built-in math module: sin, cos, exp, log, sqrt, abs, pow",
        "io" => "// built-in io module: print, read",
        "random" => "// built-in random module: rand, seed",
        _ => "// built-in module",
    };
    Module {
        name: name.to_string(),
        path: PathBuf::from(format!("<builtin:{name}>")),
        exported_functions: Vec::new(),
        exported_graphs: Vec::new(),
        body: body.to_string(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Parse import statement
    // -----------------------------------------------------------------------
    #[test]
    fn parse_import_statement() {
        let source = r#"
import "std/math"
import "./helpers.qlang"

graph foo {
  input x: f32[4]
  output y = x
}
"#;
        let (imports, body) = parse_imports(source).unwrap();
        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].path, "std/math");
        assert_eq!(imports[1].path, "./helpers.qlang");
        assert!(body.contains("graph foo"));
        // Import lines should NOT appear in body.
        assert!(!body.contains("import "));
    }

    // -----------------------------------------------------------------------
    // 2. Resolve built-in module
    // -----------------------------------------------------------------------
    #[test]
    fn resolve_builtin_module() {
        let resolver = ModuleResolver::new("/project", "/project/std");
        let result = resolver.resolve("math", Path::new("/project")).unwrap();
        assert_eq!(result, ResolvedPath::Builtin("math".into()));
    }

    // -----------------------------------------------------------------------
    // 3. Resolve relative path
    // -----------------------------------------------------------------------
    #[test]
    fn resolve_relative_path() {
        let resolver = ModuleResolver::new("/project", "/project/std");
        let result = resolver
            .resolve("./helpers.qlang", Path::new("/project/src"))
            .unwrap();
        assert_eq!(
            result,
            ResolvedPath::File(PathBuf::from("/project/src/helpers.qlang"))
        );
    }

    // -----------------------------------------------------------------------
    // 4. Detect circular imports
    // -----------------------------------------------------------------------
    #[test]
    fn detect_circular_imports() {
        let resolver = ModuleResolver::new("/project", "/project/std");

        // a.qlang imports b.qlang, b.qlang imports a.qlang
        let loader: SourceLoader = Box::new(|path: &Path| {
            let name = path.file_name().unwrap().to_str().unwrap();
            match name {
                "a.qlang" => Ok(r#"import "./b.qlang"
graph a { input x: f32[1] output y = x }"#
                    .into()),
                "b.qlang" => Ok(r#"import "./a.qlang"
graph b { input x: f32[1] output y = x }"#
                    .into()),
                _ => Err(ModuleError::ModuleNotFound(name.into())),
            }
        });

        let source = r#"import "./b.qlang"
graph main { input x: f32[1] output y = x }"#;
        let result = resolve_imports(
            source,
            Path::new("/project/a.qlang"),
            &resolver,
            &loader,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ModuleError::CircularImport(_)),
            "expected CircularImport, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Export functions
    // -----------------------------------------------------------------------
    #[test]
    fn export_functions() {
        let source = r#"export fn foo(x) {
  return x + 1
}

export graph my_model {
  input x: f32[4]
  output y = x
}

graph private_graph {
  input a: f32[2]
  output b = a
}
"#;
        let (exports, cleaned) = parse_exports(source);
        assert_eq!(exports.len(), 2);

        assert!(matches!(&exports[0], ExportedItem::Function { name, .. } if name == "foo"));
        assert!(matches!(&exports[1], ExportedItem::Graph { name, .. } if name == "my_model"));

        // "export " prefix should be stripped in the cleaned source.
        assert!(!cleaned.contains("export fn"));
        assert!(!cleaned.contains("export graph"));
        assert!(cleaned.contains("fn foo(x)"));
        assert!(cleaned.contains("graph my_model"));
        assert!(cleaned.contains("graph private_graph"));
    }

    // -----------------------------------------------------------------------
    // 6. Module registry
    // -----------------------------------------------------------------------
    #[test]
    fn module_registry_load_and_query() {
        let mut reg = ModuleRegistry::new();
        let key = ResolvedPath::Builtin("math".into());

        assert!(!reg.is_loaded(&key));

        reg.begin_loading(&key).unwrap();
        let m = make_builtin_module("math");
        reg.finish_loading(key.clone(), m);

        assert!(reg.is_loaded(&key));
        let loaded = reg.get(&key).unwrap();
        assert_eq!(loaded.name, "math");
    }

    // -----------------------------------------------------------------------
    // 7. Multiple imports
    // -----------------------------------------------------------------------
    #[test]
    fn multiple_imports_resolved() {
        let resolver = ModuleResolver::new("/project", "/project/std");

        let loader: SourceLoader = Box::new(|path: &Path| {
            let name = path.file_name().unwrap().to_str().unwrap();
            match name {
                "helpers.qlang" => Ok("// helper utilities\n".into()),
                "arrays.qlang" => Ok("// std arrays\n".into()),
                _ => Err(ModuleError::ModuleNotFound(name.into())),
            }
        });

        let source = r#"import "math"
import "./helpers.qlang"
import "std/arrays"

graph main {
  input x: f32[4]
  output y = x
}
"#;
        let program = resolve_imports(
            source,
            Path::new("/project/main.qlang"),
            &resolver,
            &loader,
        )
        .unwrap();

        // 3 imported modules + the main module = 4
        assert_eq!(program.modules.len(), 4);
        assert!(program.merged_source.contains("graph main"));
    }

    // -----------------------------------------------------------------------
    // 8. Invalid import path error
    // -----------------------------------------------------------------------
    #[test]
    fn invalid_import_path_error() {
        let resolver = ModuleResolver::new("/project", "/project/std");

        let loader: SourceLoader = Box::new(|_: &Path| {
            Err(ModuleError::ModuleNotFound("no_such_file".into()))
        });

        let source = r#"import "nonexistent_builtin"

graph main {
  input x: f32[4]
  output y = x
}
"#;
        let result = resolve_imports(
            source,
            Path::new("/project/main.qlang"),
            &resolver,
            &loader,
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Additional: classify_import
    // -----------------------------------------------------------------------
    #[test]
    fn classify_import_kinds() {
        assert_eq!(classify_import("math"), ImportKind::Builtin("math".into()));
        assert_eq!(
            classify_import("./foo.qlang"),
            ImportKind::Relative("./foo.qlang".into())
        );
        assert_eq!(
            classify_import("std/arrays"),
            ImportKind::StdLib("std/arrays".into())
        );
    }

    // -----------------------------------------------------------------------
    // Additional: resolve std lib path
    // -----------------------------------------------------------------------
    #[test]
    fn resolve_std_lib_path() {
        let resolver = ModuleResolver::new("/project", "/usr/share/qlang/std");
        let result = resolver
            .resolve("std/arrays", Path::new("/project/src"))
            .unwrap();
        assert_eq!(
            result,
            ResolvedPath::File(PathBuf::from("/usr/share/qlang/std/arrays.qlang"))
        );
    }
}
