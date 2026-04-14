# QLANG for Visual Studio Code

Language support for `.qlang` files -- Graph-based AI-to-AI Programming.

## Features

- Syntax highlighting for QLANG keywords, types, built-in operations, and annotations
- Bracket matching and auto-closing for `{}`, `[]`, `()`, quotes
- Line comment toggling with `//`
- Code folding based on braces

## Installation

Copy the extension folder into your VS Code extensions directory:

```bash
# Linux / macOS
cp -r qlang/editors/vscode ~/.vscode/extensions/qlang

# Windows
xcopy /E /I qlang\editors\vscode %USERPROFILE%\.vscode\extensions\qlang
```

Then restart VS Code. Files with the `.qlang` extension will automatically receive syntax highlighting.

## Highlighted Elements

| Category    | Tokens                                                                 |
|-------------|------------------------------------------------------------------------|
| Keywords    | `graph`, `input`, `output`, `node`, `let`, `fn`, `if`, `else`, `for`, `while`, `return`, `import`, `export`, `print` |
| Types       | `f32`, `f64`, `i32`, `ternary`, `bool`                                |
| Operations  | `matmul`, `relu`, `sigmoid`, `tanh`, `softmax`, `add`, `mul`, `to_ternary`, `evolve`, `measure` |
| Annotations | `@proof` and other `@`-prefixed annotations                           |
| Literals    | Strings (`"..."`, `'...'`), integers, floats, hex numbers             |
| Comments    | Line comments with `//`                                               |

## Screenshot

*Screenshot placeholder -- add a screenshot of QLANG syntax highlighting here.*

## Requirements

- Visual Studio Code 1.80.0 or later

## License

See the repository root for license information.
