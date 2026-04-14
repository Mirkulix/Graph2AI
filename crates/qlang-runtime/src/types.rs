//! QLANG Type System — Pattern matching, structs, enums, and runtime type checking.
//!
//! Extends the VM with user-defined types:
//! - Struct types with named fields
//! - Enum types with variants (optionally carrying data)
//! - Pattern matching on values
//! - A TypeRegistry for managing user-defined types

use std::collections::HashMap;
use std::fmt;

use crate::vm::{Value, VmError};

// ─── Type definitions ──────────────────────────────────────────────────────

/// A user-defined type: either a struct or an enum.
#[derive(Debug, Clone)]
pub enum TypeDef {
    Struct {
        name: String,
        fields: Vec<String>,
    },
    Enum {
        name: String,
        variants: Vec<EnumVariantDef>,
    },
}

/// Definition of a single enum variant.
#[derive(Debug, Clone)]
pub struct EnumVariantDef {
    pub name: String,
    /// If true, this variant carries one data payload.
    pub has_data: bool,
}

/// Convenience alias.
pub type StructDef = TypeDef;
/// Convenience alias.
pub type EnumDef = TypeDef;

// ─── Runtime values ────────────────────────────────────────────────────────

/// A struct instance with named fields.
#[derive(Debug, Clone)]
pub struct StructValue {
    pub type_name: String,
    pub fields: HashMap<String, Value>,
}

impl StructValue {
    /// Create a new struct value, validating fields against the definition.
    pub fn new(
        def: &TypeDef,
        field_values: HashMap<String, Value>,
    ) -> Result<Self, VmError> {
        match def {
            TypeDef::Struct { name, fields } => {
                // Check for unknown fields.
                for key in field_values.keys() {
                    if !fields.contains(&key.clone()) {
                        return Err(VmError::TypeError(format!(
                            "unknown field `{key}` on struct `{name}`"
                        )));
                    }
                }
                // Check all declared fields are provided.
                for f in fields {
                    if !field_values.contains_key(f) {
                        return Err(VmError::TypeError(format!(
                            "missing field `{f}` on struct `{name}`"
                        )));
                    }
                }
                Ok(StructValue {
                    type_name: name.clone(),
                    fields: field_values,
                })
            }
            TypeDef::Enum { name, .. } => Err(VmError::TypeError(format!(
                "`{name}` is an enum, not a struct"
            ))),
        }
    }

    /// Access a field by name.
    pub fn get_field(&self, field: &str) -> Result<Value, VmError> {
        self.fields.get(field).cloned().ok_or_else(|| {
            VmError::TypeError(format!(
                "unknown field `{field}` on struct `{}`",
                self.type_name
            ))
        })
    }
}

impl fmt::Display for StructValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {{ ", self.type_name)?;
        let mut first = true;
        for (k, v) in &self.fields {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{k}: {v}")?;
            first = false;
        }
        write!(f, " }}")
    }
}

impl PartialEq for StructValue {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name && self.fields == other.fields
    }
}

/// An enum variant instance, optionally carrying data.
#[derive(Debug, Clone)]
pub struct EnumValue {
    pub type_name: String,
    pub variant: String,
    pub data: Option<Box<Value>>,
}

impl EnumValue {
    /// Create a new enum value, validating against the definition.
    pub fn new(
        def: &TypeDef,
        variant: &str,
        data: Option<Value>,
    ) -> Result<Self, VmError> {
        match def {
            TypeDef::Enum { name, variants } => {
                let vdef = variants.iter().find(|v| v.name == variant).ok_or_else(|| {
                    VmError::TypeError(format!(
                        "unknown variant `{variant}` on enum `{name}`"
                    ))
                })?;
                if vdef.has_data && data.is_none() {
                    return Err(VmError::TypeError(format!(
                        "variant `{name}::{variant}` expects data"
                    )));
                }
                if !vdef.has_data && data.is_some() {
                    return Err(VmError::TypeError(format!(
                        "variant `{name}::{variant}` does not accept data"
                    )));
                }
                Ok(EnumValue {
                    type_name: name.clone(),
                    variant: variant.to_string(),
                    data: data.map(Box::new),
                })
            }
            TypeDef::Struct { name, .. } => Err(VmError::TypeError(format!(
                "`{name}` is a struct, not an enum"
            ))),
        }
    }
}

impl fmt::Display for EnumValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.type_name, self.variant)?;
        if let Some(d) = &self.data {
            write!(f, "({})", d)?;
        }
        Ok(())
    }
}

impl PartialEq for EnumValue {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && self.variant == other.variant
            && self.data == other.data
    }
}

// ─── Extended Value (wraps vm::Value + our new types) ──────────────────────

/// An extended runtime value that includes structs and enums alongside the
/// base VM values.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtValue {
    Base(Value),
    Struct(StructValue),
    Enum(EnumValue),
}

impl fmt::Display for ExtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtValue::Base(v) => write!(f, "{v}"),
            ExtValue::Struct(s) => write!(f, "{s}"),
            ExtValue::Enum(e) => write!(f, "{e}"),
        }
    }
}

impl ExtValue {
    /// Return a human-readable type name.
    pub fn type_name(&self) -> String {
        match self {
            ExtValue::Base(v) => match v {
                Value::Number(_) => "number".to_string(),
                Value::Bool(_) => "bool".to_string(),
                Value::String(_) => "string".to_string(),
                Value::Array(_) => "array".to_string(),
                Value::Tensor(_, _) => "tensor".to_string(),
                Value::Dict(_) => "dict".to_string(),
                Value::Null => "null".to_string(),
            },
            ExtValue::Struct(s) => s.type_name.clone(),
            ExtValue::Enum(e) => e.type_name.clone(),
        }
    }

    /// Field access for struct values (e.g. `point.x`).
    pub fn get_field(&self, field: &str) -> Result<ExtValue, VmError> {
        match self {
            ExtValue::Struct(s) => s.get_field(field).map(ExtValue::Base),
            other => Err(VmError::TypeError(format!(
                "cannot access field `{field}` on {}",
                other.type_name()
            ))),
        }
    }
}

// ─── Pattern matching ──────────────────────────────────────────────────────

/// A single pattern in a match arm.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Match a literal number.
    Number(f64),
    /// Match a literal string.
    StringLit(String),
    /// Match a literal bool.
    Bool(bool),
    /// Match null.
    Null,
    /// Match an enum variant by `Type::Variant`, optionally binding inner data
    /// to the given variable name.
    EnumVariant {
        type_name: String,
        variant: String,
        binding: Option<String>,
    },
    /// Wildcard `_` — matches anything.
    Wildcard,
}

/// A match arm: pattern => result value.
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub result: ExtValue,
}

/// Execute pattern matching: try each arm in order, return the result of the
/// first matching arm.  If no arm matches, return an error.
pub fn match_value(value: &ExtValue, arms: &[MatchArm]) -> Result<ExtValue, VmError> {
    for arm in arms {
        if let Some(result) = try_match(value, arm) {
            return Ok(result);
        }
    }
    Err(VmError::RuntimeError(format!(
        "no matching pattern for value `{value}`"
    )))
}

/// Try to match a single arm. Returns `Some(result)` on success.
fn try_match(value: &ExtValue, arm: &MatchArm) -> Option<ExtValue> {
    match &arm.pattern {
        Pattern::Wildcard => Some(arm.result.clone()),
        Pattern::Number(n) => match value {
            ExtValue::Base(Value::Number(v)) if (v - n).abs() < f64::EPSILON => {
                Some(arm.result.clone())
            }
            _ => None,
        },
        Pattern::StringLit(s) => match value {
            ExtValue::Base(Value::String(v)) if v == s => Some(arm.result.clone()),
            _ => None,
        },
        Pattern::Bool(b) => match value {
            ExtValue::Base(Value::Bool(v)) if v == b => Some(arm.result.clone()),
            _ => None,
        },
        Pattern::Null => match value {
            ExtValue::Base(Value::Null) => Some(arm.result.clone()),
            _ => None,
        },
        Pattern::EnumVariant {
            type_name,
            variant,
            binding: _,
        } => match value {
            ExtValue::Enum(e) if e.type_name == *type_name && e.variant == *variant => {
                Some(arm.result.clone())
            }
            _ => None,
        },
    }
}

// ─── Type registry ─────────────────────────────────────────────────────────

/// Stores user-defined types and provides factory methods.
#[derive(Debug, Clone, Default)]
pub struct TypeRegistry {
    types: HashMap<String, TypeDef>,
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a struct definition.
    pub fn register_struct(&mut self, name: &str, fields: Vec<String>) -> Result<(), VmError> {
        if self.types.contains_key(name) {
            return Err(VmError::TypeError(format!(
                "type `{name}` is already defined"
            )));
        }
        self.types.insert(
            name.to_string(),
            TypeDef::Struct {
                name: name.to_string(),
                fields,
            },
        );
        Ok(())
    }

    /// Register an enum definition.
    pub fn register_enum(
        &mut self,
        name: &str,
        variants: Vec<EnumVariantDef>,
    ) -> Result<(), VmError> {
        if self.types.contains_key(name) {
            return Err(VmError::TypeError(format!(
                "type `{name}` is already defined"
            )));
        }
        self.types.insert(
            name.to_string(),
            TypeDef::Enum {
                name: name.to_string(),
                variants,
            },
        );
        Ok(())
    }

    /// Look up a type by name.
    pub fn get(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    /// Create a struct instance.
    pub fn create_struct(
        &self,
        type_name: &str,
        field_values: HashMap<String, Value>,
    ) -> Result<StructValue, VmError> {
        let def = self.types.get(type_name).ok_or_else(|| {
            VmError::TypeError(format!("unknown type `{type_name}`"))
        })?;
        StructValue::new(def, field_values)
    }

    /// Create an enum variant instance.
    pub fn create_enum(
        &self,
        type_name: &str,
        variant: &str,
        data: Option<Value>,
    ) -> Result<EnumValue, VmError> {
        let def = self.types.get(type_name).ok_or_else(|| {
            VmError::TypeError(format!("unknown type `{type_name}`"))
        })?;
        EnumValue::new(def, variant, data)
    }

    /// Runtime type check: is the value an instance of the named type?
    pub fn is_type(&self, value: &ExtValue, type_name: &str) -> bool {
        match value {
            ExtValue::Base(v) => match v {
                Value::Number(_) => type_name == "number",
                Value::Bool(_) => type_name == "bool",
                Value::String(_) => type_name == "string",
                Value::Array(_) => type_name == "array",
                Value::Tensor(_, _) => type_name == "tensor",
                Value::Dict(_) => type_name == "dict",
                Value::Null => type_name == "null",
            },
            ExtValue::Struct(s) => s.type_name == type_name,
            ExtValue::Enum(e) => e.type_name == type_name,
        }
    }
}

// ─── Convenience: is_type free function ────────────────────────────────────

/// Convenience wrapper: runtime type check without needing the registry (uses
/// the value's own type name).
pub fn is_type(value: &ExtValue, type_name: &str) -> bool {
    value.type_name() == type_name
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry_with_point() -> TypeRegistry {
        let mut reg = TypeRegistry::new();
        reg.register_struct("Point", vec!["x".into(), "y".into()])
            .unwrap();
        reg
    }

    fn make_registry_with_color() -> TypeRegistry {
        let mut reg = TypeRegistry::new();
        reg.register_enum(
            "Color",
            vec![
                EnumVariantDef { name: "Red".into(), has_data: false },
                EnumVariantDef { name: "Green".into(), has_data: false },
                EnumVariantDef { name: "Blue".into(), has_data: false },
            ],
        )
        .unwrap();
        reg
    }

    fn make_registry_with_option() -> TypeRegistry {
        let mut reg = TypeRegistry::new();
        reg.register_enum(
            "Option",
            vec![
                EnumVariantDef { name: "Some".into(), has_data: true },
                EnumVariantDef { name: "None".into(), has_data: false },
            ],
        )
        .unwrap();
        reg
    }

    // 1. Create struct with fields
    #[test]
    fn test_create_struct() {
        let reg = make_registry_with_point();
        let mut fields = HashMap::new();
        fields.insert("x".into(), Value::Number(1.0));
        fields.insert("y".into(), Value::Number(2.0));
        let s = reg.create_struct("Point", fields).unwrap();
        assert_eq!(s.type_name, "Point");
        assert_eq!(s.fields.len(), 2);
    }

    // 2. Access struct fields
    #[test]
    fn test_access_struct_fields() {
        let reg = make_registry_with_point();
        let mut fields = HashMap::new();
        fields.insert("x".into(), Value::Number(3.0));
        fields.insert("y".into(), Value::Number(4.0));
        let s = reg.create_struct("Point", fields).unwrap();
        assert_eq!(s.get_field("x").unwrap(), Value::Number(3.0));
        assert_eq!(s.get_field("y").unwrap(), Value::Number(4.0));
    }

    // 3. Create enum variant (no data)
    #[test]
    fn test_create_enum_variant() {
        let reg = make_registry_with_color();
        let e = reg.create_enum("Color", "Red", None).unwrap();
        assert_eq!(e.type_name, "Color");
        assert_eq!(e.variant, "Red");
        assert!(e.data.is_none());
    }

    // 4. Match on number
    #[test]
    fn test_match_number() {
        let val = ExtValue::Base(Value::Number(1.0));
        let arms = vec![
            MatchArm {
                pattern: Pattern::Number(0.0),
                result: ExtValue::Base(Value::String("zero".into())),
            },
            MatchArm {
                pattern: Pattern::Number(1.0),
                result: ExtValue::Base(Value::String("one".into())),
            },
            MatchArm {
                pattern: Pattern::Wildcard,
                result: ExtValue::Base(Value::String("other".into())),
            },
        ];
        let result = match_value(&val, &arms).unwrap();
        assert_eq!(result, ExtValue::Base(Value::String("one".into())));
    }

    // 5. Match on enum variant
    #[test]
    fn test_match_enum_variant() {
        let reg = make_registry_with_color();
        let val = ExtValue::Enum(reg.create_enum("Color", "Green", None).unwrap());
        let arms = vec![
            MatchArm {
                pattern: Pattern::EnumVariant {
                    type_name: "Color".into(),
                    variant: "Red".into(),
                    binding: None,
                },
                result: ExtValue::Base(Value::String("red!".into())),
            },
            MatchArm {
                pattern: Pattern::EnumVariant {
                    type_name: "Color".into(),
                    variant: "Green".into(),
                    binding: None,
                },
                result: ExtValue::Base(Value::String("green!".into())),
            },
        ];
        let result = match_value(&val, &arms).unwrap();
        assert_eq!(result, ExtValue::Base(Value::String("green!".into())));
    }

    // 6. Wildcard pattern
    #[test]
    fn test_wildcard_pattern() {
        let val = ExtValue::Base(Value::Number(999.0));
        let arms = vec![MatchArm {
            pattern: Pattern::Wildcard,
            result: ExtValue::Base(Value::String("caught".into())),
        }];
        let result = match_value(&val, &arms).unwrap();
        assert_eq!(result, ExtValue::Base(Value::String("caught".into())));
    }

    // 7. Nested struct (struct field holds another struct via Base value representation)
    #[test]
    fn test_nested_struct() {
        let mut reg = TypeRegistry::new();
        reg.register_struct("Point", vec!["x".into(), "y".into()])
            .unwrap();
        reg.register_struct("Line", vec!["start".into(), "end".into()])
            .unwrap();

        // Create inner points as Values by encoding coordinates.
        let mut p1_fields = HashMap::new();
        p1_fields.insert("x".into(), Value::Number(0.0));
        p1_fields.insert("y".into(), Value::Number(0.0));
        let p1 = reg.create_struct("Point", p1_fields).unwrap();

        let mut p2_fields = HashMap::new();
        p2_fields.insert("x".into(), Value::Number(1.0));
        p2_fields.insert("y".into(), Value::Number(1.0));
        let p2 = reg.create_struct("Point", p2_fields).unwrap();

        // Wrap inner structs as ExtValue and verify field access.
        let ext_p1 = ExtValue::Struct(p1);
        let ext_p2 = ExtValue::Struct(p2);

        // Verify nested field access works on inner struct.
        let inner_x = ext_p1.get_field("x").unwrap();
        assert_eq!(inner_x, ExtValue::Base(Value::Number(0.0)));

        let inner_y = ext_p2.get_field("y").unwrap();
        assert_eq!(inner_y, ExtValue::Base(Value::Number(1.0)));
    }

    // 8. Type checking
    #[test]
    fn test_type_checking() {
        let reg = make_registry_with_point();
        let mut fields = HashMap::new();
        fields.insert("x".into(), Value::Number(1.0));
        fields.insert("y".into(), Value::Number(2.0));
        let s = reg.create_struct("Point", fields).unwrap();

        let ext = ExtValue::Struct(s);
        assert!(reg.is_type(&ext, "Point"));
        assert!(!reg.is_type(&ext, "Color"));

        let num = ExtValue::Base(Value::Number(42.0));
        assert!(is_type(&num, "number"));
        assert!(!is_type(&num, "string"));
    }

    // 9. Error: unknown field
    #[test]
    fn test_error_unknown_field() {
        let reg = make_registry_with_point();
        let mut fields = HashMap::new();
        fields.insert("x".into(), Value::Number(1.0));
        fields.insert("y".into(), Value::Number(2.0));
        fields.insert("z".into(), Value::Number(3.0)); // unknown

        let result = reg.create_struct("Point", fields);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown field"), "got: {err}");
    }

    // 10. Error: no matching pattern
    #[test]
    fn test_error_no_matching_pattern() {
        let val = ExtValue::Base(Value::Number(42.0));
        let arms = vec![
            MatchArm {
                pattern: Pattern::Number(0.0),
                result: ExtValue::Base(Value::String("zero".into())),
            },
            MatchArm {
                pattern: Pattern::Number(1.0),
                result: ExtValue::Base(Value::String("one".into())),
            },
        ];
        let result = match_value(&val, &arms);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no matching pattern"), "got: {err}");
    }

    // 11. Enum variant with data (Option::Some(42))
    #[test]
    fn test_enum_variant_with_data() {
        let reg = make_registry_with_option();
        let e = reg
            .create_enum("Option", "Some", Some(Value::Number(42.0)))
            .unwrap();
        assert_eq!(e.variant, "Some");
        assert_eq!(*e.data.as_ref().unwrap(), Box::new(Value::Number(42.0)));
    }

    // 12. Display formatting
    #[test]
    fn test_display_formatting() {
        let reg = make_registry_with_color();
        let e = reg.create_enum("Color", "Red", None).unwrap();
        assert_eq!(format!("{e}"), "Color::Red");

        let reg2 = make_registry_with_option();
        let e2 = reg2
            .create_enum("Option", "Some", Some(Value::Number(42.0)))
            .unwrap();
        assert_eq!(format!("{e2}"), "Option::Some(42)");
    }
}
