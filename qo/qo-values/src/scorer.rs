use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    Achtsamkeit,
    Anerkennung,
    Aufmerksamkeit,
    Entwicklung,
    Sinn,
}

impl Value {
    pub const ALL: [Value; 5] = [
        Value::Achtsamkeit,
        Value::Anerkennung,
        Value::Aufmerksamkeit,
        Value::Entwicklung,
        Value::Sinn,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueScores {
    pub achtsamkeit: f32,
    pub anerkennung: f32,
    pub aufmerksamkeit: f32,
    pub entwicklung: f32,
    pub sinn: f32,
}

impl Default for ValueScores {
    fn default() -> Self {
        Self {
            achtsamkeit: 0.5,
            anerkennung: 0.5,
            aufmerksamkeit: 0.5,
            entwicklung: 0.5,
            sinn: 0.5,
        }
    }
}

impl ValueScores {
    pub fn get(&self, value: Value) -> f32 {
        match value {
            Value::Achtsamkeit => self.achtsamkeit,
            Value::Anerkennung => self.anerkennung,
            Value::Aufmerksamkeit => self.aufmerksamkeit,
            Value::Entwicklung => self.entwicklung,
            Value::Sinn => self.sinn,
        }
    }

    pub fn set(&mut self, value: Value, score: f32) {
        let clamped = score.clamp(0.0, 1.0);
        match value {
            Value::Achtsamkeit => self.achtsamkeit = clamped,
            Value::Anerkennung => self.anerkennung = clamped,
            Value::Aufmerksamkeit => self.aufmerksamkeit = clamped,
            Value::Entwicklung => self.entwicklung = clamped,
            Value::Sinn => self.sinn = clamped,
        }
    }

    pub fn average(&self) -> f32 {
        (self.achtsamkeit
            + self.anerkennung
            + self.aufmerksamkeit
            + self.entwicklung
            + self.sinn)
            / 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scores_are_half() {
        let scores = ValueScores::default();
        for value in Value::ALL {
            assert_eq!(scores.get(value), 0.5);
        }
    }

    #[test]
    fn set_clamps_to_range() {
        let mut scores = ValueScores::default();
        scores.set(Value::Sinn, 2.0);
        assert_eq!(scores.get(Value::Sinn), 1.0);
        scores.set(Value::Entwicklung, -0.5);
        assert_eq!(scores.get(Value::Entwicklung), 0.0);
        scores.set(Value::Achtsamkeit, 0.75);
        assert_eq!(scores.get(Value::Achtsamkeit), 0.75);
    }

    #[test]
    fn average_works() {
        let mut scores = ValueScores::default();
        scores.set(Value::Achtsamkeit, 1.0);
        scores.set(Value::Anerkennung, 0.0);
        scores.set(Value::Aufmerksamkeit, 0.5);
        scores.set(Value::Entwicklung, 0.5);
        scores.set(Value::Sinn, 0.5);
        let avg = scores.average();
        assert!((avg - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_all_values_constant() {
        assert_eq!(Value::ALL.len(), 5);
    }

    #[test]
    fn test_value_names() {
        let mut scores = ValueScores::default();
        scores.set(Value::Achtsamkeit, 0.1);
        scores.set(Value::Anerkennung, 0.2);
        scores.set(Value::Aufmerksamkeit, 0.3);
        scores.set(Value::Entwicklung, 0.4);
        scores.set(Value::Sinn, 0.6);
        assert!((scores.get(Value::Achtsamkeit) - 0.1).abs() < f32::EPSILON);
        assert!((scores.get(Value::Anerkennung) - 0.2).abs() < f32::EPSILON);
        assert!((scores.get(Value::Aufmerksamkeit) - 0.3).abs() < f32::EPSILON);
        assert!((scores.get(Value::Entwicklung) - 0.4).abs() < f32::EPSILON);
        assert!((scores.get(Value::Sinn) - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_average_all_ones() {
        let mut scores = ValueScores::default();
        for value in Value::ALL {
            scores.set(value, 1.0);
        }
        assert!((scores.average() - 1.0).abs() < f32::EPSILON);
    }
}
