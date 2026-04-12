use std::collections::HashMap;

pub struct Embedding {
    table: HashMap<u16, Vec<f32>>,
}

impl Embedding {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn add_token(&mut self, id: u16, vector: Vec<f32>) {
        self.table.insert(id, vector);
    }

    pub fn forward(&self, id: u16) -> Option<Vec<f32>> {
        self.table.get(&id).cloned()
    }
}

#[cfg(test)]
#[path = "embedding_test.rs"]
mod tests;
