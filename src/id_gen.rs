pub(crate) struct IdGenerator {
    next: usize,
}

impl IdGenerator {
    pub(crate) fn new() -> Self {
        IdGenerator { next: 0 }
    }

    pub(crate) fn next(&mut self) -> usize {
        let v = self.next;
        self.next += 1;
        v
    }
}
