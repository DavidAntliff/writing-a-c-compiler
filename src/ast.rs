use crate::lexer::Identifier;

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function: Function,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) body: Statement,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Statement {
    Return(Expression),
}

#[derive(Debug, PartialEq)]
pub(crate) enum Expression {
    Constant(usize),
}
