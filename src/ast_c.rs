//! AST for the C language
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, statement body)
//!   statement = Return(exp)
//!   exp = Constant(int)
//!       | Unary(unary_operator, exp)
//!       | Binary(binary_operator, exp, exp)
//!   unary_operator = Complement | Negate
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!

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

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Expression {
    Constant(usize),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}
