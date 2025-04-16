//! AST for the C language
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, block_item* body)
//!   block_item = S(statement) | D(declaration)
//!   declaration = Declaration(identifier name, exp? init)
//!   statement = Return(exp) | Expression(exp) | Null
//!   exp = Constant(int)
//!       | Var(identifier)
//!       | Unary(unary_operator, exp)
//!       | Binary(binary_operator, exp, exp)
//!       | Assignment(exp, exp)
//!   unary_operator = Complement | Negate | Not
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!                   | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight
//!                   | And | Or | Equal | NotEqual | LessThan | GreaterThan
//!                   | LessOrEqual | GreaterOrEqual
//!

use crate::lexer::Identifier;

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function: Function,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) body: Vec<BlockItem>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum BlockItem {
    S(Statement),
    D(Declaration),
}

#[derive(Debug, PartialEq)]
pub(crate) struct Declaration {
    pub(crate) name: Identifier,
    pub(crate) init: Option<Expression>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Statement {
    Return(Expression),
    Expression(Expression),
    Null,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Expression {
    Constant(usize),
    Var(Identifier),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Assignment(Box<Expression>, Box<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}
