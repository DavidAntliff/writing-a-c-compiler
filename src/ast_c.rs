//! AST for the C language
//!
//! ASDL:
//!   program = Program(function_declaration*)
//!   declaration = FunDecl(function_declaration) | VarDecl(variable_declaration)
//!   variable_declaration = (identifier name, exp? init)
//!   function_declaration = (identifier name, identifier* params, block? body)
//!   block_item = S(statement) | D(declaration)
//!   block = Block(block_item*)
//!   for_init = InitDecl(variable_declaration) | InitExp(exp?)
//!   statement = Labeled(identifier label, statement)
//!             | Return(exp)
//!             | Expression(exp)
//!             | If(exp condition, statement then, statement? else)
//!             | Goto(identifier label)
//!             | Compound(block)
//!             | Break(identifier? loop_label)
//!             | Continue(identifier? loop_label)
//!             | While(exp condition, statement body, identifier? loop_label)
//!             | DoWhile(statement body, exp condition, identifier? loop_label)
//!             | For(for_init init, exp? condition, exp? post, statement body, identifier? loop_label)
//!             | Null
//!   exp = Constant(int)
//!       | Var(identifier)
//!       | Unary(unary_operator, exp)
//!       | Binary(binary_operator, exp, exp)
//!       | Assignment(exp, exp)
//!       | Conditional(exp condition, exp, exp)
//!       | FunctionCall(identifier, exp* args)
//!   unary_operator = Complement | Negate | Not
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!                   | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight
//!                   | And | Or | Equal | NotEqual | LessThan | GreaterThan
//!                   | LessOrEqual | GreaterOrEqual

use crate::lexer::Identifier;

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function_declarations: Vec<FunDecl>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct FunDecl {
    pub(crate) name: Identifier,
    pub(crate) params: Vec<Identifier>,
    pub(crate) body: Option<Block>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Block {
    pub(crate) items: Vec<BlockItem>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum BlockItem {
    S(Statement),
    D(Declaration),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Declaration {
    FunDecl(FunDecl),
    VarDecl(VarDecl),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct VarDecl {
    pub(crate) name: Identifier,
    pub(crate) init: Option<Expression>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Statement {
    Return(Expression),
    Expression(Expression),
    If {
        condition: Expression,
        then: Box<Statement>,
        else_: Option<Box<Statement>>,
    },
    Labeled {
        label: Identifier,
        statement: Box<Statement>,
    },
    Goto(Identifier),
    Compound(Block),
    Break(Option<Identifier>),
    Continue(Option<Identifier>),
    While {
        condition: Expression,
        body: Box<Statement>,
        loop_label: Option<Identifier>,
    },
    DoWhile {
        body: Box<Statement>,
        condition: Expression,
        loop_label: Option<Identifier>,
    },
    For {
        init: ForInit,
        condition: Option<Expression>,
        post: Option<Expression>,
        body: Box<Statement>,
        loop_label: Option<Identifier>,
    },
    Null,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ForInit {
    InitDecl(VarDecl),
    InitExp(Option<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Expression {
    Constant(usize),
    Var(Identifier),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Assignment(Box<Expression>, Box<Expression>),
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
    FunctionCall(Identifier, Vec<Expression>),
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
