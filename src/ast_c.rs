//! AST for the C language
//!
//! ASDL:
//!   program = Program(declaration*)
//!   declaration = FunDecl(function_declaration) | VarDecl(variable_declaration)
//!   variable_declaration = (identifier name, exp? init,
//!                           type var_type, storage_class?)
//!   function_declaration = (identifier name, identifier* params, block? body,
//!                           type fun_type, storage_class?)
//!   type = Int | Long | FunType(type* params, type ret)
//!   storage_class = Static | Extern
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
//!   exp = Constant(const)
//!       | Var(identifier)
//!       | Cast(type target_type, exp)
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
//!   const = ConstInt(int) | ConstLong(int)

use crate::lexer::Identifier;
use derive_more::Display;

pub(crate) type Label = String;

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Type {
    Int,
    Long,
    Fun { params: Vec<Type>, ret: Box<Type> },
}

#[derive(Debug, Display, PartialEq, Clone)]
pub enum StorageClass {
    Static,
    Extern,
}

mod ast_base {
    use super::{Label, StorageClass, Type};
    use crate::lexer::Identifier;

    /// A program: a sequence of declarations
    #[derive(Debug, Clone, PartialEq)]
    pub struct Program<E> {
        pub declarations: Vec<Declaration<E>>,
    }

    /// Top-level declaration: function or variable
    #[derive(Debug, Clone, PartialEq)]
    pub enum Declaration<E> {
        FunDecl(FunDecl<E>),
        VarDecl(VarDecl<E>),
    }

    /// Variable declaration with optional initializer `E`
    #[derive(Debug, Clone, PartialEq)]
    pub struct VarDecl<E> {
        pub name: Identifier,
        pub init: Option<E>,
        pub var_type: Type,
        pub storage_class: Option<StorageClass>,
    }

    /// Function declaration, with optional body of `Block<E>`
    #[derive(Debug, Clone, PartialEq)]
    pub struct FunDecl<E> {
        pub name: Identifier,
        pub params: Vec<Identifier>,
        pub body: Option<Block<E>>,
        pub fun_type: Type,
        pub storage_class: Option<StorageClass>,
    }

    /// `for`-initializer: either a `VarDecl<E>` or `Option<E>`
    #[derive(Debug, Clone, PartialEq)]
    pub enum ForInit<E> {
        InitDecl(VarDecl<E>),
        InitExp(Option<E>),
    }

    /// Statement parameterized by expression type `E`
    #[derive(Debug, Clone, PartialEq)]
    pub enum Statement<E> {
        Return(E),
        Expression(E),
        If {
            condition: E,
            then_block: Box<Statement<E>>,
            else_block: Option<Box<Statement<E>>>,
        },
        Labeled {
            label: Label,
            statement: Box<Statement<E>>,
        },
        Goto(Label),
        Compound(Block<E>),
        Break(Option<Label>),
        Continue(Option<Label>),
        While {
            condition: E,
            body: Box<Statement<E>>,
            loop_label: Option<Label>,
        },
        DoWhile {
            body: Box<Statement<E>>,
            condition: E,
            loop_label: Option<Label>,
        },
        For {
            init: ForInit<E>,
            condition: Option<E>,
            post: Option<E>,
            body: Box<Statement<E>>,
            loop_label: Option<Label>,
        },
        Null,
    }

    /// A block item: statement or declaration
    #[derive(Debug, Clone, PartialEq)]
    pub enum BlockItem<E> {
        S(Statement<E>),
        D(Declaration<E>),
    }

    /// A block: a sequence of block items
    #[derive(Debug, Clone, PartialEq)]
    pub struct Block<E> {
        pub items: Vec<BlockItem<E>>,
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum TypedExpression {
    Constant(Const),
    Var(Identifier),
    Cast(Type, Box<TypedExpression>),
    Unary(UnaryOperator, Box<TypedExpression>),
    Binary(BinaryOperator, Box<TypedExpression>, Box<TypedExpression>),
    Assignment(Box<TypedExpression>, Box<TypedExpression>),
    Conditional(
        Box<TypedExpression>,
        Box<TypedExpression>,
        Box<TypedExpression>,
    ),
    FunctionCall(Identifier, Vec<TypedExpression>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Expression {
    Constant(Const),
    Var(Identifier),
    Cast(Type, Box<Expression>),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Assignment(Box<Expression>, Box<Expression>),
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
    FunctionCall(Identifier, Vec<Expression>),
}

// === Untyped AST aliases ===
pub type Program = ast_base::Program<Expression>;
pub type Declaration = ast_base::Declaration<Expression>;
pub type VarDecl = ast_base::VarDecl<Expression>;
pub type FunDecl = ast_base::FunDecl<Expression>;
pub type ForInit = ast_base::ForInit<Expression>;
pub type Statement = ast_base::Statement<Expression>;
pub type BlockItem = ast_base::BlockItem<Expression>;
pub type Block = ast_base::Block<Expression>;

// === Typed AST aliases ===
pub type TypedProgram = ast_base::Program<TypedExpression>;
pub type TypedDeclaration = ast_base::Declaration<TypedExpression>;
pub type TypedVarDecl = ast_base::VarDecl<TypedExpression>;
pub type TypedFunDecl = ast_base::FunDecl<TypedExpression>;
pub type TypedForInit = ast_base::ForInit<TypedExpression>;
pub type TypedStatement = ast_base::Statement<TypedExpression>;
pub type TypedBlockItem = ast_base::BlockItem<TypedExpression>;
pub type TypedBlock = ast_base::Block<TypedExpression>;

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

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Const {
    ConstInt(i32),
    ConstLong(i64),
}
