//! TACKY AST
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* body)
//!   instruction = Return(val)
//!               | Unary(unary_operator, val src, val dst)
//!               | Binary(binary_operator, val src1, val src2, val dst)
//!   val = Constant(int) | Var(identifier)
//!   unary_operator = Complement | Negate
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!

use crate::ast_c;
use thiserror::Error;

pub(crate) type Identifier = String;

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function_definition: FunctionDefinition,
}

#[derive(Debug, PartialEq)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: Identifier,
    pub(crate) body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Return(Val),
    Unary {
        op: UnaryOperator,
        src: Val,
        dst: Val,
    },
    Binary {
        op: BinaryOperator,
        src1: Val,
        src2: Val,
        dst: Val,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Val {
    Constant(usize),
    Var(Identifier),
}

#[derive(Debug, PartialEq)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq)]
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
}

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct TackyError {
    pub message: String,
}

pub(crate) fn parse(program: &ast_c::Program) -> Result<Program, TackyError> {
    let name = program.function.name.clone();

    let mut body = vec![];
    let mut temporary = TemporaryNamer::new();

    let val = emit_statement(&program.function.body, &mut body, &mut temporary);
    body.push(val);

    let function_definition = FunctionDefinition { name, body };

    Ok(Program {
        function_definition,
    })
}

struct TemporaryNamer {
    next: usize,
}

impl TemporaryNamer {
    fn new() -> Self {
        TemporaryNamer { next: 0 }
    }

    fn next(&mut self) -> String {
        let name = format!("tmp.{}", self.next);
        self.next += 1;
        name
    }
}

fn emit_tacky(
    exp: &ast_c::Expression,
    instructions: &mut Vec<Instruction>,
    temporary: &mut TemporaryNamer,
) -> Val {
    match exp {
        ast_c::Expression::Constant(c) => Val::Constant(*c),
        ast_c::Expression::Unary(op, inner) => {
            let src = emit_tacky(inner, instructions, temporary);
            let dst = Val::Var(temporary.next().to_string());
            let tacky_op = convert_unop(op);
            instructions.push(Instruction::Unary {
                op: tacky_op,
                src,
                dst: dst.clone(),
            });
            dst
        }
        ast_c::Expression::Binary(op, e1, e2) => {
            // Unsequenced - indeterminate order of evaluation
            let src1 = emit_tacky(e1, instructions, temporary);
            let src2 = emit_tacky(e2, instructions, temporary);
            let dst = Val::Var(temporary.next().to_string());
            let tacky = convert_binop(op);
            instructions.push(Instruction::Binary {
                op: tacky,
                src1,
                src2,
                dst: dst.clone(),
            });
            dst
        }
    }
}

fn convert_unop(op: &ast_c::UnaryOperator) -> UnaryOperator {
    match op {
        ast_c::UnaryOperator::Complement => UnaryOperator::Complement,
        ast_c::UnaryOperator::Negate => UnaryOperator::Negate,
    }
}

fn convert_binop(op: &ast_c::BinaryOperator) -> BinaryOperator {
    match op {
        ast_c::BinaryOperator::Add => BinaryOperator::Add,
        ast_c::BinaryOperator::Subtract => BinaryOperator::Subtract,
        ast_c::BinaryOperator::Multiply => BinaryOperator::Multiply,
        ast_c::BinaryOperator::Divide => BinaryOperator::Divide,
        ast_c::BinaryOperator::Remainder => BinaryOperator::Remainder,
        ast_c::BinaryOperator::BitAnd => BinaryOperator::BitAnd,
        ast_c::BinaryOperator::BitOr => BinaryOperator::BitOr,
        ast_c::BinaryOperator::BitXor => BinaryOperator::BitXor,
        ast_c::BinaryOperator::ShiftLeft => BinaryOperator::ShiftLeft,
        ast_c::BinaryOperator::ShiftRight => BinaryOperator::ShiftRight,
    }
}

fn emit_statement(
    statement: &ast_c::Statement,
    instructions: &mut Vec<Instruction>,
    temporary: &mut TemporaryNamer,
) -> Instruction {
    match statement {
        ast_c::Statement::Return(exp) => {
            let val = emit_tacky(exp, instructions, temporary);
            Instruction::Return(val)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::Expression;

    #[test]
    fn test_constant_expression() {
        let exp = ast_c::Expression::Constant(2);
        let mut instructions = vec![];
        let mut temporary = TemporaryNamer::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut temporary),
            Val::Constant(2)
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_unary_expression() {
        let exp = ast_c::Expression::Unary(
            ast_c::UnaryOperator::Complement,
            Box::new(ast_c::Expression::Constant(2)),
        );
        let mut instructions = vec![];
        let mut temporary = TemporaryNamer::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut temporary),
            Val::Var("tmp.0".into())
        );
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Complement,
                src: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            },]
        );
    }

    #[test]
    fn test_nested_unary_expression() {
        let exp = ast_c::Expression::Unary(
            ast_c::UnaryOperator::Negate,
            Box::new(ast_c::Expression::Unary(
                ast_c::UnaryOperator::Complement,
                Box::new(ast_c::Expression::Unary(
                    ast_c::UnaryOperator::Negate,
                    Box::new(ast_c::Expression::Constant(8)),
                )),
            )),
        );
        let mut instructions = vec![];
        let mut temporary = TemporaryNamer::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut temporary),
            Val::Var("tmp.2".into())
        );
        assert_eq!(
            instructions,
            vec![
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::Constant(8),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Unary {
                    op: UnaryOperator::Complement,
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
            ]
        );
    }

    #[test]
    fn test_statement_return_constant() {
        let statement = ast_c::Statement::Return(ast_c::Expression::Constant(2));
        let mut instructions = vec![];
        let mut temporary = TemporaryNamer::new();

        assert_eq!(
            emit_statement(&statement, &mut instructions, &mut temporary),
            Instruction::Return(Val::Constant(2))
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_statement_return_unary() {
        let statement = ast_c::Statement::Return(ast_c::Expression::Unary(
            ast_c::UnaryOperator::Negate,
            Box::new(ast_c::Expression::Constant(2)),
        ));
        let mut instructions = vec![];
        let mut temporary = TemporaryNamer::new();

        assert_eq!(
            emit_statement(&statement, &mut instructions, &mut temporary),
            Instruction::Return(Val::Var("tmp.0".into()))
        );
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Negate,
                src: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            }]
        );
    }

    #[test]
    fn test_program_return_unary_nested() {
        // int main(void) { return -(~(-8)); }
        let program = ast_c::Program {
            function: ast_c::Function {
                name: "main".to_string(),
                body: ast_c::Statement::Return(ast_c::Expression::Unary(
                    ast_c::UnaryOperator::Negate,
                    Box::new(ast_c::Expression::Unary(
                        ast_c::UnaryOperator::Complement,
                        Box::new(ast_c::Expression::Unary(
                            ast_c::UnaryOperator::Negate,
                            Box::new(ast_c::Expression::Constant(8)),
                        )),
                    )),
                )),
            },
        };

        assert_eq!(
            parse(&program).unwrap(),
            Program {
                function_definition: FunctionDefinition {
                    name: "main".to_string(),
                    body: vec![
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::Constant(8),
                            dst: Val::Var("tmp.0".into()),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Complement,
                            src: Val::Var("tmp.0".into()),
                            dst: Val::Var("tmp.1".into()),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::Var("tmp.1".into()),
                            dst: Val::Var("tmp.2".into()),
                        },
                        Instruction::Return(Val::Var("tmp.2".into())),
                    ],
                }
            }
        );
    }

    #[test]
    fn test_program_return_binary() {
        // int main(void) { return 1 * 2 - 3 * (4 + 5); }   // -25
        let program = ast_c::Program {
            function: ast_c::Function {
                name: "main".to_string(),
                body: ast_c::Statement::Return(Expression::Binary(
                    ast_c::BinaryOperator::Subtract,
                    Box::new(Expression::Binary(
                        ast_c::BinaryOperator::Multiply,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                    Box::new(Expression::Binary(
                        ast_c::BinaryOperator::Multiply,
                        Box::new(Expression::Constant(3)),
                        Box::new(Expression::Binary(
                            ast_c::BinaryOperator::Add,
                            Box::new(Expression::Constant(4)),
                            Box::new(Expression::Constant(5)),
                        )),
                    )),
                )),
            },
        };

        assert_eq!(
            parse(&program).unwrap(),
            Program {
                function_definition: FunctionDefinition {
                    name: "main".to_string(),
                    body: vec![
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::Constant(1),
                            src2: Val::Constant(2),
                            dst: Val::Var("tmp.0".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            src1: Val::Constant(4),
                            src2: Val::Constant(5),
                            dst: Val::Var("tmp.1".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::Constant(3),
                            src2: Val::Var("tmp.1".into()),
                            dst: Val::Var("tmp.2".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Subtract,
                            src1: Val::Var("tmp.0".into()),
                            src2: Val::Var("tmp.2".into()),
                            dst: Val::Var("tmp.3".into()),
                        },
                        Instruction::Return(Val::Var("tmp.3".into())),
                    ],
                }
            }
        );
    }
}
