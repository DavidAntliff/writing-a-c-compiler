//! TACKY AST
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* body)
//!   instruction = Return(val) | Unary(unary_operator, val src, val dst)
//!   val = Constant(int) | Var(identifier)
//!   unary_operator = Complement | Negate
//!
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
            let dst_name = temporary.next();
            let dst = Val::Var(dst_name.to_string());
            let tacky_op = convert_unop(op);
            instructions.push(Instruction::Unary {
                op: tacky_op,
                src: src.clone(),
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
    fn test_program() {
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
}
