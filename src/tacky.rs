//! TACKY AST
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* body)
//!   instruction = Return(val)
//!               | Unary(unary_operator, val src, val dst)
//!               | Binary(binary_operator, val src1, val src2, val dst)
//!               | Copy(val src, val dst)
//!               | Jump(identifier target)
//!               | JumpIfZero(val condition, identifier target)
//!               | JumpIfNotZero(val condition, identifier target)
//!               | Label(identifier)
//!   val = Constant(int) | Var(identifier)
//!   unary_operator = Complement | Negate | Not
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!                   | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight
//!                   | Equal | NotEqual | LessThan | LessOrEqual | GreaterThan | GreaterOrEqual
//!

use crate::ast_c;
use thiserror::Error;

//pub(crate) type Identifier = String;
#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub(crate) struct Identifier(pub(crate) String);

impl<T> From<T> for Identifier
where
    T: Into<String>,
{
    fn from(value: T) -> Self {
        Identifier(value.into())
    }
}

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
    Copy {
        src: Val,
        dst: Val,
    },
    Jump {
        target: Identifier,
    },
    JumpIfZero {
        condition: Val,
        target: Identifier,
    },
    JumpIfNotZero {
        condition: Val,
        target: Identifier,
    },
    Label(Identifier),
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
    Not,
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
    //And,
    //Or,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct TackyError {
    pub message: String,
}

pub(crate) fn parse(program: &ast_c::Program) -> Result<Program, TackyError> {
    let name: Identifier = (&program.function.name).into();

    let mut body = vec![];
    let mut id_gen = IdGenerator::new();

    let val = emit_statement(&program.function.body, &mut body, &mut id_gen);
    body.push(val);

    let function_definition = FunctionDefinition { name, body };

    Ok(Program {
        function_definition,
    })
}

struct IdGenerator {
    next: usize,
}

impl IdGenerator {
    fn new() -> Self {
        IdGenerator { next: 0 }
    }

    fn next(&mut self) -> usize {
        let v = self.next;
        self.next += 1;
        v
    }
}

fn next_var(id_gen: &mut IdGenerator) -> Identifier {
    format!("tmp.{}", id_gen.next()).into()
}

fn emit_tacky(
    exp: &ast_c::Expression,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Val {
    match exp {
        ast_c::Expression::Constant(c) => Val::Constant(*c),
        ast_c::Expression::Unary(op, inner) => {
            let src = emit_tacky(inner, instructions, id_gen);
            let dst = Val::Var(next_var(id_gen));
            let tacky_op = convert_unop(op);
            instructions.push(Instruction::Unary {
                op: tacky_op,
                src,
                dst: dst.clone(),
            });
            dst
        }

        // Handle short-circuit evaluation for And / Or
        ast_c::Expression::Binary(ast_c::BinaryOperator::And, e1, e2) => {
            let id = id_gen.next();
            let label_false: Identifier = format!("and_false.{id}").into();
            let label_end: Identifier = format!("and_end.{id}").into();

            let v1 = emit_tacky(e1, instructions, id_gen);
            instructions.push(Instruction::JumpIfZero {
                condition: v1.clone(),
                target: label_false.clone(),
            });
            let v2 = emit_tacky(e2, instructions, id_gen);
            instructions.push(Instruction::JumpIfZero {
                condition: v2.clone(),
                target: label_false.clone(),
            });
            let dst = Val::Var(next_var(id_gen));
            instructions.push(Instruction::Copy {
                src: Val::Constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_false));
            instructions.push(Instruction::Copy {
                src: Val::Constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::Expression::Binary(ast_c::BinaryOperator::Or, e1, e2) => {
            let id = id_gen.next();
            let label_true: Identifier = format!("or_true.{id}").into();
            let label_end: Identifier = format!("or_end.{id}").into();

            let v1 = emit_tacky(e1, instructions, id_gen);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v1.clone(),
                target: label_true.clone(),
            });
            let v2 = emit_tacky(e2, instructions, id_gen);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v2.clone(),
                target: label_true.clone(),
            });
            let dst = Val::Var(next_var(id_gen));
            instructions.push(Instruction::Copy {
                src: Val::Constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_true));
            instructions.push(Instruction::Copy {
                src: Val::Constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::Expression::Binary(op, e1, e2) => {
            // Unsequenced - indeterminate order of evaluation
            let src1 = emit_tacky(e1, instructions, id_gen);
            let src2 = emit_tacky(e2, instructions, id_gen);
            let dst = Val::Var(next_var(id_gen));
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
        ast_c::UnaryOperator::Not => UnaryOperator::Not,
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
        ast_c::BinaryOperator::Equal => BinaryOperator::Equal,
        ast_c::BinaryOperator::NotEqual => BinaryOperator::NotEqual,
        ast_c::BinaryOperator::LessThan => BinaryOperator::LessThan,
        ast_c::BinaryOperator::GreaterThan => BinaryOperator::GreaterThan,
        ast_c::BinaryOperator::LessOrEqual => BinaryOperator::LessOrEqual,
        ast_c::BinaryOperator::GreaterOrEqual => BinaryOperator::GreaterOrEqual,
        _ => {
            panic!("Unsupported binary operator: {:?}", op);
        }
    }
}

fn emit_statement(
    statement: &ast_c::Statement,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Instruction {
    match statement {
        ast_c::Statement::Return(exp) => {
            let val = emit_tacky(exp, instructions, id_gen);
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
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut id_gen),
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
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut id_gen),
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
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_tacky(&exp, &mut instructions, &mut id_gen),
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
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_statement(&statement, &mut instructions, &mut id_gen),
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
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_statement(&statement, &mut instructions, &mut id_gen),
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
                name: "main".into(),
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
                    name: "main".into(),
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
                name: "main".into(),
                body: ast_c::Statement::Return(ast_c::Expression::Binary(
                    ast_c::BinaryOperator::Subtract,
                    Box::new(ast_c::Expression::Binary(
                        ast_c::BinaryOperator::Multiply,
                        Box::new(ast_c::Expression::Constant(1)),
                        Box::new(ast_c::Expression::Constant(2)),
                    )),
                    Box::new(ast_c::Expression::Binary(
                        ast_c::BinaryOperator::Multiply,
                        Box::new(ast_c::Expression::Constant(3)),
                        Box::new(ast_c::Expression::Binary(
                            ast_c::BinaryOperator::Add,
                            Box::new(ast_c::Expression::Constant(4)),
                            Box::new(ast_c::Expression::Constant(5)),
                        )),
                    )),
                )),
            },
        };

        assert_eq!(
            parse(&program).unwrap(),
            Program {
                function_definition: FunctionDefinition {
                    name: "main".into(),
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

    fn do_emit_tacky(exp: &ast_c::Expression) -> (Val, Vec<Instruction>) {
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let val = emit_tacky(exp, &mut instructions, &mut id_gen);
        (val, instructions)
    }

    #[test]
    fn test_emit_tacky_unary_not() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Unary(
            ast_c::UnaryOperator::Not,
            Box::new(ast_c::Expression::Constant(1)),
        ));

        assert_eq!(val, Val::Var("tmp.0".into()));
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Not,
                src: Val::Constant(1),
                dst: Val::Var("tmp.0".into()),
            },]
        );
    }

    #[test]
    fn test_emit_tacky_binary_and() {
        // "e1 && e2" generates:
        //   <instructions for e1>
        //   v1 = <result of e1>
        //   JumpIfZero(v1, false_label)
        //   <instructions for e2>
        //   v2 = <result of e2>
        //   JumpIfZero(v2, false_label)
        //   Copy(1, result)
        //   Jump(end)
        //   Label(false_label)
        //   Copy(0, result)
        //   Label(end)
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::And,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.2".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()), // v1
                },
                Instruction::JumpIfZero {
                    condition: Val::Var("tmp.1".into()),
                    target: "and_false.0".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::Constant(3), // v2
                    target: "and_false.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Jump {
                    target: "and_end.0".into(),
                },
                Instruction::Label("and_false.0".into()),
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Label("and_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_or() {
        // "e1 || e2" generates:
        //   <instructions for e1>
        //   v1 = <result of e1>
        //   JumpIfNotZero(v1, true_label)
        //   <instructions for e2>
        //   v2 = <result of e2>
        //   JumpIfNotZero(v2, true_label)
        //   Copy(0, result)
        //   Jump(end)
        //   Label(true_label)
        //   Copy(1, result)
        //   Label(end)
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Or,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.2".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()), // v1
                },
                Instruction::JumpIfNotZero {
                    condition: Val::Var("tmp.1".into()),
                    target: "or_true.0".into(),
                },
                Instruction::JumpIfNotZero {
                    condition: Val::Constant(3), // v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Label("or_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_or_and() {
        //   e1 || e2 && e3
        // Equivalent to:
        //   e1 || (e2 && e3)
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Or,
            Box::new(ast_c::Expression::Constant(1)),
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::And,
                Box::new(ast_c::Expression::Constant(2)),
                Box::new(ast_c::Expression::Constant(3)),
            )),
        ));

        assert_eq!(val, Val::Var("tmp.3".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::JumpIfNotZero {
                    condition: Val::Constant(1), // OR v1
                    target: "or_true.0".into(),
                },
                // AND
                Instruction::JumpIfZero {
                    condition: Val::Constant(2), // AND v1
                    target: "and_false.1".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::Constant(3), // AND v2
                    target: "and_false.1".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // AND result
                },
                Instruction::Jump {
                    target: "and_end.1".into(),
                },
                Instruction::Label("and_false.1".into()),
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // AND result
                },
                Instruction::Label("and_end.1".into()),
                // back to OR
                Instruction::JumpIfNotZero {
                    condition: Val::Var("tmp.2".into()), // OR v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.3".into()), // final result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.3".into()), // final result
                },
                Instruction::Label("or_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Equal,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::Equal,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_not_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::NotEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::NotEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::LessThan,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessThan,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::GreaterThan,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterThan,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::LessOrEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessOrEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::GreaterOrEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterOrEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }
}
