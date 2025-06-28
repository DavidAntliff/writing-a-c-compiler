use crate::ast_asm as asm;
use crate::tacky;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct CodegenError {
    pub message: String,
}

// Register used to fix source and destination operands
const SRC_SCRATCH_REGISTER: asm::Reg = asm::Reg::R10;
const DST_SCRATCH_REGISTER: asm::Reg = asm::Reg::R11;

pub(crate) fn parse(tacky: &tacky::Program) -> Result<asm::Program, CodegenError> {
    // Three passes

    // Pass 1 - convert TACKY AST to ASM AST, using pseudo registers for variables
    let ast = pass1(tacky);
    log::debug!("Codegen PASS 1: {ast:#?}");

    // Pass 2 - replace pseudo registers with Stack operands
    let (ast, stack_size) = pass2(ast)?;
    log::debug!("Codegen PASS 2: {ast:#?}");

    // Pass 3:
    //   1. Insert AllocateStack instruction at the beginning of function_definition,
    //   2. Rewrite invalid MOV instructions.
    let ast = pass3(&ast, stack_size)?;
    log::debug!("Codegen PASS 3: {ast:#?}");

    Ok(ast)
}

fn pass1(program: &tacky::Program) -> asm::Program {
    // TODO: Handle multiple function definitions
    asm::Program {
        function_definition: function_definition(&program.function_definitions[0]),
    }
}

impl From<&tacky::Val> for asm::Operand {
    fn from(val: &tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(c) => asm::Operand::Imm(*c),
            tacky::Val::Var(identifier) => {
                asm::Operand::Pseudo(asm::Identifier(identifier.0.clone()))
            }
        }
    }
}

impl From<tacky::Identifier> for asm::Identifier {
    fn from(identifier: tacky::Identifier) -> Self {
        asm::Identifier(identifier.0)
    }
}

impl From<&tacky::Identifier> for asm::Identifier {
    fn from(identifier: &tacky::Identifier) -> Self {
        asm::Identifier(identifier.0.clone())
    }
}

fn function_definition(function: &tacky::FunctionDefinition) -> asm::Function {
    let mut instructions = vec![];

    function
        .body
        .iter()
        .for_each(|instruction| gen_instruction(instruction, &mut instructions));

    asm::Function {
        name: (&function.name).into(),
        instructions,
    }
}

fn gen_instruction(instruction: &tacky::Instruction, instructions: &mut Vec<asm::Instruction>) {
    match instruction {
        tacky::Instruction::Return(val) => {
            instructions.push(asm::Instruction::Mov {
                src: asm::Operand::from(val),
                dst: asm::Operand::Reg(asm::Reg::AX),
            });
            instructions.push(asm::Instruction::Ret);
        }
        tacky::Instruction::Unary {
            op: tacky::UnaryOperator::Not,
            src,
            dst,
        } => {
            gen_not(src.into(), dst.into(), instructions);
        }
        tacky::Instruction::Unary { op, src, dst } => {
            let src = asm::Operand::from(src);
            let dst = asm::Operand::from(dst);
            let op = match op {
                tacky::UnaryOperator::Complement => asm::UnaryOperator::Not,
                tacky::UnaryOperator::Negate => asm::UnaryOperator::Neg,
                _ => {
                    panic!("Unsupported unary operator: {op:?}");
                }
            };
            instructions.push(asm::Instruction::Mov {
                src,
                dst: dst.clone(),
            });
            instructions.push(asm::Instruction::Unary { op, dst });
        }
        tacky::Instruction::Binary {
            op,
            src1,
            src2,
            dst,
        } => {
            gen_binary_operator(op, src1.into(), src2.into(), dst.into(), instructions);
        }
        tacky::Instruction::Copy { src, dst } => {
            instructions.push(asm::Instruction::Mov {
                src: src.into(),
                dst: dst.into(),
            });
        }
        tacky::Instruction::Jump { target } => {
            instructions.push(asm::Instruction::Jmp {
                target: target.into(),
            });
        }
        tacky::Instruction::JumpIfZero { condition, target } => {
            instructions.push(asm::Instruction::Cmp {
                src1: asm::Operand::Imm(0),
                src2: condition.into(),
            });
            instructions.push(asm::Instruction::JmpCC {
                cc: asm::ConditionCode::Equal,
                target: target.into(),
            });
        }
        tacky::Instruction::JumpIfNotZero { condition, target } => {
            instructions.push(asm::Instruction::Cmp {
                src1: asm::Operand::Imm(0),
                src2: condition.into(),
            });
            instructions.push(asm::Instruction::JmpCC {
                cc: asm::ConditionCode::NotEqual,
                target: target.into(),
            });
        }
        tacky::Instruction::Label(label) => {
            instructions.push(asm::Instruction::Label(label.into()));
        }
        #[allow(unreachable_patterns)]
        _ => panic!("Unsupported instruction: {instruction:?}"),
    }
}

fn gen_binary_operator(
    op: &tacky::BinaryOperator,
    src1: asm::Operand,
    src2: asm::Operand,
    dst: asm::Operand,
    instructions: &mut Vec<asm::Instruction>,
) {
    match op {
        tacky::BinaryOperator::Add
        | tacky::BinaryOperator::Subtract
        | tacky::BinaryOperator::Multiply
        | tacky::BinaryOperator::BitAnd
        | tacky::BinaryOperator::BitOr
        | tacky::BinaryOperator::BitXor
        | tacky::BinaryOperator::ShiftLeft
        | tacky::BinaryOperator::ShiftRight => {
            let asm_op = match op {
                tacky::BinaryOperator::Add => asm::BinaryOperator::Add,
                tacky::BinaryOperator::Subtract => asm::BinaryOperator::Sub,
                tacky::BinaryOperator::Multiply => asm::BinaryOperator::Mult,
                tacky::BinaryOperator::BitAnd => asm::BinaryOperator::BitAnd,
                tacky::BinaryOperator::BitOr => asm::BinaryOperator::BitOr,
                tacky::BinaryOperator::BitXor => asm::BinaryOperator::BitXor,
                tacky::BinaryOperator::ShiftLeft => asm::BinaryOperator::BitShiftLeft,
                tacky::BinaryOperator::ShiftRight => asm::BinaryOperator::BitShiftRight,
                _ => unreachable!(), // guaranteed by outer match
            };
            instructions.push(asm::Instruction::Mov {
                src: src1.clone(),
                dst: dst.clone(),
            });
            instructions.push(asm::Instruction::Binary {
                op: asm_op,
                src: src2,
                dst,
            });
        }
        tacky::BinaryOperator::Equal
        | tacky::BinaryOperator::NotEqual
        | tacky::BinaryOperator::LessThan
        | tacky::BinaryOperator::LessOrEqual
        | tacky::BinaryOperator::GreaterThan
        | tacky::BinaryOperator::GreaterOrEqual => {
            let cc = match op {
                tacky::BinaryOperator::Equal => asm::ConditionCode::Equal,
                tacky::BinaryOperator::NotEqual => asm::ConditionCode::NotEqual,
                tacky::BinaryOperator::LessThan => asm::ConditionCode::LessThan,
                tacky::BinaryOperator::LessOrEqual => asm::ConditionCode::LessOrEqual,
                tacky::BinaryOperator::GreaterThan => asm::ConditionCode::GreaterThan,
                tacky::BinaryOperator::GreaterOrEqual => asm::ConditionCode::GreaterOrEqual,
                _ => unreachable!(), // guaranteed by outer match
            };
            instructions.push(asm::Instruction::Cmp {
                src1: src2.clone(),
                src2: src1.clone(),
            });
            instructions.push(asm::Instruction::Mov {
                src: asm::Operand::Imm(0),
                dst: dst.clone(),
            });
            instructions.push(asm::Instruction::SetCC { cc, dst });
        }
        tacky::BinaryOperator::Divide | tacky::BinaryOperator::Remainder => {
            instructions.push(asm::Instruction::Mov {
                src: src1.clone(),
                dst: asm::Operand::Reg(asm::Reg::AX),
            });
            instructions.push(asm::Instruction::Cdq);
            instructions.push(asm::Instruction::Idiv(src2));
            let result_reg = match op {
                tacky::BinaryOperator::Divide => asm::Reg::AX,
                tacky::BinaryOperator::Remainder => asm::Reg::DX,
                _ => unreachable!(),
            };
            instructions.push(asm::Instruction::Mov {
                src: asm::Operand::Reg(result_reg),
                dst,
            });
        }
        #[allow(unreachable_patterns)]
        _ => unimplemented!("unexpected operator {:?}", op),
    }
}

fn gen_not(src: asm::Operand, dst: asm::Operand, instructions: &mut Vec<asm::Instruction>) {
    instructions.push(asm::Instruction::Cmp {
        src1: asm::Operand::Imm(0),
        src2: src,
    });
    // Clear the destination register
    instructions.push(asm::Instruction::Mov {
        src: asm::Operand::Imm(0),
        dst: dst.clone(),
    });
    instructions.push(asm::Instruction::SetCC {
        cc: asm::ConditionCode::Equal,
        dst,
    });
}

struct PseudoReplacer {
    map: HashMap<asm::Identifier, asm::Offset>,
    size: usize,
}

impl PseudoReplacer {
    fn new() -> Self {
        PseudoReplacer {
            map: HashMap::new(),
            size: 0,
        }
    }

    fn replace(&mut self, operand: &mut asm::Operand) {
        if let asm::Operand::Pseudo(identifier) = operand {
            if let Some(offset) = self.map.get(identifier) {
                *operand = asm::Operand::Stack(*offset);
            } else {
                self.size += asm::STACK_SLOT_SIZE;
                let offset = asm::Offset(-(self.size as isize));
                self.map.insert(identifier.clone(), offset);
                *operand = asm::Operand::Stack(offset);
            }
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// Replace each Pseudo operand with a Stack operand
/// and return the final size of the stack frame.
fn pass2(mut ast: asm::Program) -> Result<(asm::Program, usize), CodegenError> {
    let mut replacer = PseudoReplacer::new();

    ast.function_definition
        .instructions
        .iter_mut()
        .for_each(|instruction| match instruction {
            asm::Instruction::Mov { src, dst } => {
                replacer.replace(src);
                replacer.replace(dst);
            }
            asm::Instruction::Unary { op: _, dst } => {
                replacer.replace(dst);
            }
            asm::Instruction::Binary { op: _, src, dst } => {
                replacer.replace(src);
                replacer.replace(dst);
            }
            asm::Instruction::Idiv(src) => {
                replacer.replace(src);
            }
            asm::Instruction::Cmp { src1, src2 } => {
                replacer.replace(src1);
                replacer.replace(src2);
            }
            asm::Instruction::SetCC { cc: _, dst } => {
                replacer.replace(dst);
            }

            // No action required for the remaining instructions:
            asm::Instruction::Cdq
            | asm::Instruction::Jmp { .. }
            | asm::Instruction::JmpCC { .. }
            | asm::Instruction::Label(_)
            | asm::Instruction::AllocateStack(_)
            | asm::Instruction::Ret => {}
        });

    Ok((ast, replacer.size()))
}

fn pass3(ast: &asm::Program, stack_size: usize) -> Result<asm::Program, CodegenError> {
    let mut instructions = Vec::with_capacity(ast.function_definition.instructions.len() + 1);

    // Insert AllocateStack instruction at the beginning
    instructions.push(asm::Instruction::AllocateStack(stack_size));

    // Rewrite invalid MOV instructions to pass through the scratch register
    for instruction in &ast.function_definition.instructions {
        match instruction {
            // MOV instructions with two address operands are not allowed
            //   -> copy through scratch register
            asm::Instruction::Mov {
                src: asm::Operand::Stack(src),
                dst: asm::Operand::Stack(dst),
            } => {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Stack(*src),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(*dst),
                });
            }

            // IDIV instructions with a constant operand are not allowed
            //   -> copy to scratch register first
            asm::Instruction::Idiv(src @ asm::Operand::Imm(_)) => {
                instructions.push(asm::Instruction::Mov {
                    src: src.clone(),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Idiv(asm::Operand::Reg(
                    SRC_SCRATCH_REGISTER,
                )));
            }

            // ADD/SUB instructions with two address operands are not allowed
            //   -> copy through scratch register
            // Includes binary bitwise operators.
            asm::Instruction::Binary {
                op,
                src: asm::Operand::Stack(src),
                dst: asm::Operand::Stack(dst),
            } if *op == asm::BinaryOperator::Add
                || *op == asm::BinaryOperator::Sub
                || *op == asm::BinaryOperator::BitAnd
                || *op == asm::BinaryOperator::BitOr
                || *op == asm::BinaryOperator::BitXor =>
            {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Stack(*src),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Binary {
                    op: op.clone(),
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(*dst),
                });
            }

            // IMUL instructions with a memory address as destination are not allowed
            //   -> copy destination to scratch register first
            asm::Instruction::Binary {
                op,
                src,
                dst: asm::Operand::Stack(dst),
            } if *op == asm::BinaryOperator::Mult => {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Stack(*dst),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Binary {
                    op: op.clone(),
                    src: src.clone(),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(*dst),
                });
            }

            // SAL/SAR instructions with a memory address as src are not allowed
            //   -> copy source to CX register first
            asm::Instruction::Binary {
                op,
                src: asm::Operand::Stack(src),
                dst,
            } if *op == asm::BinaryOperator::BitShiftLeft
                || *op == asm::BinaryOperator::BitShiftRight =>
            {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Stack(*src),
                    dst: asm::Operand::Reg(asm::Reg::CX),
                });
                instructions.push(asm::Instruction::Binary {
                    op: op.clone(),
                    src: asm::Operand::Reg(asm::Reg::CX),
                    dst: dst.clone(),
                });
            }

            // CMP instructions with two address operands are not allowed
            //   -> copy through scratch register
            asm::Instruction::Cmp {
                src1: src1 @ asm::Operand::Stack(_),
                src2: src2 @ asm::Operand::Stack(_),
            } => {
                instructions.push(asm::Instruction::Mov {
                    src: src1.clone(),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Cmp {
                    src1: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    src2: src2.clone(),
                });
            }

            // CMP instructions with a constant second operand are not allowed
            //   -> copy through scratch register
            asm::Instruction::Cmp {
                src1,
                src2: src2 @ asm::Operand::Imm(_),
            } => {
                instructions.push(asm::Instruction::Mov {
                    src: src2.clone(),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Cmp {
                    src1: src1.clone(),
                    src2: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                });
            }

            _ => instructions.push(instruction.clone()),
        }
    }

    Ok(asm::Program {
        function_definition: asm::Function {
            name: ast.function_definition.name.clone(),
            instructions,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tacky::{BinaryOperator, Identifier, Instruction, UnaryOperator, Val};

    #[test]
    fn gen_unary_not() {
        let mut instructions = vec![];
        gen_instruction(
            &Instruction::Unary {
                op: UnaryOperator::Not,
                src: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![
                asm::Instruction::Cmp {
                    src1: asm::Operand::Imm(0),
                    src2: asm::Operand::Imm(2),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(0),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
                asm::Instruction::SetCC {
                    cc: asm::ConditionCode::Equal,
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                }
            ]
        )
    }

    #[test]
    fn test_gen_binary_comparison() {
        let mut instructions = vec![];
        let tmp0 = tacky::Identifier("tmp.0".into());
        let tmp1 = tacky::Identifier("tmp.1".into());
        gen_instruction(
            &Instruction::Binary {
                op: BinaryOperator::LessOrEqual,
                src1: Val::Constant(2),
                src2: Val::Var(tmp0.clone()),
                dst: Val::Var(tmp1.clone()),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![
                asm::Instruction::Cmp {
                    src1: asm::Operand::Pseudo((&tmp0).into()),
                    src2: asm::Operand::Imm(2),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(0),
                    dst: asm::Operand::Pseudo((&tmp1).into()),
                },
                asm::Instruction::SetCC {
                    cc: asm::ConditionCode::LessOrEqual,
                    dst: asm::Operand::Pseudo(tmp1.into()),
                }
            ]
        )
    }

    #[test]
    fn test_gen_jump() {
        let mut instructions = vec![];
        let target = Identifier("label".into());
        gen_instruction(
            &Instruction::Jump {
                target: target.clone(),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![asm::Instruction::Jmp {
                target: target.into(),
            }]
        )
    }

    #[test]
    fn test_gen_jump_if_zero() {
        let mut instructions = vec![];
        let condition = Identifier("tmp.0".into());
        let target = Identifier("label".into());
        gen_instruction(
            &Instruction::JumpIfZero {
                condition: Val::Var(condition.clone()),
                target: target.clone(),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![
                asm::Instruction::Cmp {
                    src1: asm::Operand::Imm(0),
                    src2: asm::Operand::Pseudo(condition.into()),
                },
                asm::Instruction::JmpCC {
                    cc: asm::ConditionCode::Equal,
                    target: target.into(),
                }
            ]
        )
    }

    #[test]
    fn test_gen_jump_if_not_zero() {
        let mut instructions = vec![];
        let condition = Identifier("tmp.0".into());
        let target = Identifier("label".into());
        gen_instruction(
            &Instruction::JumpIfNotZero {
                condition: Val::Var(condition.clone()),
                target: target.clone(),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![
                asm::Instruction::Cmp {
                    src1: asm::Operand::Imm(0),
                    src2: asm::Operand::Pseudo(condition.into()),
                },
                asm::Instruction::JmpCC {
                    cc: asm::ConditionCode::NotEqual,
                    target: target.into(),
                }
            ]
        )
    }

    #[test]
    fn test_gen_copy() {
        let mut instructions = vec![];
        gen_instruction(
            &Instruction::Copy {
                src: Val::Var("tmp.0".into()),
                dst: Val::Var("tmp.1".into()),
            },
            &mut instructions,
        );

        assert_eq!(
            instructions,
            vec![asm::Instruction::Mov {
                src: asm::Operand::Pseudo("tmp.0".into()),
                dst: asm::Operand::Pseudo("tmp.1".into()),
            },]
        )
    }

    #[test]
    fn test_gen_label() {
        let mut instructions = vec![];
        gen_instruction(&Instruction::Label("label".into()), &mut instructions);

        assert_eq!(instructions, vec![asm::Instruction::Label("label".into())])
    }

    #[test]
    fn test_pass1_return_constant() {
        let tacky_program = tacky::Program {
            function_definitions: vec![tacky::FunctionDefinition {
                name: "main".into(),
                params: vec![],
                body: vec![Instruction::Return(Val::Constant(2))],
            }],
        };

        let ast = pass1(&tacky_program);

        assert_eq!(ast.function_definition.name, "main".into());
        assert_eq!(
            ast.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Ret,
            ]
        );
    }

    #[test]
    fn test_pass1_return_unary() {
        let tacky_program = tacky::Program {
            function_definitions: vec![tacky::FunctionDefinition {
                name: "main".into(),
                params: vec![],
                body: vec![
                    Instruction::Unary {
                        op: UnaryOperator::Negate,
                        src: Val::Constant(2),
                        dst: Val::Var("tmp.0".into()),
                    },
                    Instruction::Return(Val::Var("tmp.0".into())),
                ],
            }],
        };

        let ast = pass1(&tacky_program);

        assert_eq!(ast.function_definition.name, "main".into());
        assert_eq!(
            ast.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Ret,
            ]
        );
    }

    #[test]
    fn test_pass1_binary_add() {
        let tacky_program = tacky::Program {
            function_definitions: vec![tacky::FunctionDefinition {
                name: "main".into(),
                params: vec![],
                body: vec![Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(2),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.0".into()),
                }],
            }],
        };

        let ast = pass1(&tacky_program);

        assert_eq!(
            ast.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::Add,
                    src: asm::Operand::Imm(3),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
            ]
        );
    }

    #[test]
    fn test_pass1_binary_remainder() {
        let tacky_program = tacky::Program {
            function_definitions: vec![tacky::FunctionDefinition {
                name: "main".into(),
                params: vec![],
                body: vec![Instruction::Binary {
                    op: BinaryOperator::Remainder,
                    src1: Val::Constant(2),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.0".into()),
                }],
            }],
        };

        let ast = pass1(&tacky_program);

        assert_eq!(
            ast.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Cdq,
                asm::Instruction::Idiv(asm::Operand::Imm(3)),
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(asm::Reg::DX),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
            ]
        );
    }

    #[test]
    fn test_pass1_binary_bitwise_xor() {
        let tacky_program = tacky::Program {
            function_definitions: vec![tacky::FunctionDefinition {
                name: "main".into(),
                params: vec![],
                body: vec![Instruction::Binary {
                    op: BinaryOperator::BitXor,
                    src1: Val::Constant(2),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.0".into()),
                }],
            }],
        };

        let ast = pass1(&tacky_program);

        assert_eq!(
            ast.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitXor,
                    src: asm::Operand::Imm(3),
                    dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                },
            ]
        );
    }

    #[test]
    fn test_pass2() {
        let pass1 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Mov {
                        src: asm::Operand::Imm(2),
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    },
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::Add,
                        src: asm::Operand::Pseudo(asm::Identifier("tmp.1".into())),
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    },
                    asm::Instruction::Idiv(asm::Operand::Pseudo(asm::Identifier("tmp.1".into()))),
                    asm::Instruction::Mov {
                        src: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                        dst: asm::Operand::Reg(asm::Reg::AX),
                    },
                    asm::Instruction::Ret,
                ],
            },
        };

        let (pass2, size) = pass2(pass1).unwrap();
        assert_eq!(size, 8);

        assert_eq!(
            pass2.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(2),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::Add,
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Idiv(asm::Operand::Stack(asm::Offset(-8))),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-4)),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Ret,
            ]
        );
    }

    #[test]
    fn test_pass2_multiple_pseudos() {
        let pass1 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Mov {
                        src: asm::Operand::Imm(8),
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Pseudo(asm::Identifier("tmp.0".into())),
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.1".into())),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.1".into())),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Pseudo(asm::Identifier("tmp.1".into())),
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.2".into())),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Pseudo(asm::Identifier("tmp.2".into())),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Pseudo(asm::Identifier("tmp.2".into())),
                        dst: asm::Operand::Reg(asm::Reg::AX),
                    },
                    asm::Instruction::Ret,
                ],
            },
        };

        let (pass2, size) = pass2(pass1).unwrap();
        assert_eq!(size, 4 * 3); // 3 pseudo registers
        assert_eq!(
            pass2.function_definition.instructions,
            vec![
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(8),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-4)),
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Ret,
            ]
        );
    }

    #[test]
    fn test_pass3() {
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Mov {
                        src: asm::Operand::Imm(8),
                        dst: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Stack(asm::Offset(-4)),
                        dst: asm::Operand::Stack(asm::Offset(-8)),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Stack(asm::Offset(-8)),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Stack(asm::Offset(-8)),
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                    asm::Instruction::Unary {
                        op: asm::UnaryOperator::Neg,
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                    asm::Instruction::Cmp {
                        src1: asm::Operand::Stack(asm::Offset(-8)),
                        src2: asm::Operand::Stack(asm::Offset(-12)),
                    },
                    asm::Instruction::Mov {
                        src: asm::Operand::Stack(asm::Offset(-12)),
                        dst: asm::Operand::Reg(asm::Reg::AX),
                    },
                    asm::Instruction::Ret,
                ],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(8),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-4)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Cmp {
                    src1: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    src2: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                },
                asm::Instruction::Ret,
            ]
        );
    }

    #[test]
    fn test_pass3_idiv() {
        // idiv cannot have a constant operand
        //   -> becomes a mov to R10 + idiv
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![asm::Instruction::Idiv(asm::Operand::Imm(3))],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(3),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Idiv(asm::Operand::Reg(SRC_SCRATCH_REGISTER)),
            ]
        );
    }

    #[test]
    fn test_pass3_add_sub() {
        // add/sub cannot have two addresses
        //   -> becomes mov + add/sub
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::Add,
                        src: asm::Operand::Stack(asm::Offset(-12)),
                        dst: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::Sub,
                        src: asm::Operand::Stack(asm::Offset(-8)),
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                ],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::Add,
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::Sub,
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
            ]
        );
    }

    #[test]
    fn test_pass3_imull() {
        // imul cannot use a memory address as destination
        //   -> becomes a mov to R11 as scratch
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![asm::Instruction::Binary {
                    op: asm::BinaryOperator::Mult,
                    src: asm::Operand::Imm(3),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                }],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::Mult,
                    src: asm::Operand::Imm(3),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
            ]
        );
    }

    #[test]
    fn test_pass3_bitwise_and_or_xor() {
        // bitwise and/or/xor cannot have two addresses
        //   -> becomes mov + and/or/xor
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::BitAnd,
                        src: asm::Operand::Stack(asm::Offset(-12)),
                        dst: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::BitOr,
                        src: asm::Operand::Stack(asm::Offset(-8)),
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::BitXor,
                        src: asm::Operand::Stack(asm::Offset(-8)),
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                ],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitAnd,
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitOr,
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitXor,
                    src: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
            ]
        );
    }

    #[test]
    fn test_pass3_bitwise_shift() {
        // bitwise shift cannot have a src address
        //   -> becomes mov + and/or/xor
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::BitShiftLeft,
                        src: asm::Operand::Stack(asm::Offset(-12)),
                        dst: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Binary {
                        op: asm::BinaryOperator::BitShiftRight,
                        src: asm::Operand::Imm(9),
                        dst: asm::Operand::Stack(asm::Offset(-12)),
                    },
                ],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(asm::Reg::CX),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitShiftLeft,
                    src: asm::Operand::Reg(asm::Reg::CX),
                    dst: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Binary {
                    op: asm::BinaryOperator::BitShiftRight,
                    src: asm::Operand::Imm(9),
                    dst: asm::Operand::Stack(asm::Offset(-12)),
                },
            ]
        );
    }

    #[test]
    fn test_pass3_cmp() {
        // cmp cannot have two addresses
        //   -> becomes mov + cmp
        // cmp cannot have a constant as second operand
        //   -> becomes mov + cmp via r11d
        let pass2 = asm::Program {
            function_definition: asm::Function {
                name: "main".into(),
                instructions: vec![
                    asm::Instruction::Cmp {
                        src1: asm::Operand::Stack(asm::Offset(-12)),
                        src2: asm::Operand::Stack(asm::Offset(-4)),
                    },
                    asm::Instruction::Cmp {
                        src1: asm::Operand::Stack(asm::Offset(-8)),
                        src2: asm::Operand::Imm(5),
                    },
                ],
            },
        };

        let pass3 = pass3(&pass2, 12).unwrap();
        assert_eq!(
            pass3.function_definition.instructions,
            vec![
                asm::Instruction::AllocateStack(12),
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-12)),
                    dst: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                },
                asm::Instruction::Cmp {
                    src1: asm::Operand::Reg(SRC_SCRATCH_REGISTER),
                    src2: asm::Operand::Stack(asm::Offset(-4)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Imm(5),
                    dst: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                },
                asm::Instruction::Cmp {
                    src1: asm::Operand::Stack(asm::Offset(-8)),
                    src2: asm::Operand::Reg(DST_SCRATCH_REGISTER),
                },
            ]
        );
    }
}
