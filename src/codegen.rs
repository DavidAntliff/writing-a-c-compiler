use crate::ast_asm as asm;
use crate::ast_asm::Reg;
use crate::tacky;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct CodegenError {
    pub message: String,
}

// Register used to move values between stack locations
const MOV_SCRATCH_REGISTER: Reg = Reg::R10;

pub(crate) fn parse(tacky: &tacky::Program) -> Result<asm::Program, CodegenError> {
    // Three passes

    // Pass 1 - convert TACKY AST to ASM AST, using pseudo registers for variables
    let ast = pass1(tacky);

    // Pass 2 - replace pseudo registers with Stack operands
    let (ast, stack_size) = pass2(ast)?;

    // Pass 3:
    //   1. Insert AllocateStack instruction at the beginning of function_definition,
    //   2. Rewrite invalid MOV instructions.
    let ast = pass3(&ast, stack_size)?;

    Ok(ast)
}

fn pass1(program: &tacky::Program) -> asm::Program {
    asm::Program {
        function_definition: function_definition(&program.function_definition),
    }
}

impl From<&tacky::Val> for asm::Operand {
    fn from(val: &tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(c) => asm::Operand::Imm(*c),
            tacky::Val::Var(identifier) => {
                asm::Operand::Pseudo(asm::Identifier(identifier.clone()))
            }
        }
    }
}

fn function_definition(function: &tacky::FunctionDefinition) -> asm::Function {
    let mut instructions = vec![];

    function
        .body
        .iter()
        .for_each(|instruction| match instruction {
            tacky::Instruction::Return(val) => {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::from(val),
                    dst: asm::Operand::Reg(asm::Reg::AX),
                });
                instructions.push(asm::Instruction::Ret);
            }
            tacky::Instruction::Unary { op, src, dst } => {
                let src = asm::Operand::from(src);
                let dst = asm::Operand::from(dst);
                let op = match op {
                    tacky::UnaryOperator::Complement => asm::UnaryOperator::Not,
                    tacky::UnaryOperator::Negate => asm::UnaryOperator::Neg,
                };
                instructions.push(asm::Instruction::Mov {
                    src,
                    dst: dst.clone(),
                });
                instructions.push(asm::Instruction::Unary { op, dst });
            }
        });

    asm::Function {
        name: asm::Identifier(function.name.clone()),
        instructions,
    }
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
        .for_each(|instruction| {
            match instruction {
                asm::Instruction::Mov { src, dst } => {
                    replacer.replace(src);
                    replacer.replace(dst);
                }
                asm::Instruction::Unary { op: _, dst } => {
                    replacer.replace(dst);
                }
                _ => {
                    // No action needed for other instructions
                }
            }
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
            asm::Instruction::Mov {
                src: asm::Operand::Stack(src),
                dst: asm::Operand::Stack(dst),
            } => {
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Stack(*src),
                    dst: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
                });
                instructions.push(asm::Instruction::Mov {
                    src: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(*dst),
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
    use crate::tacky::{Instruction, UnaryOperator, Val};

    #[test]
    fn test_pass1_return_constant() {
        let tacky_program = tacky::Program {
            function_definition: tacky::FunctionDefinition {
                name: "main".to_string(),
                body: vec![tacky::Instruction::Return(tacky::Val::Constant(2))],
            },
        };

        let ast = pass1(&tacky_program);

        assert_eq!(
            ast.function_definition.name,
            asm::Identifier("main".to_string())
        );
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
            function_definition: tacky::FunctionDefinition {
                name: "main".to_string(),
                body: vec![
                    Instruction::Unary {
                        op: UnaryOperator::Negate,
                        src: Val::Constant(2),
                        dst: Val::Var("tmp.0".into()),
                    },
                    Instruction::Return(Val::Var("tmp.0".into())),
                ],
            },
        };

        let ast = pass1(&tacky_program);

        assert_eq!(
            ast.function_definition.name,
            asm::Identifier("main".to_string())
        );
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
    fn test_pass2() {
        let pass1 = asm::Program {
            function_definition: asm::Function {
                name: asm::Identifier("main".to_string()),
                instructions: vec![
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
                ],
            },
        };

        let (pass2, size) = pass2(pass1).unwrap();
        assert_eq!(size, 4);

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
                name: asm::Identifier("main".to_string()),
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
                name: asm::Identifier("main".to_string()),
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
                    dst: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Unary {
                    op: asm::UnaryOperator::Neg,
                    dst: asm::Operand::Stack(asm::Offset(-8)),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Stack(asm::Offset(-8)),
                    dst: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
                },
                asm::Instruction::Mov {
                    src: asm::Operand::Reg(MOV_SCRATCH_REGISTER),
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
}
