use crate::ast_asm as asm;
use crate::tacky;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct CodegenError {
    pub message: String,
}

pub(crate) fn parse(tacky: &tacky::Program) -> Result<asm::Program, CodegenError> {
    // Three passes

    // Pass 1 - convert TACKY AST to ASM AST, using pseudo registers for variables
    let ast = pass1(tacky);

    // Pass 2 - replace pseudo registers with Stack operands
    //let ast = pass2(ast)?;

    // Pass 3:
    //   1. Insert AllocateStack instruction at the beginning of function_definition,
    //   2. Rewrite invalid MOV instructions.
    //let ast = pass3(ast)?;

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
}
