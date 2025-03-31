use crate::ast_assembly::{Identifier, Instruction, Operand};
use crate::ast_c::Expression;
use crate::{ast_assembly, ast_c};
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct CodegenError {
    pub message: String,
}

pub(crate) fn parse(ast: &ast_c::Program) -> Result<ast_assembly::Program, CodegenError> {
    Ok(program(ast))
}

fn program(program: &ast_c::Program) -> ast_assembly::Program {
    ast_assembly::Program {
        function_definition: function_definition(&program.function),
    }
}

fn function_definition(function: &ast_c::Function) -> ast_assembly::Function {
    let mut instructions = vec![];

    match function.body {
        ast_c::Statement::Return(ref exp) => match exp {
            Expression::Constant(c) => {
                instructions.push(Instruction::Mov {
                    src: Operand::Imm(*c),
                    dst: Operand::Register,
                });
                instructions.push(Instruction::Ret);
            }
            Expression::Unary(_, _) => unimplemented!(),
        },
    }

    ast_assembly::Function {
        name: Identifier(function.name.clone()),
        instructions,
    }
}
