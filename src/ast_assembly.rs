//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* instructions)
//!   instruction = Mov(operand src, operand dst) | Ret
//!   operand = Imm(int) | Register
//!

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function_definition: Function,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Identifier(pub(crate) String);

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Mov { src: Operand, dst: Operand },
    Ret,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Operand {
    Imm(usize),
    Register,
}
