//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* instructions)
//!   instruction = Mov(operand src, operand dst)
//!               | Unary(unary operator, operand)
//!               | AllocateStack(int)
//!               | Ret
//!   unary_operator = Neg | Not
//!   operand = Imm(int)
//!           | Reg(reg)
//!           | Pseudo(identifier)
//!           | Stack(int)
//!   reg = AX | R10

use std::fmt::{Display, Formatter};

pub(crate) const STACK_SLOT_SIZE: usize = 4; // 4 bytes per temporary variable

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function_definition: Function,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub(crate) struct Identifier(pub(crate) String);

#[derive(Debug, PartialEq, Clone, Copy, Hash, Eq)]
pub(crate) struct Offset(pub(crate) isize);

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}", self.0)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Instruction {
    Mov {
        src: Operand,
        dst: Operand,
    },
    Unary {
        op: UnaryOperator,
        dst: Operand,
    },
    /// Allocate stack space in bytes
    AllocateStack(usize),
    Ret,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum UnaryOperator {
    Neg,
    Not,
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperator::Neg => write!(f, "negl"),
            UnaryOperator::Not => write!(f, "notl"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Operand {
    Imm(usize),
    Reg(Reg),
    Pseudo(Identifier),
    /// Stack offset in bytes
    Stack(Offset),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Imm(i) => write!(f, "${i}"),
            Operand::Reg(reg) => write!(f, "%{reg}"),
            Operand::Pseudo(_) => panic!("Pseudo operands should not be emitted"),
            Operand::Stack(n) => write!(f, "{n}(%rbp)"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Reg {
    AX,
    R10,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Reg::AX => write!(f, "eax"),
            Reg::R10 => write!(f, "r10d"),
        }
    }
}
