//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* instructions)
//!   instruction = Mov(operand src, operand dst)
//!               | Unary(unary_operator, operand)
//!               | Binary(binary_operator, operand, operand)
//!               | Idiv(operand)
//!               | Cdq
//!               | AllocateStack(int)
//!               | Ret
//!   unary_operator = Neg | Not
//!   binary_operator = Add | Sub | Mult
//!   operand = Imm(int)
//!           | Reg(reg)
//!           | Pseudo(identifier)
//!           | Stack(int)
//!   reg = AX | DX | R10 | R11
//!
//!
//! Register Usage:
//!
//!   AX: return value / idiv quotient
//!   DX: division remainder
//!   R10: scratch
//!   R11: scratch
//!

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
    Binary {
        op: BinaryOperator,
        src: Operand,
        dst: Operand,
    },
    Idiv(Operand),
    Cdq,
    /// Allocate stack space in bytes
    AllocateStack(usize),
    Ret,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Mov { src, dst } => write!(f, "movl\t{src}, {dst}"),
            Instruction::Unary { op, dst } => write!(f, "{op}\t{dst}"),
            Instruction::Binary { op, src, dst } => write!(f, "{op}\t{src}, {dst}"),
            Instruction::Idiv(src) => write!(f, "idivl\t{src}"),
            Instruction::Cdq => write!(f, "cdq"),
            Instruction::AllocateStack(size) => write!(f, "subq\t${size}, %rsp"),
            Instruction::Ret => write!(f, "ret"),
        }
    }
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
pub(crate) enum BinaryOperator {
    Add,
    Sub,
    Mult,
    BitAnd,
    BitOr,
    BitXor,
    BitShiftLeft,
    BitShiftRight,
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "addl"),
            BinaryOperator::Sub => write!(f, "subl"),
            BinaryOperator::Mult => write!(f, "imull"),
            BinaryOperator::BitAnd => write!(f, "andl"),
            BinaryOperator::BitOr => write!(f, "orl"),
            BinaryOperator::BitXor => write!(f, "xorl"),
            BinaryOperator::BitShiftLeft => write!(f, "sall"),
            BinaryOperator::BitShiftRight => write!(f, "sarl"),
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
    CX,
    DX,
    R10,
    R11,
}

// TODO: we may need to ditch the Display trait and use
// a different approach to emit the registers, as sometimes
// we need to use the 64-bit registers (e.g. r10, r11) and
// sometimes we need to use the 32-bit registers (e.g. eax, ecx, edx)

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Reg::AX => write!(f, "eax"),
            Reg::CX => write!(f, "ecx"), // FIXME, this is sometimes cl!!
            Reg::DX => write!(f, "edx"),
            Reg::R10 => write!(f, "r10d"),
            Reg::R11 => write!(f, "r11d"),
        }
    }
}
