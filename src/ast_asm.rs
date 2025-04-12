//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(function_definition)
//!   function_definition = Function(identifier name, instruction* instructions)
//!   instruction = Mov(operand src, operand dst)
//!               | Unary(unary_operator, operand)
//!               | Binary(binary_operator, operand, operand)
//!               | Cmp(operand, operand)
//!               | Idiv(operand)
//!               | Cdq
//!               | Jmp(identifier)
//!               | JmpCC(cond_code, identifier)
//!               | SetCC(cond_code, operand)
//!               | Label(identifier)
//!               | AllocateStack(int)
//!               | Ret
//!   unary_operator = Neg | Not
//!   binary_operator = Add | Sub | Mult
//!   operand = Imm(int)
//!           | Reg(reg)
//!           | Pseudo(identifier)
//!           | Stack(int)
//!   cond_code = E | NE | L | LE | G | GE
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

impl<T> From<T> for Identifier
where
    T: Into<String>,
{
    fn from(value: T) -> Self {
        Identifier(value.into())
    }
}

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
    Cmp {
        src1: Operand,
        src2: Operand,
    },
    Idiv(Operand),
    Cdq,
    Jmp {
        target: Identifier,
    },
    JmpCC {
        cc: ConditionCode,
        target: Identifier,
    },
    SetCC {
        cc: ConditionCode,
        dst: Operand,
    },
    Label(Identifier),
    /// Allocate stack space in bytes
    AllocateStack(usize),
    Ret,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Mov { src, dst } => write!(f, "movl\t{src}, {dst}"),
            Instruction::Unary { op, dst } => write!(f, "{op}\t{dst}"),
            Instruction::Binary {
                op,
                src: Operand::Reg(r),
                dst,
            } if *op == BinaryOperator::BitShiftLeft || *op == BinaryOperator::BitShiftRight => {
                write!(
                    f,
                    "{op}\t%{r}, {dst}",
                    r = r.fmt_with_width(RegisterWidth::W8Low)
                )
            }
            Instruction::Binary { op, src, dst } => write!(f, "{op}\t{src}, {dst}"),
            Instruction::Idiv(src) => write!(f, "idivl\t{src}"),
            Instruction::Cdq => write!(f, "cdq"),
            Instruction::Cmp { .. } => todo!(),
            Instruction::Jmp { .. } => todo!(),
            Instruction::JmpCC { .. } => todo!(),
            Instruction::SetCC { .. } => todo!(),
            Instruction::Label(_) => todo!(),
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

#[allow(unused)]
enum RegisterWidth {
    W8Low,
    W8High,
    W16,
    W32,
    W64,
}

impl Reg {
    fn base(&self) -> &str {
        match self {
            Reg::AX => "a",
            Reg::CX => "c",
            Reg::DX => "d",
            Reg::R10 => "r10",
            Reg::R11 => "r11",
        }
    }

    fn extended(&self) -> bool {
        match self {
            Reg::AX => false,
            Reg::CX => false,
            Reg::DX => false,
            Reg::R10 => true,
            Reg::R11 => true,
        }
    }

    fn fmt_with_width(&self, width: RegisterWidth) -> String {
        if self.extended() {
            match width {
                RegisterWidth::W8Low => format!("{}b", self.base()),
                RegisterWidth::W8High => panic!("{}h is not a valid register", self.base()),
                RegisterWidth::W16 => format!("{}w", self.base()),
                RegisterWidth::W32 => format!("{}d", self.base()),
                RegisterWidth::W64 => self.base().to_string(),
            }
        } else {
            match width {
                RegisterWidth::W8Low => format!("{}l", self.base()),
                RegisterWidth::W8High => format!("{}h", self.base()),
                RegisterWidth::W16 => self.base().to_string(),
                RegisterWidth::W32 => format!("e{}", self.base()),
                RegisterWidth::W64 => format!("r{}", self.base()),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ConditionCode {
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}
