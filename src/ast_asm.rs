//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(function_definition*)
//!   function_definition = Function(identifier name,
//!                                  [ Mov(Reg(DI), param1),
//!                                    Mov(Reg(SI), param2),
//!                                    ...
//!                                    Mov(Stack(16), param7),
//!                                    MOV(Stack(24), param8),
//!                                    ...
//!                                  ] +
//!                                  instruction* instructions)
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
//!               | DeallocateStack(int)
//!               | Push(operand)
//!               | Call(identifier)
//!               | Ret
//!   unary_operator = Neg | Not
//!   binary_operator = Add | Sub | Mult
//!   operand = Imm(int)
//!           | Reg(reg)
//!           | Pseudo(identifier)
//!           | Stack(int)
//!   cond_code = E | NE | L | LE | G | GE
//!   reg = AX | CX | DX | DI | SI | R8 | R9 | R10 | R11
//!
//!
//! Register Usage:
//!
//!   AX: return value / idiv quotient
//!   DX: division remainder
//!   R10: scratch
//!   R11: scratch
//!

use crate::emitter::LABEL_PREFIX;
use std::fmt::{Display, Formatter};

pub(crate) const STACK_SLOT_SIZE: usize = 4; // 4 bytes per temporary variable
pub(crate) const ABI_PARAM_SIZE: usize = 8; // 8 bytes per parameter
pub(crate) const ABI_STACK_ALIGNMENT: usize = 16; // 16 byte alignment for stack

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) function_definitions: Vec<Function>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) stack_size: Option<usize>, // bytes
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub(crate) struct Identifier(pub(crate) String);

impl Display for Identifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{LABEL_PREFIX}{}", self.0)
    }
}

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
    DeallocateStack(usize),
    Push(Operand),
    Call(Identifier),
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
            Instruction::Cmp { src1, src2 } => write!(f, "cmpl\t{src1}, {src2}"),
            Instruction::Jmp { target } => write!(f, "jmp\t{target}"),
            Instruction::JmpCC { cc, target } => write!(f, "j{cc}\t{target}"),
            Instruction::SetCC { cc, dst } => match dst {
                Operand::Reg(r) => write!(
                    f,
                    "set{cc}\t%{r}",
                    r = r.fmt_with_width(RegisterWidth::W8Low)
                ),
                Operand::Stack(offset) => write!(f, "set{cc}\t{offset}(%rbp)"),
                _ => panic!("Invalid operand for SetCC"),
            },
            Instruction::Label(label) => write!(f, "{label}:"),
            Instruction::AllocateStack(size) => write!(f, "subq\t${size}, %rsp"),
            Instruction::Ret => write!(f, "ret"),

            Instruction::DeallocateStack(_) => {
                todo!()
            }
            Instruction::Push(_) => {
                todo!()
            }
            Instruction::Call(_) => {
                todo!()
            }
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
            BinaryOperator::BitShiftLeft => write!(f, "shll"),
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

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Reg {
    AX, // Accumulator
    CX, // Counter
    DX, // Data
    DI, // Destination Index
    SI, // Source Index
    R8,
    R9,
    R10,
    R11,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // 32-bit registers
        match self {
            Reg::AX => write!(f, "eax"),
            Reg::CX => write!(f, "ecx"), // FIXME, this is sometimes cl!!
            Reg::DX => write!(f, "edx"),
            Reg::DI => write!(f, "edi"),
            Reg::SI => write!(f, "esi"),
            Reg::R8 => write!(f, "r8d"),
            Reg::R9 => write!(f, "r9d"),
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
            Reg::DI => "di",
            Reg::SI => "si",
            Reg::R8 => "r8",
            Reg::R9 => "r9",
            Reg::R10 => "r10",
            Reg::R11 => "r11",
        }
    }

    fn extended(&self) -> bool {
        match self {
            Reg::AX => false,
            Reg::CX => false,
            Reg::DX => false,
            Reg::DI => false,
            Reg::SI => false,
            Reg::R8 => true,
            Reg::R9 => true,
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

impl Display for ConditionCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConditionCode::Equal => write!(f, "e"),
            ConditionCode::NotEqual => write!(f, "ne"),
            ConditionCode::LessThan => write!(f, "l"),
            ConditionCode::LessOrEqual => write!(f, "le"),
            ConditionCode::GreaterThan => write!(f, "g"),
            ConditionCode::GreaterOrEqual => write!(f, "ge"),
        }
    }
}
