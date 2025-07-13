//! AST for x86_64 assembly
//!
//! ASDL:
//!   program = Program(top_level*)
//!   top_level = Function(identifier name, bool global, instruction* instructions)
//!               | StaticVariable(identifier name, bool global, int init)
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
//!           | Data(identifier)
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

use crate::emitter::abi::PUBLIC_PREFIX;
use crate::emitter::abi::{INDIRECT_CALL_SUFFIX, PRIVATE_PREFIX};
use crate::semantics::{IdentifierAttrs, SymbolTable, Type};
use assert_matches::assert_matches;
use std::fmt::{Display, Formatter};

pub(crate) const STACK_SLOT_SIZE: usize = 4; // 4 bytes per temporary variable
pub(crate) const ABI_PARAM_SIZE: usize = 8; // 8 bytes per parameter
pub(crate) const ABI_STACK_ALIGNMENT: usize = 16; // 16 byte alignment for stack

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Program {
    pub(crate) top_level: Vec<TopLevel>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum TopLevel {
    Function(Function),
    StaticVariable {
        name: Identifier,
        global: bool,
        init: usize,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) global: bool,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) stack_size: Option<usize>, // bytes
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub(crate) struct Identifier(pub(crate) String);

impl Identifier {
    pub(crate) fn as_private_symbol(&self) -> String {
        format!("{PRIVATE_PREFIX}{}", self.0)
    }

    pub(crate) fn as_public_symbol(&self) -> String {
        format!("{PUBLIC_PREFIX}{}", self.0)
    }

    pub(crate) fn emit(&self, symbol_table: &SymbolTable) -> String {
        match symbol_table.get(&self.0) {
            Some(item) => match item.attrs {
                IdentifierAttrs::Fun { .. } => {
                    assert_matches!(
                        item.type_,
                        Type::Function { .. },
                        "Expected function type for identifier {}",
                        self.0
                    );
                    // Static and global function names always use the export style
                    self.as_public_symbol()
                }
                IdentifierAttrs::Static { .. } => self.as_public_symbol(),
                IdentifierAttrs::Local => self.as_private_symbol(),
            },
            None => panic!("Identifier {} not found in symbol table", self.0),
        }
    }

    fn emit_at_call_site(&self, symbol_table: &SymbolTable) -> String {
        match symbol_table.get(&self.0) {
            Some(item) => match item.attrs {
                IdentifierAttrs::Fun { global, .. } => {
                    assert_matches!(
                        item.type_,
                        Type::Function { .. },
                        "Expected function type for identifier {}",
                        self.0
                    );
                    if global {
                        format!("{}{INDIRECT_CALL_SUFFIX}", self.as_public_symbol())
                    } else {
                        self.as_public_symbol()
                    }
                }
                _ => panic!("Identifier {} is not a function", self.0),
            },
            None => panic!("Identifier {} not found in symbol table", self.0),
        }
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

impl Instruction {
    pub(crate) fn emit(&self, symbol_table: &SymbolTable) -> String {
        match self {
            Instruction::Mov { src, dst } => format!(
                "movl\t{src}, {dst}",
                src = src.emit(symbol_table),
                dst = dst.emit(symbol_table)
            ),
            Instruction::Unary { op, dst } => format!("{op}\t{dst}", dst = dst.emit(symbol_table)),
            Instruction::Binary {
                op,
                src: Operand::Reg(r),
                dst,
            } if *op == BinaryOperator::BitShiftLeft || *op == BinaryOperator::BitShiftRight => {
                format!(
                    "{op}\t%{r}, {dst}",
                    r = r.fmt_with_width(RegisterWidth::W8Low),
                    dst = dst.emit(symbol_table)
                )
            }
            Instruction::Binary { op, src, dst } => format!(
                "{op}\t{src}, {dst}",
                src = src.emit(symbol_table),
                dst = dst.emit(symbol_table)
            ),
            Instruction::Idiv(src) => format!("idivl\t{src}", src = src.emit(symbol_table)),
            Instruction::Cdq => "cdq".into(),
            Instruction::Cmp { src1, src2 } => format!(
                "cmpl\t{src1}, {src2}",
                src1 = src1.emit(symbol_table),
                src2 = src2.emit(symbol_table)
            ),
            Instruction::Jmp { target } => format!("jmp\t{}", target.as_private_symbol()),
            Instruction::JmpCC { cc, target } => {
                format!("j{cc}\t{}", target.as_private_symbol())
            }
            Instruction::SetCC { cc, dst } => match dst {
                Operand::Reg(r) => {
                    format!("set{cc}\t%{r}", r = r.fmt_with_width(RegisterWidth::W8Low))
                }
                Operand::Stack(offset) => format!("set{cc}\t{offset}(%rbp)"),
                _ => panic!("Invalid operand for SetCC"),
            },
            Instruction::Label(label) => format!("{}:", label.as_private_symbol()),
            Instruction::AllocateStack(size) => format!("subq\t${size}, %rsp"),
            Instruction::Ret => "ret".into(),

            Instruction::DeallocateStack(size) => {
                format!("addq\t${size}, %rsp")
            }
            Instruction::Push(op) => {
                // Must be 64-bit register or immediate value
                let src = match op {
                    Operand::Reg(r) => format!("%{}", r.fmt_with_width(RegisterWidth::W64)),
                    _ => op.emit(symbol_table),
                };
                format!("pushq\t{src}")
                //format!("pushq\t{op}")
            }
            Instruction::Call(label) => {
                format!("call\t{}", label.emit_at_call_site(symbol_table))
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
    Data(Identifier),
}

impl Operand {
    pub(crate) fn emit(&self, symbol_table: &SymbolTable) -> String {
        match self {
            Operand::Imm(i) => format!("${i}"),
            Operand::Reg(reg) => format!("%{reg}"), // TODO: needs width consideration
            Operand::Pseudo(_) => panic!("Pseudo operands should not be emitted"),
            Operand::Stack(n) => format!("{n}(%rbp)"),
            Operand::Data(id) => format!("{}(%rip)", id.emit(symbol_table)),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Reg {
    // Core Registers
    AX, // Accumulator
    CX, // Counter
    DX, // Data
    // Index / Pointer Registers
    DI, // Destination Index
    SI, // Source Index
    // Extended Registers
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

#[expect(unused)]
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
        matches!(self, Reg::R8 | Reg::R9 | Reg::R10 | Reg::R11)
    }

    fn core(&self) -> bool {
        matches!(self, Reg::AX | Reg::CX | Reg::DX)
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
            let suffix = if self.core() { "x" } else { "" };
            match width {
                RegisterWidth::W8Low => format!("{}l", self.base()),
                RegisterWidth::W8High => format!("{}h", self.base()),
                RegisterWidth::W16 => format!("{}{suffix}", self.base()),
                RegisterWidth::W32 => format!("e{}{suffix}", self.base()),
                RegisterWidth::W64 => format!("r{}{suffix}", self.base()),
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
