//! TACKY AST
//!
//! ASDL:
//!   program = Program(top_level*)
//!   top_level = Function(identifier name, bool global, identifier* params, instruction* body)
//!               | StaticVariable(identifier name, bool global, type t, static_init init)
//!   instruction = Return(val)
//!               | SignExtend(val src, val dst)
//!               | Truncate(val src, val dst)
//!               | Unary(unary_operator, val src, val dst)
//!               | Binary(binary_operator, val src1, val src2, val dst)
//!               | Copy(val src, val dst)
//!               | Jump(identifier target)
//!               | JumpIfZero(val condition, identifier target)
//!               | JumpIfNotZero(val condition, identifier target)
//!               | Label(identifier)
//!               | FunCall(identifier name, val* args, val dst)
//!   val = Constant(const) | Var(identifier)
//!   const = ConstInt(int) | ConstLong(int)
//!   unary_operator = Complement | Negate | Not
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!                   | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight
//!                   | Equal | NotEqual | LessThan | LessOrEqual | GreaterThan | GreaterOrEqual
//!

use crate::ast_c;
use crate::id_gen::IdGenerator;
use crate::semantics::{IdentifierAttrs, InitialValue, StaticInit, SymbolTable};
use thiserror::Error;

// TODO: maybe this should be the same as lexer::Identifier?
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

#[derive(Debug, PartialEq)]
pub(crate) struct Program {
    pub(crate) top_level: Vec<TopLevel>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum TopLevel {
    Function(Function),
    StaticVariable {
        name: Identifier,
        global: bool,
        t: ast_c::Type,
        init: StaticInit,
    },
}

#[derive(Debug, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: Identifier,
    pub(crate) global: bool,
    pub(crate) params: Vec<Identifier>,
    pub(crate) body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Return(Val),
    SignExtend {
        src: Val,
        dst: Val,
    },
    Truncate {
        src: Val,
        dst: Val,
    },
    Unary {
        op: UnaryOperator,
        src: Val,
        dst: Val,
    },
    Binary {
        op: BinaryOperator,
        src1: Val,
        src2: Val,
        dst: Val,
    },
    Copy {
        src: Val,
        dst: Val,
    },
    Jump {
        target: Identifier,
    },
    JumpIfZero {
        condition: Val,
        target: Identifier,
    },
    JumpIfNotZero {
        condition: Val,
        target: Identifier,
    },
    Label(Identifier),
    FunCall {
        name: Identifier,
        args: Vec<Val>,
        dst: Val,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Val {
    Constant(ast_c::Const),
    Var(Identifier),
}

impl Val {
    pub(crate) fn constant(v: impl Into<ast_c::Const>) -> Self {
        Val::Constant(v.into())
    }

    #[allow(dead_code)]
    pub(crate) fn var(v: impl Into<String>) -> Self {
        Val::Var(v.into().into())
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq)]
pub(crate) enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    //And,
    //Or,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct TackyError {
    pub message: String,
}

pub(crate) fn emit_program(
    program: &ast_c::TypedProgram,
    symbol_table: &mut SymbolTable,
) -> Result<Program, TackyError> {
    let mut id_gen = IdGenerator::new();
    let mut top_level = vec![];

    for declaration in &program.declarations {
        match declaration {
            ast_c::TypedDeclaration::FunDecl(fun_decl) => {
                let fun_def = emit_function_definition(fun_decl, &mut id_gen, symbol_table);
                if let Some(fun_def) = fun_def {
                    top_level.push(TopLevel::Function(fun_def));
                }
            }
            ast_c::TypedDeclaration::VarDecl(_) => {
                // Do not emit tacky for file-scope variable declarations
            }
        }
    }

    let static_symbols = convert_symbols_to_tacky(symbol_table)?;
    top_level.extend(static_symbols);

    Ok(Program { top_level })
}

fn convert_symbols_to_tacky(symbol_table: &SymbolTable) -> Result<Vec<TopLevel>, TackyError> {
    let mut tacky_defs = vec![];

    for (name, entry) in symbol_table.iter_sorted() {
        if let IdentifierAttrs::Static { init, global } = &entry.attrs {
            match init {
                InitialValue::Initial(init) => tacky_defs.push(TopLevel::StaticVariable {
                    name: name.into(),
                    global: *global,
                    t: entry.type_.clone(),
                    init: init.clone(),
                }),
                InitialValue::Tentative => {
                    let init = match entry.type_ {
                        ast_c::Type::Int => StaticInit::IntInit(0),
                        ast_c::Type::Long => StaticInit::LongInit(0),
                        _ => {
                            return Err(TackyError {
                                message: "Unsupported type for tentative definition".into(),
                            });
                        }
                    };
                    tacky_defs.push(TopLevel::StaticVariable {
                        name: name.into(),
                        global: *global,
                        t: entry.type_.clone(),
                        init,
                    })
                }
                InitialValue::NoInitialiser => (),
            }
        }
    }

    Ok(tacky_defs)
}

fn emit_function_definition(
    function_decl: &ast_c::TypedFunDecl,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Function> {
    // Drop function declarations without a body
    let body = match &function_decl.body {
        None => {
            return None;
        }
        Some(body) => body,
    };

    let name: Identifier = (&function_decl.name).into();

    let mut instructions = vec![];

    let params = function_decl
        .params
        .iter()
        .map(|p| Identifier(p.clone()))
        .collect::<Vec<_>>();

    let val = emit_block(body, &mut instructions, id_gen, symbol_table);
    assert!(val.is_none(), "emit_block should return None");

    // Add a Return(0) to the end of all functions (see page 112)
    instructions.push(Instruction::Return(Val::constant(0)));

    let global = symbol_table.expect_fun_global(&function_decl.name);

    Some(Function {
        name,
        global,
        params,
        body: instructions,
    })
}

fn make_temporary(id_gen: &mut IdGenerator) -> Identifier {
    format!("tmp.{}", id_gen.next()).into()
}

#[expect(dead_code)]
fn make_label(prefix: &str, id_gen: &mut IdGenerator) -> Identifier {
    format!("{prefix}.{}", id_gen.next()).into()
}

fn make_two_labels(prefixes: &[&str], id_gen: &mut IdGenerator) -> (Identifier, Identifier) {
    let id = id_gen.next();
    (
        format!("{}.{}", prefixes[0], id).into(),
        format!("{}.{}", prefixes[1], id).into(),
    )
}

fn make_tacky_variable(
    t: &ast_c::Type,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Val {
    let var_name = make_temporary(id_gen);
    symbol_table.add(var_name.0.clone(), t.clone(), IdentifierAttrs::Local);
    Val::Var(var_name)
}

// Previously called "emit_tacky"
fn emit_expression(
    exp: &ast_c::TypedExpression,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Val {
    match exp {
        ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Constant(c)) => {
            Val::Constant(c.clone())
        }
        ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Var(identifier)) => {
            Val::Var(identifier.into())
        }
        ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Cast(t, inner)) => {
            let result = emit_expression(inner, instructions, id_gen, symbol_table);
            if t == inner.get_type() {
                // No cast needed
                return result;
            }
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            if *t == ast_c::Type::Long {
                instructions.push(Instruction::SignExtend {
                    src: result,
                    dst: dst.clone(),
                })
            } else {
                instructions.push(Instruction::Truncate {
                    src: result,
                    dst: dst.clone(),
                })
            }
            dst
        }
        ast_c::TypedExpression(t, ast_c::InnerTypedExpression::Unary(op, inner)) => {
            let src = emit_expression(inner, instructions, id_gen, symbol_table);
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            let tacky_op = convert_unop(op);
            instructions.push(Instruction::Unary {
                op: tacky_op,
                src,
                dst: dst.clone(),
            });
            dst
        }

        // Handle short-circuit evaluation for And / Or
        ast_c::TypedExpression(
            t,
            ast_c::InnerTypedExpression::Binary(ast_c::BinaryOperator::And, e1, e2),
        ) => {
            let (label_false, label_end) = make_two_labels(&["and_false", "and_end"], id_gen);

            let v1 = emit_expression(e1, instructions, id_gen, symbol_table);
            instructions.push(Instruction::JumpIfZero {
                condition: v1.clone(),
                target: label_false.clone(),
            });
            let v2 = emit_expression(e2, instructions, id_gen, symbol_table);
            instructions.push(Instruction::JumpIfZero {
                condition: v2.clone(),
                target: label_false.clone(),
            });
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            instructions.push(Instruction::Copy {
                src: Val::constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_false));
            instructions.push(Instruction::Copy {
                src: Val::constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::TypedExpression(
            t,
            ast_c::InnerTypedExpression::Binary(ast_c::BinaryOperator::Or, e1, e2),
        ) => {
            let (label_true, label_end) = make_two_labels(&["or_true", "or_end"], id_gen);

            let v1 = emit_expression(e1, instructions, id_gen, symbol_table);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v1.clone(),
                target: label_true.clone(),
            });
            let v2 = emit_expression(e2, instructions, id_gen, symbol_table);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v2.clone(),
                target: label_true.clone(),
            });
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            instructions.push(Instruction::Copy {
                src: Val::constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_true));
            instructions.push(Instruction::Copy {
                src: Val::constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::TypedExpression(t, ast_c::InnerTypedExpression::Binary(op, e1, e2)) => {
            // Unsequenced - indeterminate order of evaluation
            let src1 = emit_expression(e1, instructions, id_gen, symbol_table);
            let src2 = emit_expression(e2, instructions, id_gen, symbol_table);
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            let tacky = convert_binop(op);
            instructions.push(Instruction::Binary {
                op: tacky,
                src1,
                src2,
                dst: dst.clone(),
            });
            dst
        }

        ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Assignment(lhs, rhs)) => {
            if let ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Var(v)) = &**lhs {
                let result = emit_expression(rhs, instructions, id_gen, symbol_table);
                instructions.push(Instruction::Copy {
                    src: result,
                    dst: Val::Var(v.into()),
                });
                Val::Var(v.into())
            } else {
                unreachable!("lhs should be a variable");
            }
        }

        ast_c::TypedExpression(_, ast_c::InnerTypedExpression::Conditional(cond, e1, e2)) => {
            emit_exp_conditional(cond, e1, e2, instructions, id_gen, symbol_table)
        }

        ast_c::TypedExpression(t, ast_c::InnerTypedExpression::FunctionCall(name, args)) => {
            let dst = make_tacky_variable(t, id_gen, symbol_table);
            let tacky_args: Vec<Val> = args
                .iter()
                .map(|arg| emit_expression(arg, instructions, id_gen, symbol_table))
                .collect();
            instructions.push(Instruction::FunCall {
                name: name.clone().into(),
                args: tacky_args,
                dst: dst.clone(),
            });
            dst
        }
    }
}

fn convert_unop(op: &ast_c::UnaryOperator) -> UnaryOperator {
    match op {
        ast_c::UnaryOperator::Complement => UnaryOperator::Complement,
        ast_c::UnaryOperator::Negate => UnaryOperator::Negate,
        ast_c::UnaryOperator::Not => UnaryOperator::Not,
    }
}

fn convert_binop(op: &ast_c::BinaryOperator) -> BinaryOperator {
    match op {
        ast_c::BinaryOperator::Add => BinaryOperator::Add,
        ast_c::BinaryOperator::Subtract => BinaryOperator::Subtract,
        ast_c::BinaryOperator::Multiply => BinaryOperator::Multiply,
        ast_c::BinaryOperator::Divide => BinaryOperator::Divide,
        ast_c::BinaryOperator::Remainder => BinaryOperator::Remainder,
        ast_c::BinaryOperator::BitAnd => BinaryOperator::BitAnd,
        ast_c::BinaryOperator::BitOr => BinaryOperator::BitOr,
        ast_c::BinaryOperator::BitXor => BinaryOperator::BitXor,
        ast_c::BinaryOperator::ShiftLeft => BinaryOperator::ShiftLeft,
        ast_c::BinaryOperator::ShiftRight => BinaryOperator::ShiftRight,
        ast_c::BinaryOperator::Equal => BinaryOperator::Equal,
        ast_c::BinaryOperator::NotEqual => BinaryOperator::NotEqual,
        ast_c::BinaryOperator::LessThan => BinaryOperator::LessThan,
        ast_c::BinaryOperator::GreaterThan => BinaryOperator::GreaterThan,
        ast_c::BinaryOperator::LessOrEqual => BinaryOperator::LessOrEqual,
        ast_c::BinaryOperator::GreaterOrEqual => BinaryOperator::GreaterOrEqual,
        _ => {
            panic!("Unsupported binary operator: {op:?}");
        }
    }
}

fn emit_block(
    block: &ast_c::TypedBlock,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    for item in &block.items {
        emit_block_item(item, instructions, id_gen, symbol_table);
    }
    None
}

fn emit_block_item(
    item: &ast_c::TypedBlockItem,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    match item {
        ast_c::TypedBlockItem::S(statement) => {
            if let Some(instruction) = emit_statement(statement, instructions, id_gen, symbol_table)
            {
                instructions.push(instruction);
            }
        }
        ast_c::TypedBlockItem::D(declaration) => {
            emit_declaration(declaration, instructions, id_gen, symbol_table);
        }
    }
    None
}

fn emit_declaration(
    decl: &ast_c::TypedDeclaration,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    match decl {
        ast_c::TypedDeclaration::FunDecl(decl) => {
            let def = emit_function_definition(decl, id_gen, symbol_table);
            assert_eq!(def, None, "Function declaration should not have a body");
        }
        ast_c::TypedDeclaration::VarDecl(decl) => {
            emit_variable_declaration(decl, instructions, id_gen, symbol_table);
        }
    }
    None
}

fn emit_variable_declaration(
    ast_c::TypedVarDecl {
        name,
        init,
        var_type: _,
        storage_class,
    }: &ast_c::TypedVarDecl,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    // Do not emit for variables with static or extern storage class
    if storage_class.is_some() {
        return None;
    }

    if let Some(init) = &init {
        let result = emit_expression(init, instructions, id_gen, symbol_table);
        instructions.push(Instruction::Copy {
            src: result,
            dst: Val::Var(name.clone().into()),
        });
    }
    None
}

fn emit_statement(
    statement: &ast_c::TypedStatement,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    match statement {
        ast_c::TypedStatement::Return(exp) => {
            let val = emit_expression(exp, instructions, id_gen, symbol_table);
            Some(Instruction::Return(val))
        }
        ast_c::TypedStatement::Expression(exp) => {
            let _ = emit_expression(exp, instructions, id_gen, symbol_table);
            // No need to return anything for an expression statement
            None
        }
        ast_c::TypedStatement::If {
            condition,
            then_block: then,
            else_block: else_,
        } => emit_statement_if(condition, then, else_, instructions, id_gen, symbol_table),
        ast_c::TypedStatement::Labeled { label, statement } => {
            instructions.push(Instruction::Label(label.clone().into()));
            emit_statement(statement, instructions, id_gen, symbol_table)
        }
        ast_c::TypedStatement::Goto(label) => {
            instructions.push(emit_jump(&label.into()));
            None
        }
        ast_c::TypedStatement::Compound(block) => {
            emit_block(block, instructions, id_gen, symbol_table)
        }
        ast_c::TypedStatement::Break(Some(label)) => {
            instructions.push(emit_jump(&break_label(label)));
            None
        }
        ast_c::TypedStatement::Break(None) => {
            panic!("Break without label is not supported in Tacky")
        }
        ast_c::TypedStatement::Continue(Some(label)) => {
            instructions.push(emit_jump(&continue_label(label)));
            None
        }
        ast_c::TypedStatement::Continue(None) => {
            panic!("Continue without label is not supported in Tacky")
        }
        ast_c::TypedStatement::While {
            condition,
            body,
            loop_label,
        } => emit_while(
            condition,
            body,
            loop_label,
            instructions,
            id_gen,
            symbol_table,
        ),
        ast_c::TypedStatement::DoWhile {
            body,
            condition,
            loop_label,
        } => emit_do_while(
            body,
            condition,
            loop_label,
            instructions,
            id_gen,
            symbol_table,
        ),
        ast_c::TypedStatement::For {
            init,
            condition,
            post,
            body,
            loop_label,
        } => emit_for(
            init,
            condition,
            post,
            body,
            loop_label,
            instructions,
            id_gen,
            symbol_table,
        ),
        ast_c::TypedStatement::Null => None,
    }
}

fn emit_statement_if(
    condition: &ast_c::TypedExpression,
    then: &ast_c::TypedStatement,
    else_: &Option<Box<ast_c::TypedStatement>>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    // if (condition) { then }:
    //   <instructions for condition>
    //   c = <result of condition>
    //   JumpIfZero(c, end)
    //   <instructions for then-statement>
    //   Label(end)
    //
    // if (condition) { then } else { else_ }:
    //   <instructions for condition>
    //   c = <result of condition>
    //   JumpIfZero(c, else_label)
    //   <instructions for then-statement>
    //   Jump(end)
    //   Label(else_label)
    //   <instructions for else-statement>
    //   Label(end)

    let (label_else, label_end) = make_two_labels(&["if_else", "if_end"], id_gen);

    // if:
    let cond_val = emit_expression(condition, instructions, id_gen, symbol_table);

    instructions.push(Instruction::JumpIfZero {
        condition: cond_val.clone(),
        target: if else_.is_none() {
            label_end.clone()
        } else {
            label_else.clone()
        },
    });

    // then:
    if let Some(instruction) = emit_statement(then, instructions, id_gen, symbol_table) {
        instructions.push(instruction);
    }

    if let Some(else_stmt) = else_ {
        // Jump to end after "then"
        instructions.push(Instruction::Jump {
            target: label_end.clone(),
        });

        // else:
        instructions.push(Instruction::Label(label_else));

        if let Some(instruction) = emit_statement(else_stmt, instructions, id_gen, symbol_table) {
            instructions.push(instruction);
        }
    }

    instructions.push(Instruction::Label(label_end));

    // No return value for if statements
    None
}

fn emit_exp_conditional(
    condition: &ast_c::TypedExpression,
    e1: &ast_c::TypedExpression,
    e2: &ast_c::TypedExpression,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Val {
    // <instructions for condition>
    // c = <result of condition>
    // JumpIfZero(c, e2_label)
    // <instructions to calculate e1>
    // v1 = <result of e1>
    // result = v1
    // Jump(end)
    // Label(e2_label)
    // <instructions to calculate e2>
    // v2 = <result of e2>
    // result = v2
    // Label(end)

    let (label_e2, label_end) = make_two_labels(&["cond_e2", "cond_end"], id_gen);
    let t = condition.get_type();

    let c = make_tacky_variable(t, id_gen, symbol_table);
    let v1 = make_tacky_variable(t, id_gen, symbol_table);
    let v2 = make_tacky_variable(t, id_gen, symbol_table);
    let result = make_tacky_variable(t, id_gen, symbol_table);

    // <instructions for condition>
    // c = <result of condition>
    let c_val = emit_expression(condition, instructions, id_gen, symbol_table);
    instructions.push(Instruction::Copy {
        src: c_val,
        dst: c.clone(),
    });

    // JumpIfZero(c, e2_label)
    instructions.push(Instruction::JumpIfZero {
        condition: c,
        target: label_e2.clone(),
    });

    // e1:
    let val_e1 = emit_expression(e1, instructions, id_gen, symbol_table);

    // v1 = <result of e1>
    instructions.push(Instruction::Copy {
        src: val_e1.clone(),
        dst: v1.clone(),
    });

    // result = v1
    instructions.push(Instruction::Copy {
        src: v1,
        dst: result.clone(),
    });

    // Jump(end)
    instructions.push(Instruction::Jump {
        target: label_end.clone(),
    });

    // Label(e2_label)
    instructions.push(Instruction::Label(label_e2));

    // <instructions to calculate e2>
    let val_e2 = emit_expression(e2, instructions, id_gen, symbol_table);

    // v2 = <result of e2>
    instructions.push(Instruction::Copy {
        src: val_e2,
        dst: v2.clone(),
    });

    // result = v2
    instructions.push(Instruction::Copy {
        src: v2,
        dst: result.clone(),
    });

    // Label(end)
    instructions.push(Instruction::Label(label_end));

    result
}

fn start_label<T: AsRef<str>>(label: T) -> Identifier {
    format!("start_{}", label.as_ref()).into()
}

fn break_label<T: AsRef<str>>(label: T) -> Identifier {
    format!("break_{}", label.as_ref()).into()
}

fn continue_label<T: AsRef<str>>(label: T) -> Identifier {
    format!("continue_{}", label.as_ref()).into()
}

fn emit_jump(label: &Identifier) -> Instruction {
    Instruction::Jump {
        target: label.clone(),
    }
}

fn emit_do_while(
    body: &ast_c::TypedStatement,
    condition: &ast_c::TypedExpression,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for do-while");
    let start_label = start_label(loop_label.clone());
    instructions.push(Instruction::Label(start_label.clone()));

    // loop body
    let val = emit_statement(body, instructions, id_gen, symbol_table);
    if let Some(val) = val {
        instructions.push(val);
    }

    // Continue label
    instructions.push(Instruction::Label(continue_label(loop_label.clone())));

    // evaluate condition and compare to zero
    let v = emit_expression(condition, instructions, id_gen, symbol_table);

    // conditionally jump to beginning of loop
    instructions.push(Instruction::JumpIfNotZero {
        condition: v,
        target: start_label,
    });

    // Break label
    instructions.push(Instruction::Label(break_label(loop_label.clone())));

    None
}

fn emit_while(
    condition: &ast_c::TypedExpression,
    body: &ast_c::TypedStatement,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for do-while");
    let continue_label = continue_label(loop_label.clone());
    let break_label = break_label(loop_label.clone());

    instructions.push(Instruction::Label(continue_label.clone()));

    // evaluate condition and compare to zero
    let v = emit_expression(condition, instructions, id_gen, symbol_table);

    // conditionally jump to end of loop
    instructions.push(Instruction::JumpIfZero {
        condition: v,
        target: break_label.clone(),
    });

    // loop body
    let val = emit_statement(body, instructions, id_gen, symbol_table);
    if let Some(val) = val {
        instructions.push(val);
    }

    // Jump to continue label
    instructions.push(emit_jump(&continue_label));

    // Break label
    instructions.push(Instruction::Label(break_label));

    None
}

#[allow(clippy::too_many_arguments)]
fn emit_for(
    init: &ast_c::TypedForInit,
    condition: &Option<ast_c::TypedExpression>,
    post: &Option<ast_c::TypedExpression>,
    body: &ast_c::TypedStatement,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
    symbol_table: &mut SymbolTable,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for for-loop");
    let start_label = start_label(loop_label.clone());
    let continue_label = continue_label(loop_label.clone());
    let break_label = break_label(loop_label.clone());

    // Emit initialization
    if let ast_c::TypedForInit::InitExp(Some(exp)) = init {
        let v = emit_expression(exp, instructions, id_gen, symbol_table);
        let dst = make_tacky_variable(exp.get_type(), id_gen, symbol_table);
        instructions.push(Instruction::Copy { src: v, dst });
    } else if let ast_c::TypedForInit::InitDecl(decl) = init {
        let _ = emit_variable_declaration(decl, instructions, id_gen, symbol_table);
    }

    instructions.push(Instruction::Label(start_label.clone()));

    // Evaluate condition if present
    if let Some(cond) = condition {
        let v = emit_expression(cond, instructions, id_gen, symbol_table);
        instructions.push(Instruction::JumpIfZero {
            condition: v,
            target: break_label.clone(),
        });
    }

    // Loop body
    let val = emit_statement(body, instructions, id_gen, symbol_table);
    if let Some(val) = val {
        instructions.push(val);
    }

    instructions.push(Instruction::Label(continue_label.clone()));

    // Post-expression
    if let Some(post_exp) = post {
        let v = emit_expression(post_exp, instructions, id_gen, symbol_table);
        let dst = make_tacky_variable(post_exp.get_type(), id_gen, symbol_table);
        instructions.push(Instruction::Copy { src: v, dst });
    }

    // Jump to start label
    instructions.push(emit_jump(&start_label));

    // Break label
    instructions.push(Instruction::Label(break_label));

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typecheck::type_checking;

    #[test]
    fn test_emit_tacky_constant_expression() {
        let exp =
            ast_c::TypedExpression(ast_c::Type::Int, ast_c::InnerTypedExpression::constant(2));
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let mut symbol_table = SymbolTable::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen, &mut symbol_table),
            Val::constant(2)
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_emit_tacky_unary_expression() {
        let exp = ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Unary(
                ast_c::UnaryOperator::Complement,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(2),
                )),
            ),
        );
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let mut symbol_table = SymbolTable::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen, &mut symbol_table),
            Val::var("tmp.0")
        );
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Complement,
                src: Val::constant(2),
                dst: Val::var("tmp.0"),
            },]
        );
    }

    #[test]
    fn test_emit_tacky_nested_unary_expression() {
        let exp = ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Unary(
                ast_c::UnaryOperator::Negate,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Unary(
                        ast_c::UnaryOperator::Complement,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::Unary(
                                ast_c::UnaryOperator::Negate,
                                Box::new(ast_c::TypedExpression(
                                    ast_c::Type::Int,
                                    ast_c::InnerTypedExpression::constant(8),
                                )),
                            ),
                        )),
                    ),
                )),
            ),
        );
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let mut symbol_table = SymbolTable::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen, &mut symbol_table),
            Val::var("tmp.2")
        );
        assert_eq!(
            instructions,
            vec![
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::constant(8),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Unary {
                    op: UnaryOperator::Complement,
                    src: Val::var("tmp.0"),
                    dst: Val::var("tmp.1"),
                },
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::var("tmp.1"),
                    dst: Val::var("tmp.2"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_statement_return_constant() {
        let (ins, instructions) = do_emit_statement(&ast_c::TypedStatement::Return(
            ast_c::TypedExpression(ast_c::Type::Int, ast_c::InnerTypedExpression::constant(2)),
        ));

        assert_eq!(ins, Some(Instruction::Return(Val::constant(2))));
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_emit_statement_return_unary() {
        let (ins, instructions) =
            do_emit_statement(&ast_c::TypedStatement::Return(ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::Unary(
                    ast_c::UnaryOperator::Negate,
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(2),
                    )),
                ),
            )));

        assert_eq!(ins, Some(Instruction::Return(Val::var("tmp.0"))));
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Negate,
                src: Val::constant(2),
                dst: Val::var("tmp.0"),
            }]
        );
    }

    #[test]
    fn test_emit_statement_expression() {
        let (ins, instructions) =
            do_emit_statement(&ast_c::TypedStatement::Expression(ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::Binary(
                    ast_c::BinaryOperator::Add,
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(1),
                    )),
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(2),
                    )),
                ),
            )));

        // No return value for expression statement
        assert!(ins.is_none());

        // But the expression is still evaluated
        assert_eq!(
            instructions,
            vec![Instruction::Binary {
                op: BinaryOperator::Add,
                src1: Val::constant(1),
                src2: Val::constant(2),
                dst: Val::var("tmp.0"),
            }]
        );
    }

    #[test]
    fn test_parse_program_return_unary_nested() {
        // int main(void) { return -(~(-8)); }
        let program = ast_c::Program {
            declarations: vec![ast_c::Declaration::FunDecl(ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(ast_c::Block {
                    items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                        ast_c::Expression::Unary(
                            ast_c::UnaryOperator::Negate,
                            Box::new(ast_c::Expression::Unary(
                                ast_c::UnaryOperator::Complement,
                                Box::new(ast_c::Expression::Unary(
                                    ast_c::UnaryOperator::Negate,
                                    Box::new(ast_c::Expression::constant(8)),
                                )),
                            )),
                        ),
                    ))],
                }),
                fun_type: ast_c::Type::Function {
                    params: vec![],
                    ret: ast_c::Type::Int.into(),
                },
                storage_class: None,
            })],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();

        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![TopLevel::Function(Function {
                    name: "main".into(),
                    global: true,
                    params: vec![],
                    body: vec![
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::constant(8),
                            dst: Val::var("tmp.0"),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Complement,
                            src: Val::var("tmp.0"),
                            dst: Val::var("tmp.1"),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::var("tmp.1"),
                            dst: Val::var("tmp.2"),
                        },
                        Instruction::Return(Val::var("tmp.2")),
                        // Default Return(0)
                        Instruction::Return(Val::constant(0)),
                    ]
                })]
            }
        );
    }

    #[test]
    fn test_parse_program_return_binary() {
        // int main(void) { return 1 * 2 - 3 * (4 + 5); }   // -25
        let program = ast_c::Program {
            declarations: vec![ast_c::Declaration::FunDecl(ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(ast_c::Block {
                    items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                        ast_c::Expression::Binary(
                            ast_c::BinaryOperator::Subtract,
                            Box::new(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Multiply,
                                Box::new(ast_c::Expression::constant(1)),
                                Box::new(ast_c::Expression::constant(2)),
                            )),
                            Box::new(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Multiply,
                                Box::new(ast_c::Expression::constant(3)),
                                Box::new(ast_c::Expression::Binary(
                                    ast_c::BinaryOperator::Add,
                                    Box::new(ast_c::Expression::constant(4)),
                                    Box::new(ast_c::Expression::constant(5)),
                                )),
                            )),
                        ),
                    ))],
                }),
                fun_type: ast_c::Type::Function {
                    params: vec![],
                    ret: ast_c::Type::Int.into(),
                },
                storage_class: None,
            })],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();

        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![TopLevel::Function(Function {
                    name: "main".into(),
                    global: true,
                    params: vec![],
                    body: vec![
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::constant(1),
                            src2: Val::constant(2),
                            dst: Val::var("tmp.0"),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            src1: Val::constant(4),
                            src2: Val::constant(5),
                            dst: Val::var("tmp.1"),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::constant(3),
                            src2: Val::var("tmp.1"),
                            dst: Val::var("tmp.2"),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Subtract,
                            src1: Val::var("tmp.0"),
                            src2: Val::var("tmp.2"),
                            dst: Val::var("tmp.3"),
                        },
                        Instruction::Return(Val::var("tmp.3")),
                        // Default Return(0)
                        Instruction::Return(Val::constant(0)),
                    ],
                })]
            }
        );
    }

    fn do_emit_tacky(exp: &ast_c::TypedExpression) -> (Val, Vec<Instruction>) {
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let mut symbol_table = SymbolTable::new();
        let val = emit_expression(exp, &mut instructions, &mut id_gen, &mut symbol_table);
        (val, instructions)
    }

    fn do_emit_statement(stmt: &ast_c::TypedStatement) -> (Option<Instruction>, Vec<Instruction>) {
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let mut symbol_table = SymbolTable::new();
        let ins = emit_statement(stmt, &mut instructions, &mut id_gen, &mut symbol_table);
        (ins, instructions)
    }

    #[test]
    fn test_emit_tacky_unary_not() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Unary(
                ast_c::UnaryOperator::Not,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(1),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.0"));
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Not,
                src: Val::constant(1),
                dst: Val::var("tmp.0"),
            },]
        );
    }

    #[test]
    fn test_emit_tacky_binary_and() {
        // "e1 && e2" generates:
        //   <instructions for e1>
        //   v1 = <result of e1>
        //   JumpIfZero(v1, false_label)
        //   <instructions for e2>
        //   v2 = <result of e2>
        //   JumpIfZero(v2, false_label)
        //   Copy(1, result)
        //   Jump(end)
        //   Label(false_label)
        //   Copy(0, result)
        //   Label(end)
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::And,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.2"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.1"), // v1
                },
                Instruction::JumpIfZero {
                    condition: Val::var("tmp.1"),
                    target: "and_false.0".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::constant(3), // v2
                    target: "and_false.0".into(),
                },
                Instruction::Copy {
                    src: Val::constant(1),
                    dst: Val::var("tmp.2"), // result
                },
                Instruction::Jump {
                    target: "and_end.0".into(),
                },
                Instruction::Label("and_false.0".into()),
                Instruction::Copy {
                    src: Val::constant(0),
                    dst: Val::var("tmp.2"), // result
                },
                Instruction::Label("and_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_or() {
        // "e1 || e2" generates:
        //   <instructions for e1>
        //   v1 = <result of e1>
        //   JumpIfNotZero(v1, true_label)
        //   <instructions for e2>
        //   v2 = <result of e2>
        //   JumpIfNotZero(v2, true_label)
        //   Copy(0, result)
        //   Jump(end)
        //   Label(true_label)
        //   Copy(1, result)
        //   Label(end)
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::Or,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.2"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.1"), // v1
                },
                Instruction::JumpIfNotZero {
                    condition: Val::var("tmp.1"),
                    target: "or_true.0".into(),
                },
                Instruction::JumpIfNotZero {
                    condition: Val::constant(3), // v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::constant(0),
                    dst: Val::var("tmp.2"), // result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::constant(1),
                    dst: Val::var("tmp.2"), // result
                },
                Instruction::Label("or_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_or_and() {
        //   e1 || e2 && e3
        // Equivalent to:
        //   e1 || (e2 && e3)
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::Or,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(1),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::And,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(3),
                        )),
                    ),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.3"));
        assert_eq!(
            instructions,
            vec![
                Instruction::JumpIfNotZero {
                    condition: Val::constant(1), // OR v1
                    target: "or_true.0".into(),
                },
                // AND
                Instruction::JumpIfZero {
                    condition: Val::constant(2), // AND v1
                    target: "and_false.1".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::constant(3), // AND v2
                    target: "and_false.1".into(),
                },
                Instruction::Copy {
                    src: Val::constant(1),
                    dst: Val::var("tmp.2"), // AND result
                },
                Instruction::Jump {
                    target: "and_end.1".into(),
                },
                Instruction::Label("and_false.1".into()),
                Instruction::Copy {
                    src: Val::constant(0),
                    dst: Val::var("tmp.2"), // AND result
                },
                Instruction::Label("and_end.1".into()),
                // back to OR
                Instruction::JumpIfNotZero {
                    condition: Val::var("tmp.2"), // OR v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::constant(0),
                    dst: Val::var("tmp.3"), // final result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::constant(1),
                    dst: Val::var("tmp.3"), // final result
                },
                Instruction::Label("or_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::Equal,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::Equal,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_not_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::NotEqual,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::NotEqual,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::LessThan,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessThan,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::GreaterThan,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterThan,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::LessOrEqual,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessOrEqual,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Binary(
                ast_c::BinaryOperator::GreaterOrEqual,
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::Binary(
                        ast_c::BinaryOperator::Add,
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(1),
                        )),
                        Box::new(ast_c::TypedExpression(
                            ast_c::Type::Int,
                            ast_c::InnerTypedExpression::constant(2),
                        )),
                    ),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        assert_eq!(val, Val::var("tmp.1"));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.0"),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterOrEqual,
                    src1: Val::var("tmp.0"),
                    src2: Val::constant(3),
                    dst: Val::var("tmp.1"),
                },
            ]
        );
    }

    #[test]
    fn test_parse_program_local_variables() {
        // Listing 5-13, page 111 (modified)
        // int main(void) {
        //     int b;
        //     int a = 10 + 1;
        //     b = a * 2;
        //     int c = 42;  // add a local variable with constant init
        //     return b;
        // }
        let program = ast_c::Program {
            declarations: vec![ast_c::Declaration::FunDecl(ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(ast_c::Block {
                    items: vec![
                        // int b;
                        ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                            name: "b.98".into(),
                            init: None,
                            var_type: ast_c::Type::Int,
                            storage_class: None,
                        })),
                        // int a = 10 + 1;
                        ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                            name: "a.99".into(),
                            init: Some(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Add,
                                Box::new(ast_c::Expression::constant(10)),
                                Box::new(ast_c::Expression::constant(1)),
                            )),
                            var_type: ast_c::Type::Int,
                            storage_class: None,
                        })),
                        // b = a * 2;
                        ast_c::BlockItem::S(ast_c::Statement::Expression(
                            ast_c::Expression::Assignment(
                                Box::new(ast_c::Expression::var("b.98")),
                                Box::new(ast_c::Expression::Binary(
                                    ast_c::BinaryOperator::Multiply,
                                    Box::new(ast_c::Expression::var("a.99")),
                                    Box::new(ast_c::Expression::constant(2)),
                                )),
                            ),
                        )),
                        // int c = 42;
                        ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                            name: "c.100".into(),
                            init: Some(ast_c::Expression::constant(42)),
                            var_type: ast_c::Type::Int,
                            storage_class: None,
                        })),
                        // return b;
                        ast_c::BlockItem::S(ast_c::Statement::Return(ast_c::Expression::var(
                            "b.98",
                        ))),
                    ],
                }),
                fun_type: ast_c::Type::Function {
                    params: vec![],
                    ret: ast_c::Type::Int.into(),
                },
                storage_class: None,
            })],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();

        // Listing 5-14: Expected TACKY:
        //   tmp.2 = 10 + 1
        //   a.1 = tmp.2
        //   tmp.3 = a.1 * 2
        //   b.0 = tmp.3
        //   tmp.4 = 42
        //   c.0 = tmp.4
        //   Return(b.0)
        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![TopLevel::Function(Function {
                    name: "main".into(),
                    global: true,
                    params: vec![],
                    body: vec![
                        // int b;
                        // NO TACKY

                        // int a = 10 + 1;
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            src1: Val::constant(10),
                            src2: Val::constant(1),
                            dst: Val::var("tmp.0"),
                        },
                        Instruction::Copy {
                            src: Val::var("tmp.0"),
                            dst: Val::var("a.99"),
                        },
                        // b = a * 2;
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::var("a.99"),
                            src2: Val::constant(2),
                            dst: Val::var("tmp.1"),
                        },
                        Instruction::Copy {
                            src: Val::var("tmp.1"),
                            dst: Val::var("b.98"),
                        },
                        // int c = 42;
                        Instruction::Copy {
                            src: Val::constant(42),
                            dst: Val::var("c.100"),
                        },
                        // return b;
                        Instruction::Return(Val::var("b.98")),
                        // Default Return(0)
                        Instruction::Return(Val::constant(0)),
                    ],
                })]
            }
        );
    }

    #[test]
    fn test_emit_statement_if() {
        // if (1 + 2) {
        //     return 1;
        // }
        let (ins, instructions) = do_emit_statement(&ast_c::TypedStatement::If {
            condition: ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::Binary(
                    ast_c::BinaryOperator::Add,
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(1),
                    )),
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(2),
                    )),
                ),
            ),
            then_block: Box::new(ast_c::TypedStatement::Return(ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::constant(1),
            ))),
            else_block: None,
        });

        // Expected TACKY:
        //   <instructions for condition>
        //   c = <result of condition>
        //   JumpIfZero(c, end)
        //   <instructions for statement>
        //   Label(end)
        assert_eq!(ins, None);
        assert_eq!(
            instructions,
            vec![
                // <instructions for condition>
                // c = <result of condition>
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.1"), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::var("tmp.1"),
                    target: "if_end.0".into(),
                },
                // instructions for statement
                Instruction::Return(Val::constant(1)),
                // Label(end)
                Instruction::Label("if_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_statement_if_else() {
        // if (1 + 2) {
        //     return 1;
        // } else {
        //     return 2;
        // }
        let (ins, instructions) = do_emit_statement(&ast_c::TypedStatement::If {
            condition: ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::Binary(
                    ast_c::BinaryOperator::Add,
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(1),
                    )),
                    Box::new(ast_c::TypedExpression(
                        ast_c::Type::Int,
                        ast_c::InnerTypedExpression::constant(2),
                    )),
                ),
            ),
            then_block: Box::new(ast_c::TypedStatement::Return(ast_c::TypedExpression(
                ast_c::Type::Int,
                ast_c::InnerTypedExpression::constant(1),
            ))),
            else_block: Some(Box::new(ast_c::TypedStatement::Return(
                ast_c::TypedExpression(ast_c::Type::Int, ast_c::InnerTypedExpression::constant(2)),
            ))),
        });

        // Expected TACKY:
        //   <instructions for condition>
        //   c = <result of condition>
        //   JumpIfZero(c, else_label)
        //   <instructions for then-statement>
        //   Jump(end)
        //   Label(else_label)
        //   <instructions for else-statement2>
        //   Label(end)
        assert_eq!(ins, None);
        assert_eq!(
            instructions,
            vec![
                // <instructions for condition>
                // c = <result of condition>
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::constant(1),
                    src2: Val::constant(2),
                    dst: Val::var("tmp.1"), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::var("tmp.1"),
                    target: "if_else.0".into(),
                },
                // instructions for then-statement
                Instruction::Return(Val::constant(1)),
                Instruction::Jump {
                    target: "if_end.0".into(),
                },
                // Label(else)
                Instruction::Label("if_else.0".into()),
                // instructions for else-statement
                Instruction::Return(Val::constant(2)),
                // Label(end)
                Instruction::Label("if_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_conditional() {
        // Page 127
        // 1 ? 2 : 3
        let (val, instructions) = do_emit_tacky(&ast_c::TypedExpression(
            ast_c::Type::Int,
            ast_c::InnerTypedExpression::Conditional(
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(1),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(2),
                )),
                Box::new(ast_c::TypedExpression(
                    ast_c::Type::Int,
                    ast_c::InnerTypedExpression::constant(3),
                )),
            ),
        ));

        // Expected TACKY (Listing 6-14):
        //   <instructions for condition>
        //   c = <result of condition>
        //   JumpIfZero(c, e2_label)
        //   <instructions to calculate e1>
        //   v1 = <result of e1>
        //   result = v1
        //   Jump(end)
        //   Label(e2_label)
        //   <instructions to calculate e2>
        //   v2 = <result of e2>
        //   result = v2
        //   Label(end)
        assert_eq!(val, Val::var("tmp.4")); // result
        assert_eq!(
            instructions,
            vec![
                // <instructions for condition>
                // c = <result of condition>
                Instruction::Copy {
                    src: Val::constant(1),
                    dst: Val::var("tmp.1"), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::var("tmp.1"),
                    target: "cond_e2.0".into(),
                },
                // instructions for e1-expression
                // v1 = <result of e1>
                Instruction::Copy {
                    src: Val::constant(2),
                    dst: Val::var("tmp.2"), // v1
                },
                // result = v1
                Instruction::Copy {
                    src: Val::var("tmp.2"),
                    dst: Val::var("tmp.4"), // result
                },
                Instruction::Jump {
                    target: "cond_end.0".into(),
                },
                // Label(e2)
                Instruction::Label("cond_e2.0".into()),
                // instructions for e2-expression
                // v2 = <result of e2>
                Instruction::Copy {
                    src: Val::constant(3),
                    dst: Val::var("tmp.3"), // v2
                },
                // result = v2
                Instruction::Copy {
                    src: Val::var("tmp.3"),
                    dst: Val::var("tmp.4"), // result
                },
                // Label(end)
                Instruction::Label("cond_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_goto() {
        // int main(void) {
        //     goto label1;
        //     label0: return 0;
        //     label1: return 1;
        //     label2: return 2;
        // }
        let program = ast_c::Program {
            declarations: vec![ast_c::Declaration::FunDecl(ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(ast_c::Block {
                    items: vec![
                        ast_c::BlockItem::S(ast_c::Statement::Goto("label1".into())),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label0".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::constant(0),
                            )),
                        }),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::constant(1),
                            )),
                        }),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label2".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::constant(2),
                            )),
                        }),
                    ],
                }),
                fun_type: ast_c::Type::Function {
                    params: vec![],
                    ret: ast_c::Type::Int.into(),
                },
                storage_class: None,
            })],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();

        // Listing 5-14: Expected TACKY:
        //   Jump(label1)
        //   .label0
        //     Return(0)
        //   .label1
        //     Return(1)
        //   .label2
        //     Return(2)
        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![TopLevel::Function(Function {
                    name: "main".into(),
                    global: true,
                    params: vec![],
                    body: vec![
                        Instruction::Jump {
                            target: "label1".into(),
                        },
                        Instruction::Label("label0".into()),
                        Instruction::Return(Val::constant(0)),
                        Instruction::Label("label1".into()),
                        Instruction::Return(Val::constant(1)),
                        Instruction::Label("label2".into()),
                        Instruction::Return(Val::constant(2)),
                        // Default Return(0)
                        Instruction::Return(Val::constant(0)),
                    ],
                })]
            }
        );
    }

    #[test]
    fn test_function_call() {
        // int foo(int a, int b) {
        //     return a + b;
        // }
        //
        // int bar(int a);
        //
        // int main(void) {
        //     return foo(42, 77);
        // }
        let program = ast_c::Program {
            declarations: vec![
                ast_c::Declaration::FunDecl(ast_c::FunDecl {
                    name: "foo".into(),
                    params: vec!["a".into(), "b".into()],
                    body: Some(ast_c::Block {
                        items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                            ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Add,
                                Box::new(ast_c::Expression::var("a")),
                                Box::new(ast_c::Expression::var("b")),
                            ),
                        ))],
                    }),
                    fun_type: ast_c::Type::Function {
                        params: vec![ast_c::Type::Int, ast_c::Type::Int],
                        ret: ast_c::Type::Int.into(),
                    },
                    storage_class: None,
                }),
                ast_c::Declaration::FunDecl(ast_c::FunDecl {
                    name: "bar".into(),
                    params: vec!["a".into()],
                    body: None,
                    fun_type: ast_c::Type::Function {
                        params: vec![ast_c::Type::Int],
                        ret: ast_c::Type::Int.into(),
                    },
                    storage_class: None,
                }),
                ast_c::Declaration::FunDecl(ast_c::FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(ast_c::Block {
                        items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                            ast_c::Expression::FunctionCall(
                                "foo".into(),
                                vec![
                                    ast_c::Expression::constant(42),
                                    ast_c::Expression::constant(77),
                                ],
                            ),
                        ))],
                    }),
                    fun_type: ast_c::Type::Function {
                        params: vec![],
                        ret: ast_c::Type::Int.into(),
                    },
                    storage_class: None,
                }),
            ],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();
        //dbg!(&typed_program);

        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![
                    TopLevel::Function(Function {
                        name: "foo".into(),
                        global: true,
                        params: vec!["a".into(), "b".into()],
                        body: vec![
                            Instruction::Binary {
                                op: BinaryOperator::Add,
                                src1: Val::var("a"),
                                src2: Val::var("b"),
                                dst: Val::var("tmp.0"),
                            },
                            Instruction::Return(Val::var("tmp.0")),
                            // Default Return(0)
                            Instruction::Return(Val::constant(0)),
                        ],
                    }),
                    TopLevel::Function(Function {
                        name: "main".into(),
                        global: true,
                        params: vec![],
                        body: vec![
                            Instruction::FunCall {
                                name: "foo".into(),
                                args: vec![Val::constant(42), Val::constant(77),],
                                dst: Val::var("tmp.1"),
                            },
                            Instruction::Return(Val::var("tmp.1")),
                            // Default Return(0)
                            Instruction::Return(Val::constant(0)),
                        ],
                    })
                ]
            }
        );
    }

    #[test]
    fn test_static_and_extern_variables() {
        // extern int a;
        // static int b = 42;
        // int c;
        //
        // int main(void) {
        //     static int d = 77;
        //     extern int e;
        //     return 0;
        // }
        let program = ast_c::Program {
            declarations: vec![
                ast_c::Declaration::VarDecl(ast_c::VarDecl {
                    name: "a".into(),
                    init: None,
                    var_type: ast_c::Type::Int,
                    storage_class: Some(ast_c::StorageClass::Extern),
                }),
                ast_c::Declaration::VarDecl(ast_c::VarDecl {
                    name: "b".into(),
                    init: Some(ast_c::Expression::constant(42)),
                    var_type: ast_c::Type::Int,
                    storage_class: Some(ast_c::StorageClass::Static),
                }),
                ast_c::Declaration::VarDecl(ast_c::VarDecl {
                    name: "c".into(),
                    init: Some(ast_c::Expression::constant(0)),
                    var_type: ast_c::Type::Int,
                    storage_class: Some(ast_c::StorageClass::Extern),
                }),
                ast_c::Declaration::FunDecl(ast_c::FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(ast_c::Block {
                        items: vec![
                            ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                                name: "d".into(),
                                init: Some(ast_c::Expression::constant(77)),
                                var_type: ast_c::Type::Int,
                                storage_class: Some(ast_c::StorageClass::Static),
                            })),
                            ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                                name: "e".into(),
                                init: None,
                                var_type: ast_c::Type::Int,
                                storage_class: Some(ast_c::StorageClass::Extern),
                            })),
                            ast_c::BlockItem::S(ast_c::Statement::Return(
                                ast_c::Expression::constant(0),
                            )),
                        ],
                    }),
                    fun_type: ast_c::Type::Function {
                        params: vec![],
                        ret: ast_c::Type::Int.into(),
                    },
                    storage_class: None,
                }),
            ],
        };
        let (typed_program, mut symbol_table) = type_checking(&program).unwrap();

        // Expected TACKY:
        assert_eq!(
            emit_program(&typed_program, &mut symbol_table).unwrap(),
            Program {
                top_level: vec![
                    TopLevel::Function(Function {
                        name: "main".into(),
                        global: true,
                        params: vec![],
                        body: vec![
                            Instruction::Return(Val::constant(0)),
                            // Default Return(0)
                            Instruction::Return(Val::constant(0)),
                        ],
                    }),
                    // a is extern
                    TopLevel::StaticVariable {
                        name: "b".into(),
                        global: false, // static, so not globally visible
                        t: ast_c::Type::Int,
                        init: StaticInit::IntInit(42),
                    },
                    TopLevel::StaticVariable {
                        name: "c".into(),
                        global: true,
                        t: ast_c::Type::Int,
                        init: StaticInit::IntInit(0), // tentative, set to zero
                    },
                    TopLevel::StaticVariable {
                        name: "d".into(),
                        global: false,
                        t: ast_c::Type::Int,
                        init: StaticInit::IntInit(77),
                    },
                    // e is extern
                ]
            }
        );
    }
}
