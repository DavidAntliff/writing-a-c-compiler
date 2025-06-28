//! TACKY AST
//!
//! ASDL:
//!   program = Program(function_definition*)
//!   function_definition = Function(identifier name, identifier* params, instruction* body)
//!   instruction = Return(val)
//!               | Unary(unary_operator, val src, val dst)
//!               | Binary(binary_operator, val src1, val src2, val dst)
//!               | Copy(val src, val dst)
//!               | Jump(identifier target)
//!               | JumpIfZero(val condition, identifier target)
//!               | JumpIfNotZero(val condition, identifier target)
//!               | Label(identifier)
//!               | FunCall(identifier name, val* args, val dst)
//!   val = Constant(int) | Var(identifier)
//!   unary_operator = Complement | Negate | Not
//!   binary_operator = Add | Subtract | Multiply | Divide | Remainder
//!                   | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight
//!                   | Equal | NotEqual | LessThan | LessOrEqual | GreaterThan | GreaterOrEqual
//!

use crate::ast_c;
use crate::id_gen::IdGenerator;
use thiserror::Error;

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
    pub(crate) function_definitions: Vec<FunctionDefinition>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: Identifier,
    pub(crate) params: Vec<Identifier>,
    pub(crate) body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Instruction {
    Return(Val),
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
    Constant(usize),
    Var(Identifier),
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

pub(crate) fn emit_program(program: &ast_c::Program) -> Result<Program, TackyError> {
    let mut function_definitions = vec![];
    for fun_decl in &program.function_declarations {
        let fun_def = emit_function_definition(fun_decl);
        if let Some(fun_def) = fun_def {
            function_definitions.push(fun_def);
        }
    }

    Ok(Program {
        function_definitions,
    })
}

fn emit_function_definition(function_decl: &ast_c::FunDecl) -> Option<FunctionDefinition> {
    // Drop function declarations without a body
    let body = match &function_decl.body {
        None => {
            return None;
        }
        Some(body) => body,
    };

    let name: Identifier = (&function_decl.name).into();

    let mut instructions = vec![];
    let mut id_gen = IdGenerator::new();

    let params = function_decl
        .params
        .iter()
        .map(|p| Identifier(p.clone()))
        .collect::<Vec<_>>();

    let val = emit_block(body, &mut instructions, &mut id_gen);
    assert!(val.is_none(), "emit_block should return None");

    // Add a Return(0) to the end of all functions (see page 112)
    instructions.push(Instruction::Return(Val::Constant(0)));

    Some(FunctionDefinition {
        name,
        params,
        body: instructions,
    })
}

fn next_var(id_gen: &mut IdGenerator) -> Identifier {
    format!("tmp.{}", id_gen.next()).into()
}

// Previously called "emit_tacky"
fn emit_expression(
    exp: &ast_c::Expression,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Val {
    match exp {
        ast_c::Expression::Constant(c) => Val::Constant(*c),

        ast_c::Expression::Var(identifier) => Val::Var(identifier.into()),

        ast_c::Expression::Unary(op, inner) => {
            let src = emit_expression(inner, instructions, id_gen);
            let dst = Val::Var(next_var(id_gen));
            let tacky_op = convert_unop(op);
            instructions.push(Instruction::Unary {
                op: tacky_op,
                src,
                dst: dst.clone(),
            });
            dst
        }

        // Handle short-circuit evaluation for And / Or
        ast_c::Expression::Binary(ast_c::BinaryOperator::And, e1, e2) => {
            let id = id_gen.next();
            let label_false: Identifier = format!("and_false.{id}").into();
            let label_end: Identifier = format!("and_end.{id}").into();

            let v1 = emit_expression(e1, instructions, id_gen);
            instructions.push(Instruction::JumpIfZero {
                condition: v1.clone(),
                target: label_false.clone(),
            });
            let v2 = emit_expression(e2, instructions, id_gen);
            instructions.push(Instruction::JumpIfZero {
                condition: v2.clone(),
                target: label_false.clone(),
            });
            let dst = Val::Var(next_var(id_gen));
            instructions.push(Instruction::Copy {
                src: Val::Constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_false));
            instructions.push(Instruction::Copy {
                src: Val::Constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::Expression::Binary(ast_c::BinaryOperator::Or, e1, e2) => {
            let id = id_gen.next();
            let label_true: Identifier = format!("or_true.{id}").into();
            let label_end: Identifier = format!("or_end.{id}").into();

            let v1 = emit_expression(e1, instructions, id_gen);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v1.clone(),
                target: label_true.clone(),
            });
            let v2 = emit_expression(e2, instructions, id_gen);
            instructions.push(Instruction::JumpIfNotZero {
                condition: v2.clone(),
                target: label_true.clone(),
            });
            let dst = Val::Var(next_var(id_gen));
            instructions.push(Instruction::Copy {
                src: Val::Constant(0),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Jump {
                target: label_end.clone(),
            });
            instructions.push(Instruction::Label(label_true));
            instructions.push(Instruction::Copy {
                src: Val::Constant(1),
                dst: dst.clone(),
            });
            instructions.push(Instruction::Label(label_end));
            dst
        }

        ast_c::Expression::Binary(op, e1, e2) => {
            // Unsequenced - indeterminate order of evaluation
            let src1 = emit_expression(e1, instructions, id_gen);
            let src2 = emit_expression(e2, instructions, id_gen);
            let dst = Val::Var(next_var(id_gen));
            let tacky = convert_binop(op);
            instructions.push(Instruction::Binary {
                op: tacky,
                src1,
                src2,
                dst: dst.clone(),
            });
            dst
        }

        ast_c::Expression::Assignment(lhs, rhs) => {
            if let ast_c::Expression::Var(v) = &**lhs {
                let result = emit_expression(rhs, instructions, id_gen);
                instructions.push(Instruction::Copy {
                    src: result,
                    dst: Val::Var(v.into()),
                });
                Val::Var(v.into())
            } else {
                unreachable!("lhs should be a variable");
            }
        }

        ast_c::Expression::Conditional(cond, e1, e2) => {
            emit_exp_conditional(cond, e1, e2, instructions, id_gen)
        }

        ast_c::Expression::FunctionCall(name, args) => {
            let dst = Val::Var(next_var(id_gen));
            let tacky_args: Vec<Val> = args
                .iter()
                .map(|arg| emit_expression(arg, instructions, id_gen))
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
            panic!("Unsupported binary operator: {:?}", op);
        }
    }
}

fn emit_block(
    block: &ast_c::Block,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    for item in &block.items {
        emit_block_item(item, instructions, id_gen);
    }
    None
}

fn emit_block_item(
    item: &ast_c::BlockItem,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    match item {
        ast_c::BlockItem::S(statement) => {
            if let Some(instruction) = emit_statement(statement, instructions, id_gen) {
                instructions.push(instruction);
            }
        }
        ast_c::BlockItem::D(declaration) => {
            emit_declaration(declaration, instructions, id_gen);
        }
    }
    None
}

fn emit_declaration(
    decl: &ast_c::Declaration,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    match decl {
        ast_c::Declaration::FunDecl(decl) => {
            let def = emit_function_definition(decl);
            assert_eq!(def, None, "Function declaration should not have a body");
        }
        ast_c::Declaration::VarDecl(decl) => {
            emit_variable_declaration(decl, instructions, id_gen);
        }
    }
    None
}

fn emit_variable_declaration(
    ast_c::VarDecl { name, init }: &ast_c::VarDecl,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    if let Some(init) = &init {
        let result = emit_expression(init, instructions, id_gen);
        instructions.push(Instruction::Copy {
            src: result,
            dst: Val::Var(name.clone().into()),
        });
    }
    None
}

fn emit_statement(
    statement: &ast_c::Statement,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    match statement {
        ast_c::Statement::Return(exp) => {
            let val = emit_expression(exp, instructions, id_gen);
            Some(Instruction::Return(val))
        }
        ast_c::Statement::Expression(exp) => {
            let _ = emit_expression(exp, instructions, id_gen);
            // No need to return anything for an expression statement
            None
        }
        ast_c::Statement::If {
            condition,
            then,
            else_,
        } => emit_statement_if(condition, then, else_, instructions, id_gen),
        ast_c::Statement::Labeled { label, statement } => {
            instructions.push(Instruction::Label(label.clone().into()));
            emit_statement(statement, instructions, id_gen)
        }
        ast_c::Statement::Goto(label) => {
            instructions.push(emit_jump(&label.into()));
            None
        }
        ast_c::Statement::Compound(block) => emit_block(block, instructions, id_gen),
        ast_c::Statement::Break(Some(label)) => {
            instructions.push(emit_jump(&break_label(label)));
            None
        }
        ast_c::Statement::Break(None) => panic!("Break without label is not supported in Tacky"),
        ast_c::Statement::Continue(Some(label)) => {
            instructions.push(emit_jump(&continue_label(label)));
            None
        }
        ast_c::Statement::Continue(None) => {
            panic!("Continue without label is not supported in Tacky")
        }
        ast_c::Statement::While {
            condition,
            body,
            loop_label,
        } => emit_while(condition, body, loop_label, instructions, id_gen),
        ast_c::Statement::DoWhile {
            body,
            condition,
            loop_label,
        } => emit_do_while(body, condition, loop_label, instructions, id_gen),
        ast_c::Statement::For {
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
        ),
        ast_c::Statement::Null => None,
    }
}

fn emit_statement_if(
    condition: &ast_c::Expression,
    then: &ast_c::Statement,
    else_: &Option<Box<ast_c::Statement>>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
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

    let label_else: Identifier = format!("if_else.{}", id_gen.next()).into();
    let label_end: Identifier = format!("if_end.{}", id_gen.next()).into();

    // if:
    let cond_val = emit_expression(condition, instructions, id_gen);

    instructions.push(Instruction::JumpIfZero {
        condition: cond_val.clone(),
        target: if else_.is_none() {
            label_end.clone()
        } else {
            label_else.clone()
        },
    });

    // then:
    if let Some(instruction) = emit_statement(then, instructions, id_gen) {
        instructions.push(instruction);
    }

    if let Some(else_stmt) = else_ {
        // Jump to end after "then"
        instructions.push(Instruction::Jump {
            target: label_end.clone(),
        });

        // else:
        instructions.push(Instruction::Label(label_else));

        if let Some(instruction) = emit_statement(else_stmt, instructions, id_gen) {
            instructions.push(instruction);
        }
    }

    instructions.push(Instruction::Label(label_end));

    // No return value for if statements
    None
}

fn emit_exp_conditional(
    condition: &ast_c::Expression,
    e1: &ast_c::Expression,
    e2: &ast_c::Expression,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
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

    let label_e2: Identifier = format!("cond_e2.{}", id_gen.next()).into();
    let label_end: Identifier = format!("cond_end.{}", id_gen.next()).into();

    let c = Val::Var(next_var(id_gen));
    let v1 = Val::Var(next_var(id_gen));
    let v2 = Val::Var(next_var(id_gen));
    let result = Val::Var(next_var(id_gen));

    // <instructions for condition>
    // c = <result of condition>
    let c_val = emit_expression(condition, instructions, id_gen);
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
    let val_e1 = emit_expression(e1, instructions, id_gen);

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
    let val_e2 = emit_expression(e2, instructions, id_gen);

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
    body: &ast_c::Statement,
    condition: &ast_c::Expression,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for do-while");
    let start_label = start_label(loop_label.clone());
    instructions.push(Instruction::Label(start_label.clone()));

    // loop body
    let val = emit_statement(body, instructions, id_gen);
    if let Some(val) = val {
        instructions.push(val);
    }

    // Continue label
    instructions.push(Instruction::Label(continue_label(loop_label.clone())));

    // evaluate condition and compare to zero
    let v = emit_expression(condition, instructions, id_gen);

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
    condition: &ast_c::Expression,
    body: &ast_c::Statement,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for do-while");
    let continue_label = continue_label(loop_label.clone());
    let break_label = break_label(loop_label.clone());

    instructions.push(Instruction::Label(continue_label.clone()));

    // evaluate condition and compare to zero
    let v = emit_expression(condition, instructions, id_gen);

    // conditionally jump to end of loop
    instructions.push(Instruction::JumpIfZero {
        condition: v,
        target: break_label.clone(),
    });

    // loop body
    let val = emit_statement(body, instructions, id_gen);
    if let Some(val) = val {
        instructions.push(val);
    }

    // Jump to continue label
    instructions.push(emit_jump(&continue_label));

    // Break label
    instructions.push(Instruction::Label(break_label));

    None
}

fn emit_for(
    init: &ast_c::ForInit,
    condition: &Option<ast_c::Expression>,
    post: &Option<ast_c::Expression>,
    body: &ast_c::Statement,
    loop_label: &Option<crate::lexer::Identifier>,
    instructions: &mut Vec<Instruction>,
    id_gen: &mut IdGenerator,
) -> Option<Instruction> {
    let loop_label = loop_label
        .clone()
        .expect("Loop label must be provided for for-loop");
    let start_label = start_label(loop_label.clone());
    let continue_label = continue_label(loop_label.clone());
    let break_label = break_label(loop_label.clone());

    // Emit initialization
    if let ast_c::ForInit::InitExp(Some(exp)) = init {
        let v = emit_expression(exp, instructions, id_gen);
        instructions.push(Instruction::Copy {
            src: v,
            dst: Val::Var(next_var(id_gen)),
        });
    } else if let ast_c::ForInit::InitDecl(decl) = init {
        let _ = emit_variable_declaration(decl, instructions, id_gen);
    }

    instructions.push(Instruction::Label(start_label.clone()));

    // Evaluate condition if present
    if let Some(cond) = condition {
        let v = emit_expression(cond, instructions, id_gen);
        instructions.push(Instruction::JumpIfZero {
            condition: v,
            target: break_label.clone(),
        });
    }

    // Loop body
    let val = emit_statement(body, instructions, id_gen);
    if let Some(val) = val {
        instructions.push(val);
    }

    instructions.push(Instruction::Label(continue_label.clone()));

    // Post-expression
    if let Some(post_exp) = post {
        let v = emit_expression(post_exp, instructions, id_gen);
        instructions.push(Instruction::Copy {
            src: v,
            dst: Val::Var(next_var(id_gen)),
        });
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
    use crate::ast_c::Block;

    #[test]
    fn test_emit_tacky_constant_expression() {
        let exp = ast_c::Expression::Constant(2);
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen),
            Val::Constant(2)
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_emit_tacky_unary_expression() {
        let exp = ast_c::Expression::Unary(
            ast_c::UnaryOperator::Complement,
            Box::new(ast_c::Expression::Constant(2)),
        );
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen),
            Val::Var("tmp.0".into())
        );
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Complement,
                src: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            },]
        );
    }

    #[test]
    fn test_emit_tacky_nested_unary_expression() {
        let exp = ast_c::Expression::Unary(
            ast_c::UnaryOperator::Negate,
            Box::new(ast_c::Expression::Unary(
                ast_c::UnaryOperator::Complement,
                Box::new(ast_c::Expression::Unary(
                    ast_c::UnaryOperator::Negate,
                    Box::new(ast_c::Expression::Constant(8)),
                )),
            )),
        );
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();

        assert_eq!(
            emit_expression(&exp, &mut instructions, &mut id_gen),
            Val::Var("tmp.2".into())
        );
        assert_eq!(
            instructions,
            vec![
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::Constant(8),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Unary {
                    op: UnaryOperator::Complement,
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Unary {
                    op: UnaryOperator::Negate,
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_statement_return_constant() {
        let (ins, instructions) =
            do_emit_statement(&ast_c::Statement::Return(ast_c::Expression::Constant(2)));

        assert_eq!(ins, Some(Instruction::Return(Val::Constant(2))));
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_emit_statement_return_unary() {
        let (ins, instructions) =
            do_emit_statement(&ast_c::Statement::Return(ast_c::Expression::Unary(
                ast_c::UnaryOperator::Negate,
                Box::new(ast_c::Expression::Constant(2)),
            )));

        assert_eq!(ins, Some(Instruction::Return(Val::Var("tmp.0".into()))));
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Negate,
                src: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            }]
        );
    }

    #[test]
    fn test_emit_statement_expression() {
        let (ins, instructions) =
            do_emit_statement(&ast_c::Statement::Expression(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )));

        // No return value for expression statement
        assert!(ins.is_none());

        // But the expression is still evaluated
        assert_eq!(
            instructions,
            vec![Instruction::Binary {
                op: BinaryOperator::Add,
                src1: Val::Constant(1),
                src2: Val::Constant(2),
                dst: Val::Var("tmp.0".into()),
            }]
        );
    }

    #[test]
    fn test_parse_program_return_unary_nested() {
        // int main(void) { return -(~(-8)); }
        let program = ast_c::Program {
            function_declarations: vec![ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                        ast_c::Expression::Unary(
                            ast_c::UnaryOperator::Negate,
                            Box::new(ast_c::Expression::Unary(
                                ast_c::UnaryOperator::Complement,
                                Box::new(ast_c::Expression::Unary(
                                    ast_c::UnaryOperator::Negate,
                                    Box::new(ast_c::Expression::Constant(8)),
                                )),
                            )),
                        ),
                    ))],
                }),
            }],
        };

        assert_eq!(
            emit_program(&program).unwrap(),
            Program {
                function_definitions: vec![FunctionDefinition {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::Constant(8),
                            dst: Val::Var("tmp.0".into()),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Complement,
                            src: Val::Var("tmp.0".into()),
                            dst: Val::Var("tmp.1".into()),
                        },
                        Instruction::Unary {
                            op: UnaryOperator::Negate,
                            src: Val::Var("tmp.1".into()),
                            dst: Val::Var("tmp.2".into()),
                        },
                        Instruction::Return(Val::Var("tmp.2".into())),
                        // Default Return(0)
                        Instruction::Return(Val::Constant(0)),
                    ]
                }]
            }
        );
    }

    #[test]
    fn test_parse_program_return_binary() {
        // int main(void) { return 1 * 2 - 3 * (4 + 5); }   // -25
        let program = ast_c::Program {
            function_declarations: vec![ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                        ast_c::Expression::Binary(
                            ast_c::BinaryOperator::Subtract,
                            Box::new(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Multiply,
                                Box::new(ast_c::Expression::Constant(1)),
                                Box::new(ast_c::Expression::Constant(2)),
                            )),
                            Box::new(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Multiply,
                                Box::new(ast_c::Expression::Constant(3)),
                                Box::new(ast_c::Expression::Binary(
                                    ast_c::BinaryOperator::Add,
                                    Box::new(ast_c::Expression::Constant(4)),
                                    Box::new(ast_c::Expression::Constant(5)),
                                )),
                            )),
                        ),
                    ))],
                }),
            }],
        };

        assert_eq!(
            emit_program(&program).unwrap(),
            Program {
                function_definitions: vec![FunctionDefinition {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::Constant(1),
                            src2: Val::Constant(2),
                            dst: Val::Var("tmp.0".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            src1: Val::Constant(4),
                            src2: Val::Constant(5),
                            dst: Val::Var("tmp.1".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::Constant(3),
                            src2: Val::Var("tmp.1".into()),
                            dst: Val::Var("tmp.2".into()),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Subtract,
                            src1: Val::Var("tmp.0".into()),
                            src2: Val::Var("tmp.2".into()),
                            dst: Val::Var("tmp.3".into()),
                        },
                        Instruction::Return(Val::Var("tmp.3".into())),
                        // Default Return(0)
                        Instruction::Return(Val::Constant(0)),
                    ],
                }]
            }
        );
    }

    fn do_emit_tacky(exp: &ast_c::Expression) -> (Val, Vec<Instruction>) {
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let val = emit_expression(exp, &mut instructions, &mut id_gen);
        (val, instructions)
    }

    fn do_emit_statement(stmt: &ast_c::Statement) -> (Option<Instruction>, Vec<Instruction>) {
        let mut instructions = vec![];
        let mut id_gen = IdGenerator::new();
        let ins = emit_statement(stmt, &mut instructions, &mut id_gen);
        (ins, instructions)
    }

    #[test]
    fn test_emit_tacky_unary_not() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Unary(
            ast_c::UnaryOperator::Not,
            Box::new(ast_c::Expression::Constant(1)),
        ));

        assert_eq!(val, Val::Var("tmp.0".into()));
        assert_eq!(
            instructions,
            vec![Instruction::Unary {
                op: UnaryOperator::Not,
                src: Val::Constant(1),
                dst: Val::Var("tmp.0".into()),
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
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::And,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.2".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()), // v1
                },
                Instruction::JumpIfZero {
                    condition: Val::Var("tmp.1".into()),
                    target: "and_false.0".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::Constant(3), // v2
                    target: "and_false.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Jump {
                    target: "and_end.0".into(),
                },
                Instruction::Label("and_false.0".into()),
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // result
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
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Or,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.2".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()), // v1
                },
                Instruction::JumpIfNotZero {
                    condition: Val::Var("tmp.1".into()),
                    target: "or_true.0".into(),
                },
                Instruction::JumpIfNotZero {
                    condition: Val::Constant(3), // v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // result
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
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Or,
            Box::new(ast_c::Expression::Constant(1)),
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::And,
                Box::new(ast_c::Expression::Constant(2)),
                Box::new(ast_c::Expression::Constant(3)),
            )),
        ));

        assert_eq!(val, Val::Var("tmp.3".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::JumpIfNotZero {
                    condition: Val::Constant(1), // OR v1
                    target: "or_true.0".into(),
                },
                // AND
                Instruction::JumpIfZero {
                    condition: Val::Constant(2), // AND v1
                    target: "and_false.1".into(),
                },
                Instruction::JumpIfZero {
                    condition: Val::Constant(3), // AND v2
                    target: "and_false.1".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // AND result
                },
                Instruction::Jump {
                    target: "and_end.1".into(),
                },
                Instruction::Label("and_false.1".into()),
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.2".into()), // AND result
                },
                Instruction::Label("and_end.1".into()),
                // back to OR
                Instruction::JumpIfNotZero {
                    condition: Val::Var("tmp.2".into()), // OR v2
                    target: "or_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.3".into()), // final result
                },
                Instruction::Jump {
                    target: "or_end.0".into(),
                },
                Instruction::Label("or_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.3".into()), // final result
                },
                Instruction::Label("or_end.0".into()),
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::Equal,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::Equal,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_not_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::NotEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::NotEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::LessThan,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessThan,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_than() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::GreaterThan,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterThan,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_less_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::LessOrEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::LessOrEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_emit_tacky_binary_greater_or_equal() {
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Binary(
            ast_c::BinaryOperator::GreaterOrEqual,
            Box::new(ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            )),
            Box::new(ast_c::Expression::Constant(3)),
        ));

        assert_eq!(val, Val::Var("tmp.1".into()));
        assert_eq!(
            instructions,
            vec![
                Instruction::Binary {
                    op: BinaryOperator::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOperator::GreaterOrEqual,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(3),
                    dst: Val::Var("tmp.1".into()),
                },
            ]
        );
    }

    #[test]
    fn test_parse_program_local_variables() {
        // Listing 5-13, page 111
        // int main(void) {
        //     int b;
        //     int a = 10 + 1;
        //     b = a * 2;
        //     return b;
        // }
        let program = ast_c::Program {
            function_declarations: vec![ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        // int b;
                        ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                            name: "b.98".into(),
                            init: None,
                        })),
                        // int a = 10 + 1;
                        ast_c::BlockItem::D(ast_c::Declaration::VarDecl(ast_c::VarDecl {
                            name: "a.99".into(),
                            init: Some(ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Add,
                                Box::new(ast_c::Expression::Constant(10)),
                                Box::new(ast_c::Expression::Constant(1)),
                            )),
                        })),
                        // b = a * 2;
                        ast_c::BlockItem::S(ast_c::Statement::Expression(
                            ast_c::Expression::Assignment(
                                Box::new(ast_c::Expression::Var("b.98".into())),
                                Box::new(ast_c::Expression::Binary(
                                    ast_c::BinaryOperator::Multiply,
                                    Box::new(ast_c::Expression::Var("a.99".into())),
                                    Box::new(ast_c::Expression::Constant(2)),
                                )),
                            ),
                        )),
                        // return b;
                        ast_c::BlockItem::S(ast_c::Statement::Return(ast_c::Expression::Var(
                            "b.98".into(),
                        ))),
                    ],
                }),
            }],
        };

        // Listing 5-14: Expected TACKY:
        //   tmp.2 = 10 + 1
        //   a.1 = tmp.2
        //   tmp.3 = a.1 * 2
        //   b.0 = tmp.3
        //   Return(b.0)
        assert_eq!(
            emit_program(&program).unwrap(),
            Program {
                function_definitions: vec![FunctionDefinition {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        // int b;
                        // NO TACKY

                        // int a = 10 + 1;
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            src1: Val::Constant(10),
                            src2: Val::Constant(1),
                            dst: Val::Var("tmp.0".into()),
                        },
                        Instruction::Copy {
                            src: Val::Var("tmp.0".into()),
                            dst: Val::Var("a.99".into()),
                        },
                        // b = a * 2;
                        Instruction::Binary {
                            op: BinaryOperator::Multiply,
                            src1: Val::Var("a.99".into()),
                            src2: Val::Constant(2),
                            dst: Val::Var("tmp.1".into()),
                        },
                        Instruction::Copy {
                            src: Val::Var("tmp.1".into()),
                            dst: Val::Var("b.98".into()),
                        },
                        // return b;
                        Instruction::Return(Val::Var("b.98".into())),
                        // Default Return(0)
                        Instruction::Return(Val::Constant(0)),
                    ],
                }]
            }
        );
    }

    #[test]
    fn test_emit_statement_if() {
        // if (1 + 2) {
        //     return 1;
        // }
        let (ins, instructions) = do_emit_statement(&ast_c::Statement::If {
            condition: ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            ),
            then: Box::new(ast_c::Statement::Return(ast_c::Expression::Constant(1))),
            else_: None,
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
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.2".into()), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::Var("tmp.2".into()),
                    target: "if_end.1".into(),
                },
                // instructions for statement
                Instruction::Return(Val::Constant(1)),
                // Label(end)
                Instruction::Label("if_end.1".into()),
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
        let (ins, instructions) = do_emit_statement(&ast_c::Statement::If {
            condition: ast_c::Expression::Binary(
                ast_c::BinaryOperator::Add,
                Box::new(ast_c::Expression::Constant(1)),
                Box::new(ast_c::Expression::Constant(2)),
            ),
            then: Box::new(ast_c::Statement::Return(ast_c::Expression::Constant(1))),
            else_: Some(Box::new(ast_c::Statement::Return(
                ast_c::Expression::Constant(2),
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
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.2".into()), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::Var("tmp.2".into()),
                    target: "if_else.0".into(),
                },
                // instructions for then-statement
                Instruction::Return(Val::Constant(1)),
                Instruction::Jump {
                    target: "if_end.1".into(),
                },
                // Label(else)
                Instruction::Label("if_else.0".into()),
                // instructions for else-statement
                Instruction::Return(Val::Constant(2)),
                // Label(end)
                Instruction::Label("if_end.1".into()),
            ]
        );
    }

    #[test]
    fn test_emit_conditional() {
        // Page 127
        // 1 ? 2 : 3
        let (val, instructions) = do_emit_tacky(&ast_c::Expression::Conditional(
            Box::new(ast_c::Expression::Constant(1)),
            Box::new(ast_c::Expression::Constant(2)),
            Box::new(ast_c::Expression::Constant(3)),
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
        assert_eq!(val, Val::Var("tmp.5".into())); // result
        assert_eq!(
            instructions,
            vec![
                // <instructions for condition>
                // c = <result of condition>
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()), // c
                },
                Instruction::JumpIfZero {
                    condition: Val::Var("tmp.2".into()),
                    target: "cond_e2.0".into(),
                },
                // instructions for e1-expression
                // v1 = <result of e1>
                Instruction::Copy {
                    src: Val::Constant(2),
                    dst: Val::Var("tmp.3".into()), // v1
                },
                // result = v1
                Instruction::Copy {
                    src: Val::Var("tmp.3".into()),
                    dst: Val::Var("tmp.5".into()), // result
                },
                Instruction::Jump {
                    target: "cond_end.1".into(),
                },
                // Label(e2)
                Instruction::Label("cond_e2.0".into()),
                // instructions for e2-expression
                // v2 = <result of e2>
                Instruction::Copy {
                    src: Val::Constant(3),
                    dst: Val::Var("tmp.4".into()), // v2
                },
                // result = v2
                Instruction::Copy {
                    src: Val::Var("tmp.4".into()),
                    dst: Val::Var("tmp.5".into()), // result
                },
                // Label(end)
                Instruction::Label("cond_end.1".into()),
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
            function_declarations: vec![ast_c::FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        ast_c::BlockItem::S(ast_c::Statement::Goto("label1".into())),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label0".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::Constant(0),
                            )),
                        }),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::Constant(1),
                            )),
                        }),
                        ast_c::BlockItem::S(ast_c::Statement::Labeled {
                            label: "label2".into(),
                            statement: Box::new(ast_c::Statement::Return(
                                ast_c::Expression::Constant(2),
                            )),
                        }),
                    ],
                }),
            }],
        };

        // Listing 5-14: Expected TACKY:
        //   Jump(label1)
        //   .label0
        //     Return(0)
        //   .label1
        //     Return(1)
        //   .label2
        //     Return(2)
        assert_eq!(
            emit_program(&program).unwrap(),
            Program {
                function_definitions: vec![FunctionDefinition {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        Instruction::Jump {
                            target: "label1".into(),
                        },
                        Instruction::Label("label0".into()),
                        Instruction::Return(Val::Constant(0)),
                        Instruction::Label("label1".into()),
                        Instruction::Return(Val::Constant(1)),
                        Instruction::Label("label2".into()),
                        Instruction::Return(Val::Constant(2)),
                        // Default Return(0)
                        Instruction::Return(Val::Constant(0)),
                    ],
                }]
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
            function_declarations: vec![
                ast_c::FunDecl {
                    name: "foo".into(),
                    params: vec!["a".into(), "b".into()],
                    body: Some(Block {
                        items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                            ast_c::Expression::Binary(
                                ast_c::BinaryOperator::Add,
                                Box::new(ast_c::Expression::Var("a".into())),
                                Box::new(ast_c::Expression::Var("b".into())),
                            ),
                        ))],
                    }),
                },
                ast_c::FunDecl {
                    name: "bar".into(),
                    params: vec!["a".into()],
                    body: None,
                },
                ast_c::FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![ast_c::BlockItem::S(ast_c::Statement::Return(
                            ast_c::Expression::FunctionCall(
                                "foo".into(),
                                vec![
                                    ast_c::Expression::Constant(42),
                                    ast_c::Expression::Constant(77),
                                ],
                            ),
                        ))],
                    }),
                },
            ],
        };

        assert_eq!(
            emit_program(&program).unwrap(),
            Program {
                function_definitions: vec![
                    FunctionDefinition {
                        name: "foo".into(),
                        params: vec!["a".into(), "b".into()],
                        body: vec![
                            Instruction::Binary {
                                op: BinaryOperator::Add,
                                src1: Val::Var("a".into()),
                                src2: Val::Var("b".into()),
                                dst: Val::Var("tmp.0".into()),
                            },
                            Instruction::Return(Val::Var("tmp.0".into())),
                            // Default Return(0)
                            Instruction::Return(Val::Constant(0)),
                        ],
                    },
                    FunctionDefinition {
                        name: "main".into(),
                        params: vec![],
                        body: vec![
                            Instruction::FunCall {
                                name: "foo".into(),
                                args: vec![Val::Constant(42), Val::Constant(77),],
                                dst: Val::Var("tmp.0".into()),
                            },
                            Instruction::Return(Val::Var("tmp.0".into())),
                            // Default Return(0)
                            Instruction::Return(Val::Constant(0)),
                        ],
                    }
                ]
            }
        );
    }
}
