use crate::ast_c::{
    Block, BlockItem, Declaration, Expression, ForInit, FunDecl, Program, Statement, VarDecl,
};
use crate::id_gen::IdGenerator;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum Error {
    #[error("Undeclared variable: {0}")]
    UndeclaredVariable(String),

    #[error("Duplicate variable declaration: {0}")]
    DuplicateVariableDeclaration(String),

    #[error("Invalid lvalue")]
    InvalidLValue,

    #[error("Undeclared label: {0}")]
    UndeclaredLabel(String),

    #[error("Duplicate label: {0}")]
    DuplicateLabel(String),

    #[error("Break statement outside of loop")]
    BreakOutsideLoop,

    #[error("Continue statement outside of loop")]
    ContinueOutsideLoop,

    #[error("{0}")]
    Message(String),
}

pub(crate) fn analyse(program: &mut Program) -> Result<(), Error> {
    variable_resolution(program)?;

    loop_labeling(program)?;

    for function in &program.function_declarations {
        goto_label_resolution(function)?;
    }

    Ok(())
}

fn variable_resolution(program: &mut Program) -> Result<(), Error> {
    let mut variable_map = HashMap::<String, MapEntry>::new();
    let mut id_gen = IdGenerator::new();

    // TODO: for globals

    for function in &mut program.function_declarations {
        if let Some(body) = &mut function.body {
            *body = resolve_block(body, &mut variable_map, &mut id_gen)?;
        }
    }

    Ok(())
}

fn unique_name(source_name: &str, id_gen: &mut IdGenerator) -> String {
    format!("{source_name}.{}", id_gen.next())
}

#[derive(Debug, Clone)]
struct MapEntry {
    unique_name: String,
    from_current_block: bool,
}

fn resolve_block(
    block: &Block,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Block, Error> {
    let mut items = Vec::new();
    for item in &block.items {
        match item {
            BlockItem::S(statement) => {
                items.push(BlockItem::S(resolve_statement(
                    statement,
                    variable_map,
                    id_gen,
                )?));
            }
            BlockItem::D(declaration) => {
                items.push(BlockItem::D(resolve_declaration(
                    declaration,
                    variable_map,
                    id_gen,
                )?));
            }
        }
    }
    Ok(Block { items })
}

fn resolve_declaration(
    declaration: &Declaration,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Declaration, Error> {
    match declaration {
        Declaration::FunDecl(_) => {
            todo!()
        }
        Declaration::VarDecl(var_decl) => Ok(Declaration::VarDecl(resolve_variable_declaration(
            var_decl,
            variable_map,
            id_gen,
        )?)),
    }
}

fn resolve_variable_declaration(
    VarDecl { name, init }: &VarDecl,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<VarDecl, Error> {
    if variable_map.contains_key(name) && variable_map[name].from_current_block {
        return Err(Error::DuplicateVariableDeclaration(name.clone()));
    }

    let unique_name = unique_name(name, id_gen);
    variable_map.insert(
        name.clone(),
        MapEntry {
            unique_name: unique_name.clone(),
            from_current_block: true,
        },
    );

    let init = if let Some(init) = init {
        Some(resolve_exp(init, variable_map)?)
    } else {
        None
    };

    Ok(VarDecl {
        name: unique_name,
        init,
    })
}

fn resolve_statement(
    statement: &Statement,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Statement, Error> {
    match statement {
        Statement::Return(exp) => Ok(Statement::Return(resolve_exp(exp, variable_map)?)),
        Statement::Expression(exp) => Ok(Statement::Expression(resolve_exp(exp, variable_map)?)),
        Statement::If {
            condition,
            then,
            else_,
        } => {
            let else_ = if let Some(else_stmt) = else_ {
                Some(Box::new(resolve_statement(
                    else_stmt,
                    variable_map,
                    id_gen,
                )?))
            } else {
                None
            };
            Ok(Statement::If {
                condition: resolve_exp(condition, variable_map)?,
                then: Box::new(resolve_statement(then, variable_map, id_gen)?),
                else_,
            })
        }
        Statement::Labeled { label, statement } => {
            resolve_statement(statement, variable_map, id_gen).map(|stmt| Statement::Labeled {
                label: label.clone(),
                statement: Box::new(stmt),
            })
        }
        Statement::Goto(identifier) => Ok(Statement::Goto(identifier.clone())),
        Statement::Compound(block) => {
            let mut new_variable_map = copy_variable_map(variable_map);
            Ok(Statement::Compound(resolve_block(
                block,
                &mut new_variable_map,
                id_gen,
            )?))
        }
        Statement::Break(_) => Ok(statement.clone()),
        Statement::Continue(_) => Ok(statement.clone()),
        Statement::While {
            condition,
            body,
            loop_label,
        } => Ok(Statement::While {
            condition: resolve_exp(condition, variable_map)?,
            body: Box::new(resolve_statement(body, variable_map, id_gen)?),
            loop_label: loop_label.clone(),
        }),
        Statement::DoWhile {
            body,
            condition,
            loop_label,
        } => Ok(Statement::DoWhile {
            body: Box::new(resolve_statement(body, variable_map, id_gen)?),
            condition: resolve_exp(condition, variable_map)?,
            loop_label: loop_label.clone(),
        }),
        Statement::For {
            init,
            condition,
            post,
            body,
            loop_label,
        } => {
            let mut new_variable_map = copy_variable_map(variable_map);
            let init = resolve_for_init(init, &mut new_variable_map, id_gen)?;
            let condition = resolve_optional_exp(condition, &mut new_variable_map)?;
            let post = resolve_optional_exp(post, &mut new_variable_map)?;
            let body = resolve_statement(body, &mut new_variable_map, id_gen)?;
            Ok(Statement::For {
                init,
                condition,
                post,
                body: Box::new(body),
                loop_label: loop_label.clone(),
            })
        }
        Statement::Null => Ok(Statement::Null),
    }
}

fn copy_variable_map(variable_map: &HashMap<String, MapEntry>) -> HashMap<String, MapEntry> {
    // clone the hashmap but reset the `from_current_block` flag
    variable_map
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                MapEntry {
                    unique_name: v.unique_name.clone(),
                    from_current_block: false,
                },
            )
        })
        .collect()
}

fn resolve_exp(
    exp: &Expression,
    variable_map: &mut HashMap<String, MapEntry>,
) -> Result<Expression, Error> {
    match exp {
        Expression::Constant(_) => Ok(exp.clone()),
        Expression::Var(v) => {
            if variable_map.contains_key(v) {
                Ok(Expression::Var(
                    variable_map.get(v).unwrap().unique_name.clone(),
                ))
            } else {
                Err(Error::UndeclaredVariable(v.clone()))
            }
        }
        Expression::Unary(op, exp) => Ok(Expression::Unary(
            op.clone(),
            resolve_exp(exp, variable_map)?.into(),
        )),
        Expression::Binary(op, left, right) => Ok(Expression::Binary(
            op.clone(),
            resolve_exp(left, variable_map)?.into(),
            resolve_exp(right, variable_map)?.into(),
        )),
        Expression::Assignment(left, right) => {
            if !matches!(**left, Expression::Var(_)) {
                return Err(Error::InvalidLValue);
            }
            Ok(Expression::Assignment(
                resolve_exp(left, variable_map)?.into(),
                resolve_exp(right, variable_map)?.into(),
            ))
        }
        Expression::Conditional(cond, then, else_) => Ok(Expression::Conditional(
            resolve_exp(cond, variable_map)?.into(),
            resolve_exp(then, variable_map)?.into(),
            resolve_exp(else_, variable_map)?.into(),
        )),
        Expression::FunctionCall(..) => {
            todo!()
        }
    }
}

fn resolve_for_init(
    init: &ForInit,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<ForInit, Error> {
    match init {
        ForInit::InitDecl(decl) => Ok(ForInit::InitDecl(resolve_variable_declaration(
            decl,
            variable_map,
            id_gen,
        )?)),
        ForInit::InitExp(exp) => Ok(ForInit::InitExp(resolve_optional_exp(exp, variable_map)?)),
    }
}

fn resolve_optional_exp(
    exp: &Option<Expression>,
    variable_map: &mut HashMap<String, MapEntry>,
) -> Result<Option<Expression>, Error> {
    if let Some(exp) = exp {
        Ok(Some(resolve_exp(exp, variable_map)?))
    } else {
        Ok(None)
    }
}

fn loop_labeling(program: &mut Program) -> Result<(), Error> {
    let mut variable_map = HashMap::<String, MapEntry>::new();
    let mut id_gen = IdGenerator::new();

    // TODO: for globals

    for function in &mut program.function_declarations {
        if let Some(body) = &mut function.body {
            *body = loop_label_block(body, &mut variable_map, &mut id_gen)?;
        }
    }

    Ok(())
}

fn loop_label_block(
    block: &Block,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Block, Error> {
    let mut items = Vec::new();
    for item in &block.items {
        match item {
            BlockItem::S(statement) => {
                items.push(BlockItem::S(loop_label_statement(
                    statement,
                    None,
                    variable_map,
                    id_gen,
                )?));
            }
            BlockItem::D(_) => {
                items.push(item.clone()); // declarations do not have loops
            }
        }
    }
    Ok(Block { items })
}

fn loop_label_statement(
    statement: &Statement,
    current_label: Option<String>,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Statement, Error> {
    match statement {
        Statement::If {
            condition,
            then,
            else_,
        } => Ok(Statement::If {
            condition: condition.clone(),
            then: Box::new(loop_label_statement(
                then,
                current_label.clone(),
                variable_map,
                id_gen,
            )?),
            else_: if let Some(else_stmt) = else_ {
                Some(Box::new(loop_label_statement(
                    else_stmt,
                    current_label.clone(),
                    variable_map,
                    id_gen,
                )?))
            } else {
                None
            },
        }),
        Statement::Labeled { label, statement } => Ok(Statement::Labeled {
            label: label.clone(),
            statement: Box::new(loop_label_statement(
                statement,
                current_label.clone(),
                variable_map,
                id_gen,
            )?),
        }),
        Statement::Compound(block) => {
            let mut items = vec![];
            for item in &block.items {
                match item {
                    BlockItem::S(statement) => {
                        let labeled_statement = loop_label_statement(
                            statement,
                            current_label.clone(),
                            variable_map,
                            id_gen,
                        )?;
                        items.push(BlockItem::S(labeled_statement));
                    }
                    BlockItem::D(_) => {
                        items.push(item.clone()); // declarations do not have loops
                    }
                }
            }
            Ok(Statement::Compound(Block { items }))
        }
        Statement::Break(_) => {
            if let Some(label) = &current_label {
                Ok(Statement::Break(Some(label.clone())))
            } else {
                Err(Error::BreakOutsideLoop)
            }
        }
        Statement::Continue(_) => {
            if let Some(label) = &current_label {
                Ok(Statement::Continue(Some(label.clone())))
            } else {
                Err(Error::ContinueOutsideLoop)
            }
        }
        Statement::While {
            condition,
            body,
            loop_label: _,
        } => {
            let loop_label = Some(unique_name("while", id_gen));
            let labeled_body =
                loop_label_statement(body, loop_label.clone(), variable_map, id_gen)?;
            Ok(Statement::While {
                condition: condition.clone(),
                body: Box::new(labeled_body),
                loop_label,
            })
        }
        Statement::DoWhile {
            body,
            condition,
            loop_label: _,
        } => {
            let loop_label = Some(unique_name("do_while", id_gen));
            let labeled_body =
                loop_label_statement(body, loop_label.clone(), variable_map, id_gen)?;
            Ok(Statement::DoWhile {
                body: Box::new(labeled_body),
                condition: condition.clone(),
                loop_label,
            })
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            loop_label: _,
        } => {
            let loop_label = Some(unique_name("for", id_gen));
            let labeled_body =
                loop_label_statement(body, loop_label.clone(), variable_map, id_gen)?;
            Ok(Statement::For {
                init: init.clone(),
                condition: condition.clone(),
                post: post.clone(),
                body: labeled_body.into(),
                loop_label,
            })
        }
        Statement::Return(_) | Statement::Expression(_) | Statement::Goto(_) | Statement::Null => {
            Ok(statement.clone())
        }
    }
}

// GOTO labels:

fn goto_label_resolution(function_decl: &FunDecl) -> Result<(), Error> {
    let mut labels = HashMap::<String, usize>::new();

    if let Some(body) = &function_decl.body {
        // AST is a tree, so we need to traverse it recursively,
        // building up the labels as we go, then check for duplicates afterwards.
        for item in body.items.iter() {
            if let BlockItem::S(statement) = item {
                let nested_labels = nested_goto_labels(statement)?;
                extend_goto_labels(&mut labels, &nested_labels)?;
            }
        }

        // Check for undeclared labels in Goto statements
        for item in &body.items {
            if let BlockItem::S(Statement::Goto(label)) = item {
                if !labels.contains_key(label) {
                    return Err(Error::UndeclaredLabel(label.as_str().to_owned()));
                }
            }
        }
    }

    Ok(())
}

fn extend_goto_labels(
    labels: &mut HashMap<String, usize>,
    new_labels: &HashMap<String, usize>,
) -> Result<(), Error> {
    for label in new_labels.keys() {
        if labels.contains_key(label) {
            return Err(Error::DuplicateLabel(label.clone()));
        }
        labels.insert(label.clone(), 0); // value is not used, just to ensure uniqueness
    }
    Ok(())
}

fn nested_goto_labels(statement: &Statement) -> Result<HashMap<String, usize>, Error> {
    let mut labels = HashMap::new();
    match statement {
        Statement::Labeled { label, statement } => {
            labels.insert(label.clone(), 0);
            let nested_labels = nested_goto_labels(statement)?;
            extend_goto_labels(&mut labels, &nested_labels)?;
        }
        Statement::If { then, else_, .. } => {
            let nested_labels_ = nested_goto_labels(then)?;
            extend_goto_labels(&mut labels, &nested_labels_)?;
            if let Some(else_stmt) = else_ {
                let nested_labels_ = nested_goto_labels(else_stmt)?;
                extend_goto_labels(&mut labels, &nested_labels_)?;
            }
        }
        Statement::Compound(block) => {
            // must be unique within a function, not a scope
            for item in &block.items {
                match item {
                    BlockItem::D(_) => continue, // declarations do not have labels
                    BlockItem::S(statement) => {
                        let nested = nested_goto_labels(statement)?;
                        extend_goto_labels(&mut labels, &nested)?;
                    }
                }
            }
        }
        Statement::While { body, .. } => {
            let nested_labels = nested_goto_labels(body)?;
            extend_goto_labels(&mut labels, &nested_labels)?;
        },
        Statement::DoWhile { body, .. } => {
            let nested_labels = nested_goto_labels(body)?;
            extend_goto_labels(&mut labels, &nested_labels)?;
        },
        Statement::For { body, .. } => {
            let nested_labels = nested_goto_labels(body)?;
            extend_goto_labels(&mut labels, &nested_labels)?;
        },
        Statement::Return(_) // explicit listing of all variants
        | Statement::Break(_)
        | Statement::Continue(_)
        | Statement::Expression(_)
        | Statement::Goto(_)
        | Statement::Null => {
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::{BinaryOperator, Block, FunDecl, VarDecl};
    use assert_matches::assert_matches;

    #[test]
    fn test_analyse() {
        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec!["void".into()],
                body: Some(Block {
                    items: vec![
                        BlockItem::D(Declaration::VarDecl(VarDecl {
                            name: "x".into(),
                            init: Some(Expression::Constant(42)),
                        })),
                        BlockItem::S(Statement::Return(Expression::Var("x".into()))),
                    ],
                }),
            }],
        };

        assert!(variable_resolution(&mut program).is_ok());

        assert_eq!(
            program.function_declarations[0].body,
            Some(Block {
                items: vec![
                    BlockItem::D(Declaration::VarDecl(VarDecl {
                        name: "x.0".into(),
                        init: Some(Expression::Constant(42)),
                    })),
                    BlockItem::S(Statement::Return(Expression::Var("x.0".into()))),
                ]
            })
        );
    }

    #[test]
    fn test_goto_undeclared_label() {
        // check for undeclared label

        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec!["void".into()],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("label1".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "label0".into(),
                            statement: Box::new(Statement::Null),
                        }),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ],
                }),
            }],
        };

        assert!(analyse(&mut program).is_err());
    }

    #[test]
    fn test_goto_duplicate_label() {
        // check for duplicate label
        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec!["void".into()],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("label1".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(Statement::Null),
                        }),
                        BlockItem::S(Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(Statement::Null),
                        }),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ],
                }),
            }],
        };

        assert!(analyse(&mut program).is_err());
    }

    #[test]
    fn test_goto_duplicate_nested_label() {
        // check for duplicate label
        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec!["void".into()],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("label1".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(Statement::Labeled {
                                label: "label1".into(),
                                statement: Box::new(Statement::Null),
                            }),
                        }),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ],
                }),
            }],
        };

        assert!(analyse(&mut program).is_err());
    }

    #[test]
    fn test_loop_labeling() {
        // Listing 8-22:
        //
        // while (a > 0) {                            // loop 1
        //     for (int i = 0; i < 10; i = i + 1) {   // loop 2
        //         if (i % 2 == 0)
        //             continue;                      // -> loop 2
        //         a = a / 2;
        //     }
        //     if (a == b)
        //         break;                             // -> loop 1
        // }
        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec!["void".into()],
                body: Some(Block {
                    items: vec![BlockItem::S(Statement::While {
                        condition: Expression::Binary(
                            BinaryOperator::GreaterThan,
                            Box::new(Expression::Var("a".into())),
                            Box::new(Expression::Constant(0)),
                        ),
                        body: Box::new(Statement::Compound(Block {
                            items: vec![
                                BlockItem::S(Statement::For {
                                    init: ForInit::InitDecl(VarDecl {
                                        name: "i".into(),
                                        init: Some(Expression::Constant(0)),
                                    }),
                                    condition: Some(Expression::Binary(
                                        BinaryOperator::LessThan,
                                        Box::new(Expression::Var("i".into())),
                                        Box::new(Expression::Constant(10)),
                                    )),
                                    post: Some(Expression::Assignment(
                                        Box::new(Expression::Var("i".into())),
                                        Box::new(Expression::Binary(
                                            BinaryOperator::Add,
                                            Box::new(Expression::Var("i".into())),
                                            Box::new(Expression::Constant(1)),
                                        )),
                                    )),
                                    body: Box::new(Statement::Compound(Block {
                                        items: vec![
                                            BlockItem::S(Statement::If {
                                                condition: Expression::Binary(
                                                    BinaryOperator::Equal,
                                                    Box::new(Expression::Binary(
                                                        BinaryOperator::Remainder,
                                                        Box::new(Expression::Var("i".into())),
                                                        Box::new(Expression::Constant(2)),
                                                    )),
                                                    Box::new(Expression::Constant(0)),
                                                ),
                                                then: Box::new(Statement::Continue(None)),
                                                else_: None,
                                            }),
                                            BlockItem::S(Statement::Expression(
                                                Expression::Assignment(
                                                    Box::new(Expression::Var("a".into())),
                                                    Box::new(Expression::Assignment(
                                                        Box::new(Expression::Var("a".into())),
                                                        Box::new(Expression::Binary(
                                                            BinaryOperator::Divide,
                                                            Box::new(Expression::Var("a".into())),
                                                            Box::new(Expression::Constant(2)),
                                                        )),
                                                    )),
                                                ),
                                            )),
                                        ],
                                    })),
                                    loop_label: None,
                                }),
                                BlockItem::S(Statement::If {
                                    condition: Expression::Binary(
                                        BinaryOperator::Equal,
                                        Box::new(Expression::Var("a".into())),
                                        Box::new(Expression::Var("b".into())),
                                    ),
                                    then: Box::new(Statement::Break(None)),
                                    else_: None,
                                }),
                            ],
                        })),
                        loop_label: None,
                    })],
                }),
            }],
        };

        let result = loop_labeling(&mut program);
        assert!(result.is_ok());

        // Perhaps these complex matches are better than just checking the full AST?

        // Outer loop should be labeled "while.0"
        assert_matches!(
                    &program.function_declarations[0].body.as_ref().unwrap().items[0],
                        BlockItem::S(stmt) if matches!(
                            &stmt, Statement::While { loop_label, .. } if *loop_label == Some("while.0".to_string())
            )
        );
        assert_matches!(
            &program.function_declarations[0].body.as_ref().unwrap().items[0],
            BlockItem::S(stmt) if matches!(
                &stmt, Statement::While { body, .. } if matches!(
                    &**body, Statement::Compound(block) if matches!(
                        &block.items[1], BlockItem::S(Statement::If { then, .. }) if matches!(
                            &**then, Statement::Break(loop_label) if
                                *loop_label == Some("while.0".to_string())
                            )
                    )
                )
            )
        );

        // Inner loop should be labeled "for.1"
        assert_matches!(
            &program.function_declarations[0].body.as_ref().unwrap().items[0],
            BlockItem::S(stmt) if matches!(
                &stmt, Statement::While { body, .. } if matches!(
                    &**body, Statement::Compound(block) if matches!(
                        &block.items[0], BlockItem::S(Statement::For { loop_label, .. }) if *loop_label == Some("for.1".to_string())
                    )
                )
            )
        );
        assert_matches!(
            &program.function_declarations[0].body.as_ref().unwrap().items[0],
            BlockItem::S(stmt) if matches!(
                &stmt, Statement::While { body, .. } if matches!(
                    &**body, Statement::Compound(block) if matches!(
                        &block.items[0], BlockItem::S(Statement::For { body, .. }) if matches!(
                            &**body, Statement::Compound(block) if matches!(
                                &block.items[0], BlockItem::S(Statement::If { then, .. } ) if matches!(
                                    &**then, Statement::Continue(loop_label) if
                                *loop_label == Some("for.1".to_string())
                            ))
                        )
                    )
                )
            )
        );
    }
}
