use crate::ast_c::{Block, BlockItem, Declaration, Expression, Function, Program, Statement};
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

    #[error("{0}")]
    Message(String),
}

pub(crate) fn analyse(program: &mut Program) -> Result<(), Error> {
    variable_resolution(program)?;

    // TODO: for each function...
    label_resolution(&program.function)?;

    Ok(())
}

fn variable_resolution(program: &mut Program) -> Result<(), Error> {
    let mut variable_map = HashMap::<String, MapEntry>::new();
    let mut id_gen = IdGenerator::new();

    // TODO: for globals, and for each function:
    program.function.body = resolve_block(&program.function.body, &mut variable_map, &mut id_gen)?;

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
    Declaration { name, init }: &Declaration,
    variable_map: &mut HashMap<String, MapEntry>,
    id_gen: &mut IdGenerator,
) -> Result<Declaration, Error> {
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

    Ok(Declaration {
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
    }
}

fn label_resolution(function: &Function) -> Result<(), Error> {
    let mut labels = HashMap::<String, usize>::new();

    // AST is a tree, so we need to traverse it recursively,
    // building up the labels as we go, then check for duplicates afterwards.
    for (label_count, item) in function.body.items.iter().enumerate() {
        if let BlockItem::S(statement) = item {
            let nested_labels = nested_labels(&statement);
            for label in nested_labels {
                if labels.contains_key(&label) {
                    return Err(Error::DuplicateLabel(label.clone()));
                }
                labels.insert(label.clone(), label_count);
            }
        }
    }

    // Check for undeclared labels in Goto statements
    for item in &function.body.items {
        if let BlockItem::S(Statement::Goto(label)) = item {
            if !labels.contains_key(label) {
                return Err(Error::UndeclaredLabel(label.as_str().to_owned()));
            }
        }
    }

    Ok(())
}

fn nested_labels(statement: &Statement) -> Vec<String> {
    match statement {
        Statement::Labeled { label, statement } => {
            let mut labels = vec![label.clone()];
            labels.extend(nested_labels(statement));
            labels
        }
        Statement::If { then, else_, .. } => {
            let mut labels = nested_labels(then);
            if let Some(else_stmt) = else_ {
                labels.extend(nested_labels(else_stmt));
            }
            labels
        }
        Statement::Compound(_) => {
            // must be unique within a function, not a scope
            vec![]
        }
        Statement::Return(_) // explicit listing of all variants
        | Statement::Expression(_)
        | Statement::Goto(_)
        | Statement::Null => {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::{Block, Function};

    #[test]
    fn test_analyse() {
        let mut program = Program {
            function: Function {
                name: "main".into(),
                body: Block {
                    items: vec![
                        BlockItem::D(Declaration {
                            name: "x".into(),
                            init: Some(Expression::Constant(42)),
                        }),
                        BlockItem::S(Statement::Return(Expression::Var("x".into()))),
                    ],
                },
            },
        };

        assert!(variable_resolution(&mut program).is_ok());

        assert_eq!(
            program.function.body,
            Block {
                items: vec![
                    BlockItem::D(Declaration {
                        name: "x.0".into(),
                        init: Some(Expression::Constant(42)),
                    }),
                    BlockItem::S(Statement::Return(Expression::Var("x.0".into()))),
                ]
            }
        );
    }

    #[test]
    fn test_goto_undeclared_label() {
        // check for undeclared label

        let mut program = Program {
            function: Function {
                name: "main".into(),
                body: Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("label1".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "label0".into(),
                            statement: Box::new(Statement::Null),
                        }),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ],
                },
            },
        };

        assert!(analyse(&mut program).is_err());
    }

    #[test]
    fn test_goto_duplicate_label() {
        // check for duplicate label
        let mut program = Program {
            function: Function {
                name: "main".into(),
                body: Block {
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
                },
            },
        };

        assert!(analyse(&mut program).is_err());
    }

    #[test]
    fn test_goto_duplicate_nested_label() {
        // check for duplicate label
        let mut program = Program {
            function: Function {
                name: "main".into(),
                body: Block {
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
                },
            },
        };

        assert!(analyse(&mut program).is_err());
    }
}
