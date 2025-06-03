use crate::ast_c::{BlockItem, Declaration, Expression, Program, Statement};
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

    #[error("{0}")]
    Message(String),
}

pub(crate) fn analyse(program: &mut Program) -> Result<(), Error> {
    variable_resolution(program)?;

    Ok(())
}

fn variable_resolution(program: &mut Program) -> Result<(), Error> {
    let mut variable_map = HashMap::<String, String>::new();
    let mut id_gen = IdGenerator::new();

    program
        .function
        .body
        .iter_mut()
        .try_for_each(|item| match item {
            BlockItem::S(statement) => {
                *statement = resolve_statement(statement, &mut variable_map)?;
                Ok(())
            }
            BlockItem::D(declaration) => {
                *declaration = resolve_declaration(declaration, &mut variable_map, &mut id_gen)?;
                Ok(())
            }
        })
}

fn unique_name(source_name: &str, id_gen: &mut IdGenerator) -> String {
    format!("{source_name}.{}", id_gen.next())
}

fn resolve_statement(
    statement: &Statement,
    variable_map: &mut HashMap<String, String>,
) -> Result<Statement, Error> {
    match statement {
        Statement::Return(exp) => Ok(Statement::Return(resolve_exp(exp, variable_map)?)),
        Statement::Expression(exp) => Ok(Statement::Expression(resolve_exp(exp, variable_map)?)),
        Statement::If {
            condition,
            then,
            else_,
        } => todo!(),
        Statement::Null => Ok(Statement::Null),
    }
}

fn resolve_declaration(
    Declaration { name, init }: &Declaration,
    variable_map: &mut HashMap<String, String>,
    id_gen: &mut IdGenerator,
) -> Result<Declaration, Error> {
    if variable_map.contains_key(name) {
        return Err(Error::DuplicateVariableDeclaration(name.clone()));
    }

    let unique_name = unique_name(name, id_gen);
    variable_map.insert(name.clone(), unique_name.clone());

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

fn resolve_exp(
    exp: &Expression,
    variable_map: &mut HashMap<String, String>,
) -> Result<Expression, Error> {
    match exp {
        Expression::Constant(_) => Ok(exp.clone()),
        Expression::Var(v) => {
            if variable_map.contains_key(v) {
                Ok(Expression::Var(variable_map.get(v).unwrap().clone()))
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
        Expression::Conditional(cond, then, else_) =>
        //Ok(Expression::Conditional(
        // resolve_exp(cond, variable_map)?.into(),
        // resolve_exp(then, variable_map)?.into(),
        // resolve_exp(else_, variable_map)?.into(),
        //))
        {
            todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::Function;

    #[test]
    fn test_analyse() {
        let mut program = Program {
            function: Function {
                name: "main".into(),
                body: vec![
                    BlockItem::D(Declaration {
                        name: "x".into(),
                        init: Some(Expression::Constant(42)),
                    }),
                    BlockItem::S(Statement::Return(Expression::Var("x".into()))),
                ],
            },
        };

        assert!(variable_resolution(&mut program).is_ok());

        assert_eq!(
            program.function.body,
            vec![
                BlockItem::D(Declaration {
                    name: "x.0".into(),
                    init: Some(Expression::Constant(42)),
                }),
                BlockItem::S(Statement::Return(Expression::Var("x.0".into()))),
            ]
        );
    }
}
