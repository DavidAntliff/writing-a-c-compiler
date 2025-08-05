use crate::ast_c::{
    BinaryOperator, Block, BlockItem, Const, Declaration, Expression, ForInit, FunDecl,
    InnerTypedExpression, Program, Statement, StorageClass, Type, TypedBlock, TypedBlockItem,
    TypedDeclaration, TypedExpression, TypedForInit, TypedFunDecl, TypedProgram, TypedStatement,
    TypedVarDecl, UnaryOperator, VarDecl,
};
use crate::lexer::Identifier;
use crate::semantics::{Error, IdentifierAttrs, InitialValue, StaticInit, SymbolTable};

pub(crate) fn type_checking(program: &Program) -> Result<SymbolTable, Error> {
    let mut symbol_table = SymbolTable::new();

    let mut typed_program = TypedProgram {
        declarations: vec![],
    };

    for declaration in &program.declarations {
        typed_program.declarations.push(match declaration {
            Declaration::FunDecl(fun_decl) => TypedDeclaration::FunDecl(
                typecheck_function_declaration(fun_decl, &mut symbol_table)?,
            ),
            Declaration::VarDecl(var_decl) => TypedDeclaration::VarDecl(
                typecheck_file_scope_variable_declaration(var_decl, &mut symbol_table)?,
            ),
        })
    }

    //dbg!(&typed_program);

    Ok(symbol_table)
}

fn typecheck_file_scope_variable_declaration(
    decl: &VarDecl,
    symbol_table: &mut SymbolTable,
) -> Result<TypedVarDecl, Error> {
    // TODO disallow void variables
    // TODO disallow initialisers for incomplete types?

    // Determine the variable's initial value, and make sure it is converted to the correct
    // representation for the variable's type.
    let mut initial_value = match &decl.init {
        Some(Expression::Constant(Const::ConstInt(i))) => match decl.var_type {
            Type::Int => InitialValue::Initial(StaticInit::IntInit(*i)),
            Type::Long => InitialValue::Initial(StaticInit::LongInit(*i as i64)),
            _ => {
                return Err(Error::InvalidTypeForVariableInitialiser(
                    decl.name.clone(),
                    decl.var_type.clone(),
                ));
            }
        },
        Some(Expression::Constant(Const::ConstLong(i))) => match decl.var_type {
            Type::Int => InitialValue::Initial(StaticInit::IntInit(*i as i32)),
            Type::Long => InitialValue::Initial(StaticInit::LongInit(*i)),
            _ => {
                return Err(Error::InvalidTypeForVariableInitialiser(
                    decl.name.clone(),
                    decl.var_type.clone(),
                ));
            }
        },
        None => {
            if decl.storage_class == Some(StorageClass::Extern) {
                InitialValue::NoInitialiser
            } else {
                InitialValue::Tentative
            }
        }
        _ => {
            return Err(Error::NonConstantInitialiser(decl.name.clone()));
        }
    };

    // Determine if tentatively globally visible, or static
    let mut global = decl.storage_class != Some(StorageClass::Static);

    // Consider prior declarations, disagreements result in errors
    if let Some(old_decl) = symbol_table.get(&decl.name) {
        if old_decl.type_ != decl.var_type {
            return Err(Error::VariableRedeclaredWithDifferentType(
                decl.name.clone(),
            ));
        }

        match &old_decl.attrs {
            IdentifierAttrs::Static {
                init,
                global: _global,
            } => {
                // Extern solidifies any tentative declaration
                if decl.storage_class == Some(StorageClass::Extern) {
                    global = *_global;
                } else if *_global != global {
                    return Err(Error::ConflictingVariableLinkage(decl.name.clone()));
                }

                // Account for explicit initialisers
                if matches!(init, InitialValue::Initial(_)) {
                    if matches!(initial_value, InitialValue::Initial(_)) {
                        return Err(Error::ConflictingFileScopeVariableDefinitions(
                            decl.name.clone(),
                        ));
                    } else {
                        initial_value = init.clone();
                    }
                } else if !matches!(initial_value, InitialValue::Initial(_))
                    && matches!(init, InitialValue::Tentative)
                {
                    initial_value = InitialValue::Tentative;
                }
            }
            _ => panic!("File-scope variable declaration should have static storage duration"),
        }
    }

    // Add to symbol table
    let attrs = IdentifierAttrs::Static {
        init: initial_value,
        global,
    };
    symbol_table.add(decl.name.clone(), decl.var_type.clone(), attrs);

    // We don't need the initialiser any more, so we can drop it
    // let typed_init = if let Some(exp) = &decl.init {
    //     Some(typecheck_exp(exp, symbol_table)?)
    // } else {
    //     None
    // };
    let typed_init = None;

    Ok(TypedVarDecl {
        name: decl.name.clone(),
        init: typed_init,
        var_type: decl.var_type.clone(),
        storage_class: decl.storage_class.clone(),
    })
}

fn typecheck_local_variable_declaration(
    decl: &VarDecl,
    symbol_table: &mut SymbolTable,
) -> Result<TypedVarDecl, Error> {
    // TODO disallow void variables

    Ok(match decl.storage_class {
        Some(StorageClass::Extern) => {
            // Extern variables cannot have initialisers
            if decl.init.is_some() {
                return Err(Error::InitialiserOnLocalExternVariableDeclaration(
                    decl.name.clone(),
                ));
            }
            if let Some(old_decl) = symbol_table.get(&decl.name) {
                if old_decl.type_ != decl.var_type {
                    return Err(Error::VariableRedeclaredWithDifferentType(
                        decl.name.clone(),
                    ));
                }
            } else {
                symbol_table.add(
                    decl.name.clone(),
                    decl.var_type.clone(),
                    IdentifierAttrs::Static {
                        init: InitialValue::NoInitialiser,
                        global: true,
                    },
                );
            }
            TypedVarDecl {
                name: decl.name.clone(),
                init: None,
                var_type: decl.var_type.clone(),
                storage_class: decl.storage_class.clone(),
            }
        }
        Some(StorageClass::Static) => {
            // Static variables have no linkage, and must have an initialiser otherwise zero.
            let initial_value = match &decl.init {
                Some(Expression::Constant(Const::ConstInt(i))) => {
                    InitialValue::Initial(StaticInit::IntInit(*i))
                }
                Some(Expression::Constant(Const::ConstLong(i))) => {
                    InitialValue::Initial(StaticInit::LongInit(*i))
                }
                None => InitialValue::Initial(StaticInit::IntInit(0)),
                _ => {
                    return Err(Error::NonConstantLocalStaticInitialiser(decl.name.clone()));
                }
            };
            symbol_table.add(
                decl.name.clone(),
                Type::Int,
                IdentifierAttrs::Static {
                    init: initial_value,
                    global: false,
                },
            );
            TypedVarDecl {
                name: decl.name.clone(),
                init: None,
                var_type: decl.var_type.clone(),
                storage_class: decl.storage_class.clone(),
            }
        }
        None => {
            // Automatic variable
            symbol_table.add(decl.name.clone(), Type::Int, IdentifierAttrs::Local);
            let init = if let Some(init) = &decl.init {
                Some(typecheck_exp(init, symbol_table)?)
            } else {
                None
            };
            TypedVarDecl {
                name: decl.name.clone(),
                init,
                var_type: decl.var_type.clone(),
                storage_class: decl.storage_class.clone(),
            }
        }
    })
}

fn typecheck_function_declaration(
    decl: &FunDecl,
    symbol_table: &mut SymbolTable,
) -> Result<TypedFunDecl, Error> {
    let fun_type = match &decl.fun_type {
        t @ Type::Function { .. } => t.clone(),
        _ => panic!("Function declaration should have function type"),
    };
    let has_body = decl.body.is_some();
    let mut already_defined = false;

    // Linkage is internal if static, or tentatively external
    let mut global = decl.storage_class != Some(StorageClass::Static);

    if let Some(old_decl) = symbol_table.get(&decl.name) {
        if old_decl.type_ != fun_type {
            return Err(Error::IncompatibleFunctionDeclaration(decl.name.clone()));
        }

        match old_decl.attrs {
            IdentifierAttrs::Fun {
                defined,
                global: global_,
            } => {
                already_defined = defined;
                if already_defined && has_body {
                    return Err(Error::MultipleDefinitions(decl.name.clone()));
                }

                if global_ && decl.storage_class == Some(StorageClass::Static) {
                    return Err(Error::StaticFunctionDeclarationFollowsNonStatic(
                        decl.name.clone(),
                    ));
                }
                global = global_;
            }
            _ => panic!("Function declaration should have function attributes"),
        }
    }

    let attrs = IdentifierAttrs::Fun {
        defined: already_defined || has_body,
        global,
    };

    let fun_name = decl.name.clone();
    symbol_table.add(fun_name.clone(), fun_type.clone(), attrs);

    let typed_body = if let Some(body) = &decl.body {
        for param in decl.params.iter() {
            symbol_table.add(param.clone(), Type::Int, IdentifierAttrs::Local);
        }
        Some(typecheck_block(body, &fun_name, symbol_table)?)
    } else {
        None
    };

    Ok(TypedFunDecl {
        name: decl.name.clone(),
        params: decl.params.clone(),
        body: typed_body,
        fun_type,
        storage_class: decl.storage_class.clone(),
    })
}

fn typecheck_block(
    block: &Block,
    fun_name: &Identifier,
    symbol_table: &mut SymbolTable,
) -> Result<TypedBlock, Error> {
    let mut typed_block = TypedBlock { items: vec![] };
    for item in &block.items {
        match item {
            BlockItem::S(statement) => {
                typed_block
                    .items
                    .push(TypedBlockItem::S(typecheck_statement(
                        statement,
                        fun_name,
                        symbol_table,
                    )?))
            }
            BlockItem::D(declaration) => {
                typed_block
                    .items
                    .push(TypedBlockItem::D(typecheck_declaration(
                        declaration,
                        symbol_table,
                    )?))
            }
        }
    }
    Ok(typed_block)
}

fn typecheck_statement(
    statement: &Statement,
    fun_name: &Identifier,
    symbol_table: &mut SymbolTable,
) -> Result<TypedStatement, Error> {
    Ok(match statement {
        Statement::Return(exp) => {
            let fun_return_type = symbol_table
                .get(fun_name)
                .ok_or_else(|| Error::UndeclaredFunction(fun_name.clone()))?;
            let typed_exp = typecheck_exp(exp, symbol_table)?;
            let converted_type = convert_to(typed_exp, &fun_return_type.type_);
            TypedStatement::Return(converted_type)
        }
        Statement::Expression(exp) => TypedStatement::Expression(typecheck_exp(exp, symbol_table)?),
        Statement::If {
            condition,
            then_block: then,
            else_block: else_,
        } => {
            let typed_condition = typecheck_exp(condition, symbol_table)?;
            let typed_then = typecheck_statement(then, fun_name, symbol_table)?;
            let typed_else = if let Some(else_stmt) = else_ {
                Some(Box::new(typecheck_statement(
                    else_stmt,
                    fun_name,
                    symbol_table,
                )?))
            } else {
                None
            };
            TypedStatement::If {
                condition: typed_condition,
                then_block: typed_then.into(),
                else_block: typed_else,
            }
        }
        Statement::Labeled { statement, .. } => {
            // Labels are not checked here, they are checked in goto_label_resolution
            typecheck_statement(statement, fun_name, symbol_table)?
        }
        Statement::Goto(statement) => {
            // Goto labels are checked in goto_label_resolution
            TypedStatement::Goto(statement.clone())
        }
        Statement::Compound(block) => {
            TypedStatement::Compound(typecheck_block(block, fun_name, symbol_table)?)
        }
        // 'break' and 'continue' do not have expressions to check
        Statement::Break(maybe_label) => TypedStatement::Break(maybe_label.clone()),
        Statement::Continue(maybe_label) => TypedStatement::Continue(maybe_label.clone()),
        Statement::While {
            condition,
            body,
            loop_label,
        } => {
            let typed_condition = typecheck_exp(condition, symbol_table)?;
            let typed_body = typecheck_statement(body, fun_name, symbol_table)?;
            TypedStatement::While {
                condition: typed_condition,
                body: typed_body.into(),
                loop_label: loop_label.clone(),
            }
        }
        Statement::DoWhile {
            body,
            condition,
            loop_label,
        } => {
            let typed_body = typecheck_statement(body, fun_name, symbol_table)?;
            let typed_condition = typecheck_exp(condition, symbol_table)?;
            TypedStatement::DoWhile {
                body: typed_body.into(),
                condition: typed_condition,
                loop_label: loop_label.clone(),
            }
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            loop_label,
        } => {
            let typed_init = match init {
                ForInit::InitDecl(var_decl) => {
                    // Storage-class specifiers not allowed in for-init-decls
                    if var_decl.storage_class.is_some() {
                        return Err(Error::InvalidStorageClass(
                            var_decl.storage_class.clone().unwrap(),
                        ));
                    }
                    TypedForInit::InitDecl(typecheck_local_variable_declaration(
                        var_decl,
                        symbol_table,
                    )?)
                }
                ForInit::InitExp(Some(exp)) => {
                    TypedForInit::InitExp(Some(typecheck_exp(exp, symbol_table)?))
                }
                ForInit::InitExp(None) => TypedForInit::InitExp(None),
            };

            let typed_condition = if let Some(cond) = condition {
                Some(typecheck_exp(cond, symbol_table)?)
            } else {
                None
            };

            let typed_post = if let Some(post_exp) = post {
                Some(typecheck_exp(post_exp, symbol_table)?)
            } else {
                None
            };

            let typed_body = typecheck_statement(body, fun_name, symbol_table)?;

            TypedStatement::For {
                init: typed_init,
                condition: typed_condition,
                post: typed_post,
                body: typed_body.into(),
                loop_label: loop_label.clone(),
            }
        }
        Statement::Null => TypedStatement::Null,
    })
}

fn typecheck_declaration(
    declaration: &Declaration,
    symbol_table: &mut SymbolTable,
) -> Result<TypedDeclaration, Error> {
    Ok(match declaration {
        Declaration::VarDecl(var_decl) => TypedDeclaration::VarDecl(
            typecheck_local_variable_declaration(var_decl, symbol_table)?,
        ),
        Declaration::FunDecl(fun_decl) => {
            TypedDeclaration::FunDecl(typecheck_function_declaration(fun_decl, symbol_table)?)
        }
    })
}

fn typecheck_exp(
    expression: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    match expression {
        Expression::Constant(c) => typecheck_constant(c),
        Expression::Var(name) => typecheck_variable(name, symbol_table),
        Expression::Cast(t, inner) => typecheck_cast(t, inner, symbol_table),
        Expression::Unary(op, exp) => typecheck_unary(op, exp, symbol_table),
        Expression::Binary(op, left, right) => typecheck_binary(op, left, right, symbol_table),
        Expression::Assignment(left, right) => typecheck_assignment(left, right, symbol_table),
        Expression::Conditional(cond, then_expr, else_expr) => {
            typecheck_conditional(cond, then_expr, else_expr, symbol_table)
        }
        Expression::FunctionCall(fun_name, args) => {
            typecheck_function_call(fun_name, args, symbol_table)
        }
    }
}

fn typecheck_constant(constant: &Const) -> Result<TypedExpression, Error> {
    match constant {
        Const::ConstInt(_) => Ok(TypedExpression(
            Type::Int,
            InnerTypedExpression::Constant(constant.clone()),
        )),
        Const::ConstLong(_) => Ok(TypedExpression(
            Type::Long,
            InnerTypedExpression::Constant(constant.clone()),
        )),
    }
}

fn typecheck_variable(
    name: &Identifier,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let item = symbol_table.get(name);
    match item {
        Some(entry) if matches!(entry.type_, Type::Function { .. }) => {
            Err(Error::FunctionNameUsedAsVariable(name.clone()))
        }
        Some(entry) => Ok(TypedExpression(
            entry.type_.clone(),
            InnerTypedExpression::Var(name.clone()),
        )),
        None => Err(Error::UndeclaredVariable(name.clone())),
    }
}

fn typecheck_cast(
    t: &Type,
    inner: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let typed_inner = typecheck_exp(inner, symbol_table)?;
    Ok(TypedExpression(
        t.clone(),
        InnerTypedExpression::Cast(t.clone(), typed_inner.into()),
    ))
}

fn typecheck_unary(
    op: &UnaryOperator,
    inner: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let typed_inner = typecheck_exp(inner, symbol_table)?;
    let t = match op {
        UnaryOperator::Not => Type::Int,
        _ => typed_inner.0.clone(),
    };

    Ok(TypedExpression(
        t,
        InnerTypedExpression::Unary(op.clone(), Box::new(typed_inner)),
    ))
}

fn get_common_type(t1: Type, t2: Type) -> Type {
    // Simple, for now
    if t1 == t2 { t1 } else { Type::Long }
}

fn convert_to(e: TypedExpression, t: &Type) -> TypedExpression {
    if e.0 == *t {
        return e;
    }
    TypedExpression(
        t.clone(),
        InnerTypedExpression::Cast(t.clone(), Box::new(e)),
    )
}

fn typecheck_binary(
    op: &BinaryOperator,
    e1: &Expression,
    e2: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let typed_e1 = typecheck_exp(e1, symbol_table)?;
    let typed_e2 = typecheck_exp(e2, symbol_table)?;

    // No type conversions for AND / OR:
    if *op == BinaryOperator::And || *op == BinaryOperator::Or {
        return Ok(TypedExpression(
            Type::Int,
            InnerTypedExpression::Binary(op.clone(), Box::new(typed_e1), Box::new(typed_e2)),
        ));
    }

    let t1 = typed_e1.0.clone();
    let t2 = typed_e2.0.clone();
    let common_type = get_common_type(t1, t2);
    let converted_e1 = convert_to(typed_e1, &common_type);
    let converted_e2 = convert_to(typed_e2, &common_type);

    Ok(match op {
        BinaryOperator::Add
        | BinaryOperator::Subtract
        | BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::Remainder => TypedExpression(
            common_type.clone(),
            InnerTypedExpression::Binary(
                op.clone(),
                Box::new(converted_e1),
                Box::new(converted_e2),
            ),
        ),
        _ => TypedExpression(
            Type::Int,
            InnerTypedExpression::Binary(
                op.clone(),
                Box::new(converted_e1),
                Box::new(converted_e2),
            ),
        ),
    })
}

fn typecheck_assignment(
    left: &Expression,
    right: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let typed_left = typecheck_exp(left, symbol_table)?;
    let typed_right = typecheck_exp(right, symbol_table)?;
    let t_left = typed_left.0.clone();
    let converted_right = convert_to(typed_right, &t_left);

    Ok(TypedExpression(
        t_left,
        InnerTypedExpression::Assignment(Box::new(typed_left), Box::new(converted_right)),
    ))
}

fn typecheck_conditional(
    cond: &Expression,
    then_expr: &Expression,
    else_expr: &Expression,
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let typed_cond = typecheck_exp(cond, symbol_table)?;
    let typed_then = typecheck_exp(then_expr, symbol_table)?;
    let typed_else = typecheck_exp(else_expr, symbol_table)?;

    let t_then = typed_then.0.clone();
    let t_else = typed_else.0.clone();
    let common_type = get_common_type(t_then, t_else);
    let converted_then = convert_to(typed_then, &common_type);
    let converted_else = convert_to(typed_else, &common_type);

    Ok(TypedExpression(
        common_type.clone(),
        InnerTypedExpression::Conditional(
            Box::new(typed_cond),
            Box::new(converted_then),
            Box::new(converted_else),
        ),
    ))
}

fn typecheck_function_call(
    name: &Identifier,
    args: &[Expression],
    symbol_table: &SymbolTable,
) -> Result<TypedExpression, Error> {
    let f_type = symbol_table
        .get(name)
        .ok_or_else(|| Error::UndeclaredFunction(name.clone()))?;
    match &f_type.type_ {
        Type::Function { params, .. } => {
            if params.len() != args.len() {
                return Err(Error::MismatchedFunctionArguments(name.clone()));
            }

            let converted_args = args
                .iter()
                .zip(params.iter())
                .map(|(arg, param)| {
                    let typed_arg = typecheck_exp(arg, symbol_table)?;
                    Ok(convert_to(typed_arg, param))
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(TypedExpression(
                f_type.type_.clone(),
                InnerTypedExpression::FunctionCall(name.clone(), converted_args),
            ))
        }
        _ => Err(Error::InvalidFunctionCall(name.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::SymbolTableEntry;

    #[test]
    fn test_typecheck_exp() {
        let symbol_table = SymbolTable::new();
        let typed_exp = typecheck_exp(&Expression::Constant(Const::ConstInt(42)), &symbol_table);

        assert_eq!(
            Ok(TypedExpression(
                Type::Int,
                InnerTypedExpression::Constant(Const::ConstInt(42)),
            )),
            typed_exp
        );
    }

    #[test]
    fn test_typecheck_file_scope_variable() {
        let mut symbol_table = SymbolTable::new();
        let var_decl = VarDecl {
            name: Identifier::from("x"),
            var_type: Type::Long,
            init: Some(Expression::Constant(Const::ConstLong(10))),
            storage_class: Some(StorageClass::Static),
        };

        let typed_var_decl =
            typecheck_file_scope_variable_declaration(&var_decl, &mut symbol_table);

        assert_eq!(
            typed_var_decl,
            Ok(TypedVarDecl {
                name: Identifier::from("x"),
                init: None, // dropped
                var_type: Type::Long,
                storage_class: Some(StorageClass::Static),
            }),
        );
        assert_eq!(
            symbol_table.get(&Identifier::from("x")),
            Some(&SymbolTableEntry {
                type_: Type::Long,
                attrs: IdentifierAttrs::Static {
                    init: InitialValue::Initial(StaticInit::LongInit(10)),
                    global: false
                }
            })
        );
    }

    #[test]
    fn test_long_int_constant() {
        // Implementation-defined behaviour:
        //   static int i = 2147483650L;
        // should result in an IntInit(-2147483646)

        let mut symbol_table = SymbolTable::new();
        let var_decl = VarDecl {
            name: Identifier::from("i"),
            var_type: Type::Int,
            init: Some(Expression::Constant(Const::ConstLong(2147483650))),
            storage_class: Some(StorageClass::Static),
        };

        let typed_var_decl =
            typecheck_file_scope_variable_declaration(&var_decl, &mut symbol_table);

        assert_eq!(
            typed_var_decl,
            Ok(TypedVarDecl {
                name: Identifier::from("i"),
                init: None, // dropped
                var_type: Type::Int,
                storage_class: Some(StorageClass::Static),
            }),
        );
        assert_eq!(
            symbol_table.get(&Identifier::from("i")),
            Some(&SymbolTableEntry {
                type_: Type::Int,
                attrs: IdentifierAttrs::Static {
                    init: InitialValue::Initial(StaticInit::IntInit(-2147483646)),
                    global: false
                }
            })
        );
    }
}
