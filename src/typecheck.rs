use crate::ast_c::{
    Block, BlockItem, Const, Declaration, Expression, ForInit, FunDecl, Program, Statement,
    StorageClass, TypedExpression, VarDecl,
};
use crate::semantics::{Error, IdentifierAttrs, InitialValue, SymbolTable, Type};

pub(crate) fn type_checking(program: &Program) -> Result<SymbolTable, Error> {
    let mut symbol_table = SymbolTable::new();

    for declaration in &program.declarations {
        match declaration {
            Declaration::FunDecl(fun_decl) => {
                typecheck_function_declaration(fun_decl, &mut symbol_table)?;
            }
            Declaration::VarDecl(var_decl) => {
                typecheck_file_scope_variable_declaration(var_decl, &mut symbol_table)?;
            }
        }
    }

    Ok(symbol_table)
}

fn typecheck_file_scope_variable_declaration(
    decl: &VarDecl,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    // Determine the variable's initial value
    let mut initial_value = match &decl.init {
        // TODO: other Const variants
        Some(Expression::Constant(Const::ConstInt(i))) => InitialValue::Initial(i64::from(*i)),
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
        if old_decl.type_ != Type::Int {
            return Err(Error::FunctionRedeclaredAsVariable(decl.name.clone()));
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
    symbol_table.add(decl.name.clone(), Type::Int, attrs);
    Ok(())
}

fn typecheck_local_variable_declaration(
    decl: &VarDecl,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    match decl.storage_class {
        Some(StorageClass::Extern) => {
            // Extern variables cannot have initialisers
            if decl.init.is_some() {
                return Err(Error::InitialiserOnLocalExternVariableDeclaration(
                    decl.name.clone(),
                ));
            }
            if let Some(old_decl) = symbol_table.get(&decl.name) {
                if old_decl.type_ != Type::Int {
                    return Err(Error::FunctionRedeclaredAsVariable(decl.name.clone()));
                }
            } else {
                symbol_table.add(
                    decl.name.clone(),
                    Type::Int,
                    IdentifierAttrs::Static {
                        init: InitialValue::NoInitialiser,
                        global: true,
                    },
                );
            }
        }
        Some(StorageClass::Static) => {
            // Static variables have no linkage, and must have an initialiser otherwise zero.
            let initial_value = match &decl.init {
                // TODO: other Const variants
                Some(Expression::Constant(Const::ConstInt(i))) => {
                    InitialValue::Initial(i64::from(*i))
                }
                None => InitialValue::Initial(0),
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
        }
        None => {
            // Automatic variable
            symbol_table.add(decl.name.clone(), Type::Int, IdentifierAttrs::Local);
            if let Some(init) = &decl.init {
                typecheck_exp(init, symbol_table)?;
            }
        }
    }

    Ok(())
}

fn typecheck_function_declaration(
    decl: &FunDecl,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    let fun_type = Type::Function {
        param_count: decl.params.len(),
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

    symbol_table.add(decl.name.clone(), fun_type, attrs);

    if let Some(body) = &decl.body {
        for param in decl.params.iter() {
            symbol_table.add(param.clone(), Type::Int, IdentifierAttrs::Local);
        }
        typecheck_block(body, symbol_table)?;
    }
    Ok(())
}

fn typecheck_block(block: &Block, symbol_table: &mut SymbolTable) -> Result<(), Error> {
    for item in &block.items {
        match item {
            BlockItem::S(statement) => typecheck_statement(statement, symbol_table)?,
            BlockItem::D(declaration) => typecheck_declaration(declaration, symbol_table)?,
        }
    }
    Ok(())
}

fn typecheck_statement(statement: &Statement, symbol_table: &mut SymbolTable) -> Result<(), Error> {
    match statement {
        Statement::Return(exp) => {
            typecheck_exp(exp, symbol_table)?;
        }
        Statement::Expression(exp) => {
            typecheck_exp(exp, symbol_table)?;
        }
        Statement::If {
            condition,
            then_block: then,
            else_block: else_,
        } => {
            typecheck_exp(condition, symbol_table)?;
            typecheck_statement(then, symbol_table)?;
            if let Some(else_stmt) = else_ {
                typecheck_statement(else_stmt, symbol_table)?;
            }
        }
        Statement::Labeled { statement, .. } => {
            // Labels are not checked here, they are checked in goto_label_resolution
            typecheck_statement(statement, symbol_table)?;
        }
        Statement::Goto(_) => {
            // Goto labels are checked in goto_label_resolution
        }
        Statement::Compound(block) => {
            typecheck_block(block, symbol_table)?;
        }
        Statement::Break(_) | Statement::Continue(_) => {
            // Break and continue do not have expressions to check
        }
        Statement::While {
            condition, body, ..
        } => {
            typecheck_exp(condition, symbol_table)?;
            typecheck_statement(body, symbol_table)?;
        }
        Statement::DoWhile {
            body, condition, ..
        } => {
            typecheck_statement(body, symbol_table)?;
            typecheck_exp(condition, symbol_table)?;
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            loop_label: _,
        } => {
            if let ForInit::InitDecl(var_decl) = init {
                // Storage-class specifiers not allowed in for-init-decls
                if var_decl.storage_class.is_some() {
                    return Err(Error::InvalidStorageClass(
                        var_decl.storage_class.clone().unwrap(),
                    ));
                }
                typecheck_local_variable_declaration(var_decl, symbol_table)?;
            } else if let ForInit::InitExp(Some(exp)) = init {
                typecheck_exp(exp, symbol_table)?;
            }
            if let Some(cond) = condition {
                typecheck_exp(cond, symbol_table)?;
            }
            if let Some(post_exp) = post {
                typecheck_exp(post_exp, symbol_table)?;
            }
            typecheck_statement(body, symbol_table)?;
        }
        Statement::Null => {}
    }
    Ok(())
}

fn typecheck_declaration(
    declaration: &Declaration,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    match declaration {
        Declaration::VarDecl(var_decl) => {
            typecheck_local_variable_declaration(var_decl, symbol_table)?;
        }
        Declaration::FunDecl(fun_decl) => {
            typecheck_function_declaration(fun_decl, symbol_table)?;
        }
    }
    Ok(())
}

fn typecheck_exp(
    expression: &Expression,
    symbol_table: &mut SymbolTable,
) -> Result<TypedExpression, Error> {
    match expression {
        Expression::Constant(c) => Ok(expression.with_type(c)),
        Expression::Var(name) => {
            let item = symbol_table.get(name);
            match item {
                Some(entry) if entry.type_ == Type::Int => Ok(()),
                Some(_) => Err(Error::InvalidFunctionCall(name.clone())),
                None => Err(Error::UndeclaredVariable(name.clone())),
            }
        }
        Expression::Cast(_, _) => todo!(),
        Expression::Unary(_, exp) => typecheck_exp(exp, symbol_table),
        Expression::Binary(_, left, right) => {
            typecheck_exp(left, symbol_table)?;
            typecheck_exp(right, symbol_table)
        }
        Expression::Assignment(left, right) => {
            if !matches!(**left, Expression::Var(_)) {
                return Err(Error::InvalidLValue);
            }
            typecheck_exp(left, symbol_table)?;
            typecheck_exp(right, symbol_table)
        }
        Expression::Conditional(cond, then, else_) => {
            typecheck_exp(cond, symbol_table)?;
            typecheck_exp(then, symbol_table)?;
            typecheck_exp(else_, symbol_table)
        }
        Expression::FunctionCall(fun_name, args) => {
            let f_type = symbol_table
                .get(fun_name)
                .ok_or_else(|| Error::UndeclaredFunction(fun_name.clone()))?;
            match f_type.type_ {
                Type::Int => Err(Error::InvalidFunctionCall(fun_name.clone())),
                Type::Function { param_count } => {
                    if param_count != args.len() {
                        return Err(Error::MismatchedFunctionArguments(fun_name.clone()));
                    }
                    for arg in args {
                        typecheck_exp(arg, symbol_table)?;
                    }
                    Ok(())
                }
            }
        }
    }
}
