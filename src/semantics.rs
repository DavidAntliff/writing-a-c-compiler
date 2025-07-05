use crate::ast_c::{
    Block, BlockItem, Declaration, Expression, ForInit, FunDecl, Label, Program, Statement, VarDecl,
};
use crate::id_gen::IdGenerator;
use crate::lexer::Identifier;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum Error {
    #[error("Undeclared variable: {0}")]
    UndeclaredVariable(Identifier),

    #[error("Duplicate declaration: {0}")]
    DuplicateVariableDeclaration(Identifier),

    #[error("Invalid lvalue")]
    InvalidLValue,

    #[error("Undeclared label: {0}")]
    UndeclaredLabel(Label),

    #[error("Duplicate label: {0}")]
    DuplicateLabel(Label),

    #[error("Break statement outside of loop")]
    BreakOutsideLoop,

    #[error("Continue statement outside of loop")]
    ContinueOutsideLoop,

    #[error("Invalid function definition: {0}")]
    InvalidFunctionDefinition(Identifier),

    #[error("Undeclared function: {0}")]
    UndeclaredFunction(Identifier),

    #[error("Duplicate function parameter: {0}")]
    DuplicateFunctionParameter(Identifier),

    #[error("Invalid function call: {0}")]
    InvalidFunctionCall(Identifier),

    #[error("Duplicate function declaration: {0}")]
    DuplicateFunctionDeclaration(Identifier),

    #[error("Incompatible function declaration: {0}")]
    IncompatibleFunctionDeclaration(Identifier),

    #[error("Mismatched number of function arguments: {0}")]
    MismatchedFunctionArguments(Identifier),

    #[error("Multiple function definitions: {0}")]
    MultipleDefinitions(Identifier),

    #[error("{0}")]
    Message(String),
}

pub(crate) fn analyse(program: &mut Program) -> Result<SymbolTable, Error> {
    identifier_resolution(program)?;

    loop_labeling(program)?;

    let symbol_table = type_checking(program)?;

    let _label_table = goto_label_resolution(program)?;

    // TODO: combine the tables into one symbol table

    Ok(symbol_table)
}

#[derive(Debug, Clone)]
struct IdentifierMapEntry {
    unique_name: Identifier,
    from_current_scope: bool,
    has_linkage: bool,
}

type IdentifierMap = HashMap<Identifier, IdentifierMapEntry>;

fn identifier_resolution(program: &mut Program) -> Result<(), Error> {
    let mut identifier_map = IdentifierMap::new();
    let mut id_gen = IdGenerator::new();

    // TODO: for globals

    for function in &mut program.function_declarations {
        *function = resolve_function_declaration(function, &mut identifier_map, &mut id_gen)?;
    }

    Ok(())
}

fn unique_name(source_name: &str, id_gen: &mut IdGenerator) -> String {
    format!("{source_name}.{}", id_gen.next())
}

fn resolve_block(
    block: &Block,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<Block, Error> {
    let mut items = Vec::new();
    for item in &block.items {
        match item {
            BlockItem::S(statement) => {
                items.push(BlockItem::S(resolve_statement(
                    statement,
                    identifier_map,
                    id_gen,
                )?));
            }
            BlockItem::D(declaration) => {
                items.push(BlockItem::D(resolve_declaration(
                    declaration,
                    identifier_map,
                    id_gen,
                )?));
            }
        }
    }
    Ok(Block { items })
}

fn resolve_declaration(
    declaration: &Declaration,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<Declaration, Error> {
    match declaration {
        Declaration::FunDecl(fun_decl) => {
            if fun_decl.body.is_some() {
                Err(Error::InvalidFunctionDefinition(fun_decl.name.clone()))
            } else {
                Ok(Declaration::FunDecl(resolve_function_declaration(
                    fun_decl,
                    identifier_map,
                    id_gen,
                )?))
            }
        }
        Declaration::VarDecl(var_decl) => Ok(Declaration::VarDecl(resolve_variable_declaration(
            var_decl,
            identifier_map,
            id_gen,
        )?)),
    }
}

fn resolve_variable_declaration(
    VarDecl { name, init }: &VarDecl,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<VarDecl, Error> {
    if identifier_map.contains_key(name) && identifier_map[name].from_current_scope {
        return Err(Error::DuplicateVariableDeclaration(name.clone()));
    }

    let unique_name = unique_name(name, id_gen);
    identifier_map.insert(
        name.clone(),
        IdentifierMapEntry {
            unique_name: unique_name.clone(),
            from_current_scope: true,
            has_linkage: false, // local variables do not have linkage
        },
    );

    let init = if let Some(init) = init {
        Some(resolve_exp(init, identifier_map)?)
    } else {
        None
    };

    Ok(VarDecl {
        name: unique_name,
        init,
    })
}

fn resolve_function_declaration(
    FunDecl { name, params, body }: &FunDecl,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<FunDecl, Error> {
    if identifier_map.contains_key(name) {
        let prev_entry = &identifier_map.get(name).unwrap();
        if prev_entry.from_current_scope && !prev_entry.has_linkage {
            return Err(Error::DuplicateFunctionDeclaration(name.clone()));
        }
    }

    identifier_map.insert(
        name.clone(),
        IdentifierMapEntry {
            unique_name: name.clone(),
            from_current_scope: true,
            has_linkage: true,
        },
    );

    let mut inner_map = copy_identifier_map(identifier_map);
    let new_params = params
        .iter()
        .map(|param| resolve_param(param, &mut inner_map, id_gen))
        .collect::<Result<Vec<_>, _>>()?;

    let new_body = if let Some(body) = body {
        Some(resolve_block(body, &mut inner_map, id_gen)?)
    } else {
        None
    };

    Ok(FunDecl {
        name: name.clone(),
        params: new_params,
        body: new_body,
    })
}

fn resolve_param(
    name: &Identifier,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<Identifier, Error> {
    if identifier_map.contains_key(name) && identifier_map[name].from_current_scope {
        return Err(Error::DuplicateFunctionParameter(name.clone()));
    }

    let unique_name = unique_name(name, id_gen);
    identifier_map.insert(
        name.clone(),
        IdentifierMapEntry {
            unique_name: unique_name.clone(),
            from_current_scope: true,
            has_linkage: false, // parameters do not have linkage
        },
    );

    Ok(unique_name)
}

fn resolve_statement(
    statement: &Statement,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<Statement, Error> {
    match statement {
        Statement::Return(exp) => Ok(Statement::Return(resolve_exp(exp, identifier_map)?)),
        Statement::Expression(exp) => Ok(Statement::Expression(resolve_exp(exp, identifier_map)?)),
        Statement::If {
            condition,
            then,
            else_,
        } => {
            let else_ = if let Some(else_stmt) = else_ {
                Some(Box::new(resolve_statement(
                    else_stmt,
                    identifier_map,
                    id_gen,
                )?))
            } else {
                None
            };
            Ok(Statement::If {
                condition: resolve_exp(condition, identifier_map)?,
                then: Box::new(resolve_statement(then, identifier_map, id_gen)?),
                else_,
            })
        }
        Statement::Labeled { label, statement } => {
            resolve_statement(statement, identifier_map, id_gen).map(|stmt| Statement::Labeled {
                label: label.clone(),
                statement: Box::new(stmt),
            })
        }
        Statement::Goto(label) => Ok(Statement::Goto(label.clone())),
        Statement::Compound(block) => {
            let mut new_identifier_map = copy_identifier_map(identifier_map);
            Ok(Statement::Compound(resolve_block(
                block,
                &mut new_identifier_map,
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
            condition: resolve_exp(condition, identifier_map)?,
            body: Box::new(resolve_statement(body, identifier_map, id_gen)?),
            loop_label: loop_label.clone(),
        }),
        Statement::DoWhile {
            body,
            condition,
            loop_label,
        } => Ok(Statement::DoWhile {
            body: Box::new(resolve_statement(body, identifier_map, id_gen)?),
            condition: resolve_exp(condition, identifier_map)?,
            loop_label: loop_label.clone(),
        }),
        Statement::For {
            init,
            condition,
            post,
            body,
            loop_label,
        } => {
            let mut new_identifier_map = copy_identifier_map(identifier_map);
            let init = resolve_for_init(init, &mut new_identifier_map, id_gen)?;
            let condition = resolve_optional_exp(condition, &mut new_identifier_map)?;
            let post = resolve_optional_exp(post, &mut new_identifier_map)?;
            let body = resolve_statement(body, &mut new_identifier_map, id_gen)?;
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

fn copy_identifier_map(identifier_map: &IdentifierMap) -> IdentifierMap {
    // clone the hashmap but reset the `from_current_block` flag
    identifier_map
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                IdentifierMapEntry {
                    unique_name: v.unique_name.clone(),
                    from_current_scope: false,
                    has_linkage: v.has_linkage,
                },
            )
        })
        .collect()
}

fn resolve_exp(exp: &Expression, identifier_map: &mut IdentifierMap) -> Result<Expression, Error> {
    match exp {
        Expression::Constant(_) => Ok(exp.clone()),
        Expression::Var(v) => identifier_map
            .get(v)
            .map(|var| Expression::Var(var.unique_name.clone()))
            .ok_or_else(|| Error::UndeclaredVariable(v.clone()))
            .map(Ok)?,
        Expression::Unary(op, exp) => Ok(Expression::Unary(
            op.clone(),
            resolve_exp(exp, identifier_map)?.into(),
        )),
        Expression::Binary(op, left, right) => Ok(Expression::Binary(
            op.clone(),
            resolve_exp(left, identifier_map)?.into(),
            resolve_exp(right, identifier_map)?.into(),
        )),
        Expression::Assignment(left, right) => {
            if !matches!(**left, Expression::Var(_)) {
                return Err(Error::InvalidLValue);
            }
            Ok(Expression::Assignment(
                resolve_exp(left, identifier_map)?.into(),
                resolve_exp(right, identifier_map)?.into(),
            ))
        }
        Expression::Conditional(cond, then, else_) => Ok(Expression::Conditional(
            resolve_exp(cond, identifier_map)?.into(),
            resolve_exp(then, identifier_map)?.into(),
            resolve_exp(else_, identifier_map)?.into(),
        )),
        Expression::FunctionCall(fun_name, args) => identifier_map
            .get(fun_name)
            .map(|var| var.unique_name.clone())
            .ok_or_else(|| Error::UndeclaredFunction(fun_name.clone()))
            .and_then(|unique_name| {
                args.iter()
                    .map(|arg| resolve_exp(arg, identifier_map))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|resolved_args| Expression::FunctionCall(unique_name, resolved_args))
            }),
    }
}

fn resolve_for_init(
    init: &ForInit,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<ForInit, Error> {
    match init {
        ForInit::InitDecl(decl) => Ok(ForInit::InitDecl(resolve_variable_declaration(
            decl,
            identifier_map,
            id_gen,
        )?)),
        ForInit::InitExp(exp) => Ok(ForInit::InitExp(resolve_optional_exp(exp, identifier_map)?)),
    }
}

fn resolve_optional_exp(
    exp: &Option<Expression>,
    identifier_map: &mut IdentifierMap,
) -> Result<Option<Expression>, Error> {
    if let Some(exp) = exp {
        Ok(Some(resolve_exp(exp, identifier_map)?))
    } else {
        Ok(None)
    }
}

fn loop_labeling(program: &mut Program) -> Result<(), Error> {
    let mut identifier_map = IdentifierMap::new();
    let mut id_gen = IdGenerator::new();

    // TODO: for globals

    for function in &mut program.function_declarations {
        if let Some(body) = &mut function.body {
            *body = loop_label_block(body, &mut identifier_map, &mut id_gen)?;
        }
    }

    Ok(())
}

fn loop_label_block(
    block: &Block,
    identifier_map: &mut IdentifierMap,
    id_gen: &mut IdGenerator,
) -> Result<Block, Error> {
    let mut items = Vec::new();
    for item in &block.items {
        match item {
            BlockItem::S(statement) => {
                items.push(BlockItem::S(loop_label_statement(
                    statement,
                    None,
                    identifier_map,
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
    current_label: Option<Label>,
    _identifier_map: &mut IdentifierMap,
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
                _identifier_map,
                id_gen,
            )?),
            else_: if let Some(else_stmt) = else_ {
                Some(Box::new(loop_label_statement(
                    else_stmt,
                    current_label.clone(),
                    _identifier_map,
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
                _identifier_map,
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
                            _identifier_map,
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
                loop_label_statement(body, loop_label.clone(), _identifier_map, id_gen)?;
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
                loop_label_statement(body, loop_label.clone(), _identifier_map, id_gen)?;
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
                loop_label_statement(body, loop_label.clone(), _identifier_map, id_gen)?;
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

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Type {
    Int,
    Function { param_count: usize },
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct SymbolTableEntry {
    pub(crate) type_: Type,
    pub(crate) defined: bool,
}

#[derive(Debug)]
pub(crate) struct SymbolTable {
    table: HashMap<Identifier, SymbolTableEntry>,
}

impl SymbolTable {
    pub(crate) fn new() -> Self {
        SymbolTable {
            table: HashMap::new(),
        }
    }

    fn add(&mut self, identifier: Identifier, type_: Type, defined: bool) {
        self.table
            .insert(identifier, SymbolTableEntry { type_, defined });
    }

    pub(crate) fn get(&self, identifier: &Identifier) -> Option<&SymbolTableEntry> {
        self.table.get(identifier)
    }

    fn contains_key(&self, identifier: &Identifier) -> bool {
        self.table.contains_key(identifier)
    }
}

fn type_checking(program: &Program) -> Result<SymbolTable, Error> {
    let mut symbol_table = SymbolTable::new();

    for function_decl in &program.function_declarations {
        typecheck_function_declaration(function_decl, &mut symbol_table)?;
    }

    Ok(symbol_table)
}

fn typecheck_variable_declaration(
    var_decl: &VarDecl,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    // Variable names are already unique at this point
    assert!(
        !symbol_table.contains_key(&var_decl.name),
        "Variable already declared"
    );

    symbol_table.add(
        var_decl.name.clone(),
        Type::Int, // assuming all variables are of type Int for simplicity
        var_decl.init.is_some(),
    );
    if let Some(init) = &var_decl.init {
        typecheck_exp(init, symbol_table)?;
    }

    Ok(())
}

fn typecheck_function_declaration(
    function_decl: &FunDecl,
    symbol_table: &mut SymbolTable,
) -> Result<(), Error> {
    let fun_type = Type::Function {
        param_count: function_decl.params.len(),
    };
    let has_body = function_decl.body.is_some();
    let mut already_defined = false;

    if symbol_table.contains_key(&function_decl.name) {
        let existing_type = symbol_table.get(&function_decl.name).expect("should exist");
        if existing_type.type_ != fun_type {
            return Err(Error::IncompatibleFunctionDeclaration(
                function_decl.name.clone(),
            ));
        }
        already_defined = existing_type.defined;
        if already_defined && has_body {
            return Err(Error::MultipleDefinitions(function_decl.name.clone()));
        }
    }

    symbol_table.add(
        function_decl.name.clone(),
        fun_type,
        already_defined || has_body,
    );

    if let Some(body) = &function_decl.body {
        for param in function_decl.params.iter() {
            symbol_table.add(param.clone(), Type::Int, false);
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
            then,
            else_,
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
                typecheck_variable_declaration(var_decl, symbol_table)?;
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
            typecheck_variable_declaration(var_decl, symbol_table)?;
        }
        Declaration::FunDecl(fun_decl) => {
            typecheck_function_declaration(fun_decl, symbol_table)?;
        }
    }
    Ok(())
}

fn typecheck_exp(expression: &Expression, symbol_table: &mut SymbolTable) -> Result<(), Error> {
    match expression {
        Expression::Constant(_) => Ok(()),
        Expression::Var(name) => {
            let item = symbol_table.get(name);
            match item {
                Some(entry) if entry.type_ == Type::Int => Ok(()),
                Some(_) => Err(Error::InvalidFunctionCall(name.clone())),
                None => Err(Error::UndeclaredVariable(name.clone())),
            }
        }
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

// GOTO labels:

#[derive(Debug)]
pub(crate) struct LabelTableEntry {
    //function_name: String,
}

#[derive(Debug)]
pub(crate) struct LabelTable {
    table: HashMap<Label, LabelTableEntry>,
}

impl LabelTable {
    pub(crate) fn new() -> Self {
        LabelTable {
            table: HashMap::new(),
        }
    }

    fn add(&mut self, label: Label) -> Result<(), Error> {
        if self.table.contains_key(&label) {
            return Err(Error::DuplicateLabel(label));
        }
        self.table.insert(label, LabelTableEntry {});
        Ok(())
    }

    fn is_defined(&self, label: &Label) -> bool {
        self.table.contains_key(label)
    }
}

fn goto_label_resolution(program: &mut Program) -> Result<LabelTable, Error> {
    // Labels must be unique across the entire program, but within a function, labels
    // are renamed with the function name as a prefix. This ensures that labels remain unique
    // globally, but duplicate labels within a function are detected.
    let mut label_table = LabelTable::new();

    for function_decl in &mut program.function_declarations {
        // Use the function name as a prefix for all goto labels in this function
        let function_id = &function_decl.name;

        if let Some(body) = &mut function_decl.body {
            // AST is a tree, so we need to traverse it recursively,
            // building up the labels as we go
            for item in body.items.iter_mut() {
                if let BlockItem::S(statement) = item {
                    resolve_statement_goto(statement, function_id, &mut label_table)?;
                }
            }

            // Check for undeclared labels in Goto statements
            for item in &body.items {
                if let BlockItem::S(statement) = item {
                    check_statement_goto(statement, &label_table)?;
                }
            }
        }
    }

    Ok(label_table)
}

fn resolve_statement_goto(
    statement: &mut Statement,
    function_id: &str,
    label_table: &mut LabelTable,
) -> Result<(), Error> {
    fn resolve_goto_label(label: &Label, function_id: &str) -> String {
        format!("{label}.{function_id}")
    }

    match statement {
        Statement::Labeled { label, statement } => {
            let new_label = resolve_goto_label(label, function_id);
            label_table.add(new_label.clone())?;
            *label = new_label;
            resolve_statement_goto(statement, function_id, label_table)?;
        }
        Statement::If { then, else_, .. } => {
            resolve_statement_goto(then, function_id, label_table)?;
            if let Some(else_stmt) = else_ {
                resolve_statement_goto(else_stmt, function_id, label_table)?;
            }
        }
        Statement::Compound(block) => {
            // goto labels must be unique within a function, not a scope
            for item in &mut block.items {
                match item {
                    BlockItem::D(_) => continue, // declarations do not have labels
                    BlockItem::S(statement) => {
                        resolve_statement_goto(statement, function_id, label_table)?;
                    }
                }
            }
        }
        Statement::While { body, .. } => {
            resolve_statement_goto(body, function_id, label_table)?;
        }
        Statement::DoWhile { body, .. } => {
            resolve_statement_goto(body, function_id, label_table)?;
        }
        Statement::For { body, .. } => {
            resolve_statement_goto(body, function_id, label_table)?;
        }
        Statement::Goto(label) => {
            // Check later if the label is defined
            *label = resolve_goto_label(label, function_id);
        }
        Statement::Return(_) // explicit listing of all variants
        | Statement::Break(_)
        | Statement::Continue(_)
        | Statement::Expression(_)
        | Statement::Null => {}
    }
    Ok(())
}

fn check_statement_goto(statement: &Statement, label_table: &LabelTable) -> Result<(), Error> {
    // Check that all goto labels are defined in the label table

    match statement {
        Statement::Labeled { label, statement } => {
            if !label_table.is_defined(label) {
                return Err(Error::UndeclaredLabel(label.clone()));
            }
            check_statement_goto(statement, label_table)?;
        }
        Statement::Goto(label) => {
            if !label_table.is_defined(label) {
                return Err(Error::UndeclaredLabel(label.clone()));
            }
        }
        Statement::If { then, else_, .. } => {
            check_statement_goto(then, label_table)?;
            if let Some(else_stmt) = else_ {
                check_statement_goto(else_stmt, label_table)?;
            }
        }
        Statement::Compound(block) => {
            for item in &block.items {
                if let BlockItem::S(statement) = item {
                    check_statement_goto(statement, label_table)?;
                }
            }
        }
        Statement::While { body, .. } => {
            check_statement_goto(body, label_table)?;
        }
        Statement::DoWhile { body, .. } => {
            check_statement_goto(body, label_table)?;
        }
        Statement::For { body, .. } => {
            check_statement_goto(body, label_table)?;
        }
        Statement::Return(_)  // explicit listing of all variants
        | Statement::Break(_)
        | Statement::Continue(_)
        | Statement::Expression(_)
        | Statement::Null => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::{BinaryOperator, Block, FunDecl, VarDecl};
    use assert_matches::assert_matches;
    use assertables::assert_ok;

    #[test]
    fn test_analyse() {
        let mut program = Program {
            function_declarations: vec![FunDecl {
                name: "main".into(),
                params: vec![],
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

        assert!(identifier_resolution(&mut program).is_ok());

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
                params: vec![],
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
                params: vec![],
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
                params: vec![],
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
    fn test_goto_duplicate_label_in_different_functions() {
        let mut program = Program {
            function_declarations: vec![
                FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![
                            BlockItem::S(Statement::Goto("label1".into())),
                            BlockItem::S(Statement::Labeled {
                                label: "label1".into(),
                                statement: Box::new(Statement::Null),
                            }),
                        ],
                    }),
                },
                FunDecl {
                    name: "other".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![
                            BlockItem::S(Statement::Goto("label1".into())),
                            BlockItem::S(Statement::Labeled {
                                label: "label1".into(),
                                statement: Box::new(Statement::Null),
                            }),
                        ],
                    }),
                },
            ],
        };

        assert_ok!(analyse(&mut program));
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
                params: vec![],
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
                            &stmt, Statement::While { loop_label, .. } if *loop_label == Some("while.0".into())
            )
        );
        assert_matches!(
            &program.function_declarations[0].body.as_ref().unwrap().items[0],
            BlockItem::S(stmt) if matches!(
                &stmt, Statement::While { body, .. } if matches!(
                    &**body, Statement::Compound(block) if matches!(
                        &block.items[1], BlockItem::S(Statement::If { then, .. }) if matches!(
                            &**then, Statement::Break(loop_label) if
                                *loop_label == Some("while.0".into())
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
                        &block.items[0], BlockItem::S(Statement::For { loop_label, .. }) if *loop_label == Some("for.1".into())
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
                                *loop_label == Some("for.1".into())
                            ))
                        )
                    )
                )
            )
        );
    }
}
